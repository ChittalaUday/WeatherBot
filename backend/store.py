"""
Every turn of every conversation, in SQLite, so real usage can retrain the model.

    python -m backend.store --stats
    python -m backend.store --export data/from_users.csv    # gold rows for the next build

Honest naming: this is not reinforcement learning. There is no reward signal and no policy -
it is the same supervised loop MODEL_RULES.md already describes, fed by real users instead of
templates: collect turns, keep the ones a human labelled, fold them into `data/intents.csv`,
rebuild, retrain, and check the frozen hand-written eval set still improves.

The labels worth trusting, in order:
  1. `choice`     - the user picked an intent from a clarify prompt. A gold label, free.
  2. `correction` - the user said what it should have been.
  3. `down`       - something was wrong, but not what. Triage material, not a label.
  4. `up`         - the prediction was right. Only useful above the confidence floor, since
                    an approved guess is still a guess.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "conversations.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    session       TEXT NOT NULL,   -- one websocket connection
    chat_id       TEXT,            -- one conversation: survives reloads and reconnects
    turn          INTEGER,         -- position within the chat, 0-based
    text          TEXT NOT NULL,
    intent        TEXT,
    action        TEXT,
    confidence    REAL,
    location      TEXT,           -- json: raw spans (Rule 4.1)
    time_raw      TEXT,           -- json: raw spans (Rule 4.2)
    time_norm     TEXT,           -- json: canonical forms (Rule 4.3)
    places        TEXT,           -- json: resolved lat/lng from Solr or the browser
    unresolved    TEXT,           -- json: names the location index could not find
    outcome       TEXT NOT NULL,  -- answered | clarified | need_location | error
    detail        TEXT,           -- summary, clarify message, or error text
    latency_ms    INTEGER,
    scores        TEXT,           -- json: full intent probability vector
    operation     TEXT,           -- SET | REPLACE | MODIFY | INHERIT | COMPARE
    normalized    TEXT            -- what the model actually saw, after src/normalize.py
);
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    turn_id       INTEGER NOT NULL REFERENCES turns(id),
    ts            TEXT NOT NULL,
    kind          TEXT NOT NULL,  -- up | down | correction | choice
    intent        TEXT,
    action        TEXT,
    location      TEXT,
    time_raw      TEXT,
    error_type    TEXT,           -- intent_confusion | vocabulary_gap | context_required |
                                  -- location_resolution | time_resolution | other
    note          TEXT
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session);
CREATE INDEX IF NOT EXISTS turns_chat ON turns(chat_id);
CREATE INDEX IF NOT EXISTS turns_outcome ON turns(outcome);
CREATE INDEX IF NOT EXISTS feedback_turn ON feedback(turn_id);
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")     # chat writes must not block reads
    connection.executescript(SCHEMA)
    return connection


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_turn(connection, session, text, *, chat_id=None, turn=None, intent=None,
                action=None, confidence=None,
                location=None, time_raw=None, time_norm=None, places=None, unresolved=None,
                outcome="answered", detail=None, latency_ms=None, scores=None,
                operation=None, normalized=None) -> int:
    """One turn, with the full score vector - knowing *which* intents competed is what makes
    a failed prediction useful later."""
    cursor = connection.execute(
        """INSERT INTO turns (ts, session, chat_id, turn, text, intent, action, confidence,
                              location, time_raw, time_norm, places, unresolved, outcome,
                              detail, latency_ms, scores, operation, normalized)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (_now(), session, chat_id, turn, text, intent, action, confidence,
         json.dumps(location or []), json.dumps(time_raw or []), json.dumps(time_norm or []),
         json.dumps(places or []), json.dumps(unresolved or []), outcome, detail, latency_ms,
         json.dumps(scores or {}), operation, normalized),
    )
    connection.commit()
    return cursor.lastrowid


def record_feedback(connection, turn_id, kind, *, intent=None, action=None,
                    location=None, time_raw=None, note=None, error_type=None) -> int:
    cursor = connection.execute(
        """INSERT INTO feedback (turn_id, ts, kind, intent, action, location, time_raw,
                                 error_type, note)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (turn_id, _now(), kind, intent, action,
         json.dumps(location) if location is not None else None,
         json.dumps(time_raw) if time_raw is not None else None, error_type, note),
    )
    connection.commit()
    return cursor.lastrowid


def conversation(connection, chat_id: str) -> list[dict]:
    """Every turn of one chat, in order - what the user actually asked, end to end."""
    return [dict(row) for row in connection.execute(
        "SELECT * FROM turns WHERE chat_id = ? ORDER BY turn, id", (chat_id,))]


def stats(connection) -> dict:
    row = connection.execute(
        """SELECT COUNT(*) turns, COUNT(DISTINCT session) sessions,
                  COUNT(DISTINCT chat_id) chats,
                  AVG(confidence) confidence, AVG(latency_ms) latency FROM turns""").fetchone()
    outcomes = {r["outcome"]: r["n"] for r in
                connection.execute("SELECT outcome, COUNT(*) n FROM turns GROUP BY outcome")}
    kinds = {r["kind"]: r["n"] for r in
             connection.execute("SELECT kind, COUNT(*) n FROM feedback GROUP BY kind")}
    return {
        "turns": row["turns"], "sessions": row["sessions"], "chats": row["chats"],
        "mean_confidence": round(row["confidence"], 3) if row["confidence"] else None,
        "mean_latency_ms": round(row["latency"]) if row["latency"] else None,
        "outcomes": outcomes, "feedback": kinds,
    }


def confusion(connection) -> dict:
    """Predicted-vs-actual counts over human-labelled turns.

    Accuracy alone hides which pair is the problem; this shows whether FORECAST keeps
    swallowing CURRENT_CONDITIONS, which is what tells you what to write next.
    """
    pairs = Counter()
    for row in connection.execute(
            """SELECT t.intent predicted, f.intent actual FROM feedback f
               JOIN turns t ON t.id = f.turn_id
               WHERE f.kind IN ('choice', 'correction') AND f.intent IS NOT NULL"""):
        pairs[(row["actual"], row["predicted"])] += 1
    return {"pairs": pairs,
            "worst": [f"{actual} -> {predicted} ({count}x)"
                      for (actual, predicted), count in pairs.most_common(5)
                      if actual != predicted]}


def competing_intents(connection, limit: int = 10) -> list[tuple[str, float, str]]:
    """Turns where two intents were close - the boundary cases worth labelling first."""
    out = []
    for row in connection.execute(
            "SELECT text, scores FROM turns WHERE scores IS NOT NULL AND scores != '{}'"):
        scores = sorted(json.loads(row["scores"]).items(), key=lambda kv: -kv[1])
        if len(scores) > 1 and scores[0][1] - scores[1][1] < 0.25:
            out.append((row["text"], scores[0][1] - scores[1][1],
                        f"{scores[0][0]} {scores[0][1]:.2f} vs {scores[1][0]} {scores[1][1]:.2f}"))
    return sorted(out, key=lambda item: item[1])[:limit]


def training_rows(connection, include_approved=False, min_confidence=0.9) -> list[dict]:
    """Human-labelled turns, in the exact schema of data/intents.csv.

    A `choice` or `correction` supplies the label. `up` is only taken when the model was
    already confident, and never by default: approving a guess does not make it evidence.
    """
    wanted = ["choice", "correction"] + (["up"] if include_approved else [])
    placeholders = ",".join("?" * len(wanted))
    rows, seen = [], set()
    for record in connection.execute(
        f"""SELECT t.text, t.intent t_intent, t.action t_action, t.location t_location,
                   t.time_raw t_time, t.confidence,
                   f.kind, f.intent f_intent, f.action f_action, f.location f_location,
                   f.time_raw f_time
            FROM feedback f JOIN turns t ON t.id = f.turn_id
            WHERE f.kind IN ({placeholders})
            ORDER BY f.id""", wanted):
        if record["kind"] == "up" and (record["confidence"] or 0) < min_confidence:
            continue
        key = record["text"].strip().lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "text": record["text"],
            "weather_intent": record["f_intent"] or record["t_intent"],
            "action": record["f_action"] or record["t_action"],
            "location": record["f_location"] or record["t_location"] or "[]",
            "time": record["f_time"] or record["t_time"] or "[]",
        })
    return [r for r in rows if r["weather_intent"] and r["action"]]


def export(connection, path: Path, include_approved=False) -> int:
    rows = training_rows(connection, include_approved)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "weather_intent", "action", "location", "time"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def demo():
    """Self-check: a turn plus a human label must come back out as a training row."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        connection = connect(Path(tmp) / "check.db")
        turn = record_turn(connection, "s1", "angara vs hyderbad", intent="TEMPERATURE",
                           action="COMPARE", confidence=0.25,
                           location=["angara", "hyderbad"], time_raw=[], time_norm=[],
                           outcome="clarified", detail="low confidence")
        record_feedback(connection, turn, "choice", intent="RAIN", action="COMPARE",
                        error_type="intent_confusion")

        rows = training_rows(connection)
        assert len(rows) == 1, rows
        assert rows[0]["weather_intent"] == "RAIN", rows          # the human label wins
        assert json.loads(rows[0]["location"]) == ["angara", "hyderbad"], rows

        # an approved guess is not evidence unless the model was already confident
        low = record_turn(connection, "s1", "wind", intent="WIND_SPEED", action="GET",
                          confidence=0.30, outcome="answered")
        record_feedback(connection, low, "up", intent="WIND_SPEED", action="GET")
        assert len(training_rows(connection, include_approved=True)) == 1, "low-confidence up leaked"
        assert stats(connection)["turns"] == 2
        print("store demo OK:", rows[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--export", metavar="CSV")
    parser.add_argument("--include-approved", action="store_true",
                        help="also export thumbs-up turns the model was already confident about")
    parser.add_argument("--recent", type=int, metavar="N", help="print the last N turns")
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--confusion", action="store_true", help="predicted vs actual, from labels")
    parser.add_argument("--competing", action="store_true", help="turns where intents were close")
    args = parser.parse_args()

    if args.selfcheck:
        return demo()

    connection = connect(args.db)
    if args.stats or not (args.export or args.recent):
        for key, value in stats(connection).items():
            print(f"  {key:16s} {value}")
    if args.recent:
        print()
        for row in connection.execute(
                "SELECT * FROM turns ORDER BY id DESC LIMIT ?", (args.recent,)):
            print(f"  [{row['id']}] {row['outcome']:13s} {row['intent'] or '-':18s} "
                  f"conf {row['confidence'] or 0:.2f}  {row['text'][:52]!r}")
    if args.confusion:
        report = confusion(connection)
        print("\nconfusion (actual -> predicted):")
        for (actual, predicted), count in sorted(report["pairs"].items(), key=lambda kv: -kv[1]):
            mark = "  " if actual == predicted else " <-"
            print(f"  {actual:20s} -> {predicted:20s} {count:3d}{mark}")
        print("worst pairs:", report["worst"] or "none")
    if args.competing:
        print("\nclosest calls (label these first):")
        for text, margin, detail in competing_intents(connection):
            print(f"  margin {margin:.2f}  {text[:48]!r}  {detail}")
    if args.export:
        print(f"\nwrote {export(connection, args.export, args.include_approved)} labelled rows "
              f"-> {args.export}")


if __name__ == "__main__":
    main()
