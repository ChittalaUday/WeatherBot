"""
Every turn of every conversation, in SQLite, so real usage can retrain the model.

    python -m backend.store --stats
    python -m backend.store --export data/from_users.csv    # gold rows for the next build
    python tests/test_store_units.py                        # the checks for this module

Not reinforcement learning: the same supervised loop MODEL_RULES.md describes, fed by real
users instead of templates. Collect turns, keep the ones a human labelled, fold them into
`data/intents.csv`, rebuild, retrain, check the frozen eval set still improves.

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
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from backend.config import DB_PATH

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
    normalized    TEXT,           -- what the model actually saw, after src/normalize.py
    payload       TEXT,           -- json: the answer as rendered, so history replays exactly
    state         TEXT            -- json: ConversationState after the turn, to resume context
);
CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    -- one row per turn: a rating is the user's current opinion, not a log of clicks.
    -- Changing your mind updates this row; it never appends a second one.
    turn_id       INTEGER NOT NULL UNIQUE REFERENCES turns(id),
    ts            TEXT NOT NULL,   -- first rated
    updated_at    TEXT,            -- last changed
    revisions     INTEGER NOT NULL DEFAULT 0,
    kind          TEXT NOT NULL,  -- up | down | correction | choice
    intent        TEXT,
    action        TEXT,
    variables     TEXT,           -- json list: what v2 should have extracted
    location      TEXT,
    time_raw      TEXT,
    model         TEXT,           -- which version was corrected
    error_type    TEXT,           -- intent_confusion | vocabulary_gap | context_required |
                                  -- location_resolution | time_resolution | other
    note          TEXT
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session);
CREATE INDEX IF NOT EXISTS turns_chat ON turns(chat_id);
CREATE INDEX IF NOT EXISTS turns_outcome ON turns(outcome);
CREATE UNIQUE INDEX IF NOT EXISTS feedback_turn ON feedback(turn_id);
"""


def _migrate(connection) -> None:
    """Older databases allowed several feedback rows per turn. Keep the newest of each and
    add the uniqueness the schema now assumes."""
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(feedback)")}
    if not columns:
        return
    for column, ddl in (("updated_at", "TEXT"), ("revisions", "INTEGER NOT NULL DEFAULT 0")):
        if column not in columns:
            connection.execute(f"ALTER TABLE feedback ADD COLUMN {column} {ddl}")
    duplicates = connection.execute(
        """DELETE FROM feedback WHERE id NOT IN
           (SELECT MAX(id) FROM feedback GROUP BY turn_id)""").rowcount
    if duplicates:
        print(f"store: collapsed {duplicates} duplicate feedback rows")
    connection.execute("DROP INDEX IF EXISTS feedback_turn")
    connection.execute("CREATE UNIQUE INDEX IF NOT EXISTS feedback_turn ON feedback(turn_id)")
    connection.commit()


def _open(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")     # chat writes must not block reads
    connection.executescript(SCHEMA)
    _migrate(connection)
    return connection


BROKEN = ("malformed", "not a database", "disk image", "corrupt")


class _Rows:
    """A cursor's results, already read, so nothing touches the connection after the lock.

    `execute(...).fetchone()` looks atomic and is not - the fetch goes back to the connection
    after the lock would have been released.
    """

    __slots__ = ("rows", "lastrowid", "rowcount")

    def __init__(self, cursor):
        self.rows = cursor.fetchall()
        self.lastrowid = cursor.lastrowid
        self.rowcount = cursor.rowcount

    def fetchone(self):
        return self.rows[0] if self.rows else None

    def fetchall(self):
        return self.rows

    def __iter__(self):
        return iter(self.rows)

    def __len__(self):
        return len(self.rows)


class Resilient:
    """A connection that survives its file being replaced or going bad underneath it.

    A swapped file leaves the open handle mapping the old image; a corrupt one turns the
    frontend's per-turn polling into a wall of 500s. Either way: reopen once and retry, and if
    the file itself is the problem move it aside - never delete it, it is the only copy.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._connection = _open(self.path)
        # sqlite3.threadsafety is 1 here: threads may share the module, not connections.
        # FastAPI runs sync endpoints in worker threads, so one shared connection was used from
        # several at once. check_same_thread=False turns off the warning, not the hazard - it
        # corrupted the database one night and segfaulted the interpreter the next.
        self._lock = threading.RLock()

    def _quarantine(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        for suffix in ("", "-wal", "-shm"):
            broken = Path(str(self.path) + suffix)
            if broken.exists():
                broken.rename(f"{self.path}.corrupt-{stamp}{suffix}")
        print(f"store: {self.path.name} was unreadable - moved aside as "
              f"{self.path.name}.corrupt-{stamp}, starting a fresh one")

    def _reopen(self) -> None:
        try:
            self._connection.close()
        except sqlite3.Error:
            pass
        try:
            self._connection = _open(self.path)
            self._connection.execute("SELECT 1 FROM turns LIMIT 1").fetchone()
            return                                    # the file was only swapped, not broken
        except sqlite3.DatabaseError:
            self._quarantine()
            self._connection = _open(self.path)

    def execute(self, *args, **kwargs) -> _Rows:
        with self._lock:
            try:
                return _Rows(self._connection.execute(*args, **kwargs))
            except sqlite3.DatabaseError as exc:
                if not any(word in str(exc).lower() for word in BROKEN):
                    raise
                self._reopen()
                return _Rows(self._connection.execute(*args, **kwargs))

    def executescript(self, *args, **kwargs):
        with self._lock:
            return self._connection.executescript(*args, **kwargs)

    def commit(self):
        with self._lock:
            return self._connection.commit()

    def close(self):
        with self._lock:
            return self._connection.close()

    def __getattr__(self, name):                      # cursor, row_factory, in_transaction ...
        return getattr(self._connection, name)


def healthy(path: Path | str = DB_PATH) -> bool:
    """True when the file on disk passes SQLite's own integrity check."""
    path = Path(path)
    if not path.exists():
        return True                                   # nothing to be wrong yet
    try:
        with sqlite3.connect(path) as probe:
            return probe.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    except sqlite3.DatabaseError:
        return False


def connect(path: Path | str = DB_PATH) -> Resilient:
    return Resilient(Path(path))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_chat_id() -> str:
    """A fresh conversation id. The client keeps it, so a reload resumes the same chat."""
    return f"chat-{uuid.uuid4().hex[:10]}"


def record_turn(connection, session, text, *, chat_id=None, turn=None, intent=None,
                action=None, confidence=None,
                location=None, time_raw=None, time_norm=None, places=None, unresolved=None,
                outcome="answered", detail=None, latency_ms=None, scores=None,
                operation=None, normalized=None, payload=None, state=None) -> int:
    """One turn, with the full score vector - knowing *which* intents competed is what makes
    a failed prediction useful later."""
    cursor = connection.execute(
        """INSERT INTO turns (ts, session, chat_id, turn, text, intent, action, confidence,
                              location, time_raw, time_norm, places, unresolved, outcome,
                              detail, latency_ms, scores, operation, normalized,
                              payload, state)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (_now(), session, chat_id, turn, text, intent, action, confidence,
         json.dumps(location or []), json.dumps(time_raw or []), json.dumps(time_norm or []),
         json.dumps(places or []), json.dumps(unresolved or []), outcome, detail, latency_ms,
         json.dumps(scores or {}), operation, normalized,
         json.dumps(payload) if payload is not None else None,
         json.dumps(state) if state is not None else None),
    )
    connection.commit()
    return cursor.lastrowid


def record_feedback(connection, turn_id, kind, *, intent=None, action=None, variables=None,
                    location=None, time_raw=None, note=None, error_type=None,
                    model=None) -> int:
    """The user's current verdict on one turn - inserted once, updated thereafter.

    A thumbs-down followed by a correction is one opinion refined, so it replaces the earlier
    row; otherwise the same turn trains twice, once with a label and once without. Fields left
    as None keep what the previous revision held.
    """
    connection.execute(
        """INSERT INTO feedback (turn_id, ts, updated_at, revisions, kind, intent, action,
                                 variables, location, time_raw, model, error_type, note)
           VALUES (?,?,?,0,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(turn_id) DO UPDATE SET
               updated_at = excluded.updated_at,
               revisions  = feedback.revisions + 1,
               kind       = excluded.kind,
               intent     = COALESCE(excluded.intent, feedback.intent),
               action     = COALESCE(excluded.action, feedback.action),
               variables  = COALESCE(excluded.variables, feedback.variables),
               location   = COALESCE(excluded.location, feedback.location),
               time_raw   = COALESCE(excluded.time_raw, feedback.time_raw),
               model      = COALESCE(excluded.model, feedback.model),
               error_type = COALESCE(excluded.error_type, feedback.error_type),
               note       = COALESCE(excluded.note, feedback.note)""",
        (turn_id, _now(), _now(), kind, intent, action,
         json.dumps(variables) if variables is not None else None,
         json.dumps(location) if location is not None else None,
         json.dumps(time_raw) if time_raw is not None else None, model, error_type, note),
    )
    connection.commit()
    row = connection.execute("SELECT id FROM feedback WHERE turn_id = ?", (turn_id,)).fetchone()
    return row["id"]


def feedback_for(connection, turn_id: int) -> dict | None:
    """The current verdict on a turn, so the UI can show what was already said."""
    row = connection.execute("SELECT * FROM feedback WHERE turn_id = ?", (turn_id,)).fetchone()
    return dict(row) if row else None


def attach_payload(connection, turn_id: int, payload: dict) -> None:
    """Store the answer once its turn_id exists, so a replayed answer can still be rated.

    The payload has to carry the turn_id it belongs to; without it, a thumbs-up on a chat
    reopened from history has nothing to attach itself to.
    """
    connection.execute("UPDATE turns SET payload = ? WHERE id = ?",
                       (json.dumps(payload), turn_id))
    connection.commit()


def list_chats(connection, limit: int = 40) -> list[dict]:
    """Recent conversations, newest first, titled by their opening question."""
    rows = connection.execute(
        """SELECT chat_id,
                  COUNT(*)                                   AS turns,
                  MIN(ts)                                    AS started,
                  MAX(ts)                                    AS last_active,
                  SUM(outcome IN ('answered', 'uncertain'))  AS answered
           FROM turns
           WHERE chat_id IS NOT NULL
           GROUP BY chat_id
           ORDER BY MAX(ts) DESC
           LIMIT ?""", (limit,)).fetchall()

    chats = []
    for row in rows:
        opener = connection.execute(
            "SELECT text FROM turns WHERE chat_id = ? ORDER BY id LIMIT 1",
            (row["chat_id"],)).fetchone()
        title = (opener["text"] if opener else "").strip()
        model = "legacy"          # pre-dates the version tag; every new turn carries one
        if title.startswith("[") and "]" in title:            # logged as "[v3] question"
            model, title = title[1:title.index("]")], title[title.index("]") + 1:].strip()
        chats.append({
            "chat_id": row["chat_id"], "title": title[:80] or "(empty)",
            "turns": row["turns"], "answered": row["answered"],
            "started": row["started"], "last_active": row["last_active"], "model": model,
        })
    return chats


def last_state(connection, chat_id: str) -> dict | None:
    """The slot state after the most recent answered turn - used to resume a chat whose
    in-memory state died with a restart."""
    row = connection.execute(
        """SELECT state FROM turns
           WHERE chat_id = ? AND state IS NOT NULL
           ORDER BY id DESC LIMIT 1""", (chat_id,)).fetchone()
    return json.loads(row["state"]) if row and row["state"] else None


def conversation(connection, chat_id: str) -> list[dict]:
    """Every turn of one chat, in order - what the user actually asked, end to end."""
    turns = []
    for row in connection.execute(
            "SELECT * FROM turns WHERE chat_id = ? ORDER BY id", (chat_id,)):
        record = dict(row)
        for column in ("payload", "scores", "state", "location", "time_raw", "time_norm"):
            if record.get(column):
                try:
                    record[column] = json.loads(record[column])
                except (TypeError, ValueError):
                    pass
        text = record.get("text") or ""
        if text.startswith("[") and "]" in text:              # strip the logged model tag
            record["model"] = text[1:text.index("]")]
            record["text"] = text[text.index("]") + 1:].strip()
        turns.append(record)
    return turns


def recent_exchanges(connection, chat_id: str, limit: int = 3) -> list[tuple]:
    """The last few (question, answer) pairs of one chat, oldest first.

    Only turns that actually answered: a "which place did you mean?" in the history reads to a
    small model as a question it still owes a reply to. The "[v4] " tag is stripped - in a
    prompt it is one more thing to copy.
    """
    rows = connection.execute(
        "SELECT text, detail FROM turns WHERE chat_id = ? AND outcome IN "
        "('answered', 'uncertain') AND detail != '' ORDER BY id DESC LIMIT ?",
        (chat_id, limit)).fetchall()
    out = []
    for row in reversed(list(rows)):
        asked = row["text"] or ""
        if asked.startswith("[") and "]" in asked:
            asked = asked[asked.index("]") + 1:].strip()
        out.append((asked, row["detail"]))
    return out


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
    """Predicted-vs-actual counts over human-labelled turns - accuracy alone hides which pair
    is the problem, and the pair is what tells you what to write next."""
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
    """Human-labelled turns, ready to append to data/intents.csv or feed src.v2.dataset.

    A `choice` or `correction` supplies the label; the prediction fills what the human did not
    touch. `up` is only taken when the model was already confident - approving a guess does
    not make it evidence.
    """
    wanted = ["choice", "correction"] + (["up"] if include_approved else [])
    placeholders = ",".join("?" * len(wanted))
    rows, seen = [], set()
    for record in connection.execute(
        f"""SELECT t.text, t.intent t_intent, t.action t_action, t.location t_location,
                   t.time_raw t_time, t.confidence,
                   f.kind, f.intent f_intent, f.action f_action, f.variables f_variables,
                   f.location f_location, f.time_raw f_time, f.model, f.error_type, f.note
            FROM feedback f JOIN turns t ON t.id = f.turn_id
            WHERE f.kind IN ({placeholders})
            ORDER BY f.id""", wanted):
        if record["kind"] == "up" and (record["confidence"] or 0) < min_confidence:
            continue
        # the logged text carries a "[v2] " tag; training data must not
        text = (record["text"] or "").strip()
        if text.startswith("[") and "]" in text:
            text = text[text.index("]") + 1:].strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        rows.append({
            "text": text,
            "weather_intent": record["f_intent"] or record["t_intent"],
            "action": record["f_action"] or record["t_action"],
            "variables": record["f_variables"] or None,
            "location": record["f_location"] or record["t_location"] or "[]",
            "time": record["f_time"] or record["t_time"] or "[]",
            "source_kind": record["kind"],
            "model": record["model"],
            "error_type": record["error_type"],
        })
    return [r for r in rows if r["weather_intent"] and r["action"]]


def review_queue(connection, limit: int = 50) -> list[dict]:
    """Turns a human flagged wrong but did not correct, plus every uncertain turn nobody
    judged. This is the labelling worklist."""
    return [dict(row) for row in connection.execute(
        """SELECT t.id, t.text, t.intent, t.action, t.confidence, t.outcome,
                  MAX(CASE WHEN f.kind = 'down' THEN 1 ELSE 0 END) AS flagged
           FROM turns t
           LEFT JOIN feedback f ON f.turn_id = t.id
           GROUP BY t.id
           HAVING flagged = 1
               OR (t.outcome IN ('uncertain', 'clarified')
                   AND SUM(CASE WHEN f.kind IN ('choice','correction','up') THEN 1 ELSE 0 END) = 0)
           ORDER BY flagged DESC, t.confidence ASC
           LIMIT ?""", (limit,))]


def failed_turns(connection, limit: int = 500) -> list[dict]:
    """Every turn that did not answer, with what the model read and why it stopped.

    The cheapest labels in the database: the failure itself says what the right answer was not.
    Grouped by `error_type`, because 12 turns over four causes is four small problems and 12
    of one cause is one worth fixing properly.
    """
    rows = []
    for record in connection.execute(
        """SELECT t.id, t.text, t.normalized, t.intent, t.action, t.confidence, t.outcome,
                  t.location, t.time_raw, t.unresolved, t.detail, f.kind, f.error_type,
                  f.intent AS corrected_intent, f.location AS corrected_location
           FROM turns t
           LEFT JOIN feedback f ON f.turn_id = t.id
           WHERE t.outcome IN ('error', 'need_location', 'clarified')
              OR f.kind IN ('down', 'correction')
           ORDER BY t.id DESC LIMIT ?""", (limit,)):
        text = (record["text"] or "").strip()
        if text.startswith("[") and "]" in text:          # strip the "[v4] " model tag
            text = text[text.index("]") + 1:].strip()
        detail = record["detail"] or ""
        rows.append({
            "turn_id": record["id"], "text": text,
            "outcome": record["outcome"],
            # the failure's own words are the label when nobody left one
            "cause": record["error_type"] or (
                "unresolved_location" if detail.startswith("unresolved:")
                else "no_place_named" if record["outcome"] == "need_location"
                else record["outcome"]),
            "detail": detail,
            "read_as": record["location"], "read_time": record["time_raw"],
            "corrected_intent": record["corrected_intent"],
            "corrected_location": record["corrected_location"],
        })
    return rows


def export(connection, path: Path, include_approved=False) -> int:
    rows = training_rows(connection, include_approved)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["text", "weather_intent", "action", "variables", "location",
                                "time", "source_kind", "model", "error_type"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--export", metavar="CSV")
    parser.add_argument("--include-approved", action="store_true",
                        help="also export thumbs-up turns the model was already confident about")
    parser.add_argument("--recent", type=int, metavar="N", help="print the last N turns")
    parser.add_argument("--confusion", action="store_true", help="predicted vs actual, from labels")
    parser.add_argument("--competing", action="store_true", help="turns where intents were close")
    parser.add_argument("--review", action="store_true", help="turns waiting to be labelled")
    args = parser.parse_args()

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
    if args.review:
        queue = review_queue(connection)
        print(f"\n{len(queue)} turns waiting for a label:")
        for row in queue[:20]:
            mark = "flagged" if row["flagged"] else row["outcome"]
            print(f"  [{row['id']:4d}] {mark:9s} {row['intent'] or '-':18s} "
                  f"conf {row['confidence'] or 0:.2f}  {row['text'][:46]!r}")
    if args.export:
        print(f"\nwrote {export(connection, args.export, args.include_approved)} labelled rows "
              f"-> {args.export}")


if __name__ == "__main__":
    main()
