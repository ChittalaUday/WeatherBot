"""Every logged turn of the last N days that had any, as one CSV, exactly as SQLite holds it.

    python scripts/export_turns.py                     # last 10 active days -> data/turns_export.csv
    python scripts/export_turns.py --days 3 --out /tmp/x.csv
    python scripts/export_turns.py --unique            # one row per distinct question instead
    python scripts/export_turns.py --keep-retries      # including the need_location stubs

`--days N` counts days that *have* turns, not calendar days: nobody chats every day.

A question asked without a place is logged twice: once as `need_location`, then again a few
seconds later under the same chat and turn index once the browser has sent coordinates. Both
rows are the same question, and any feedback is on the second one, so the stub is dropped and
its id kept in `retry_of`. `--keep-retries` leaves both, which is what you want if you are
measuring how often the location prompt fires.

One row per turn, every `turns` column plus the feedback row joined onto it as `fb_*`. Nothing
is re-predicted and nothing is judged - `intent`/`action`/`confidence` are what the model said
at the time, `fb_*` is what a human said afterwards, and the labelling happens elsewhere.
`text` keeps the "[v4] " tag the store writes; `clean_text` is the same string without it.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv" / "bin" / "python"
if VENV.exists() and Path(sys.executable).resolve() != VENV.resolve():
    os.execv(str(VENV), [str(VENV)] + sys.argv)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.config import DB_PATH  # noqa: E402

TAG = re.compile(r"^\[[^\]]*\]\s*")          # the "[v4] " the store prefixes onto logged text

QUERY = """
SELECT t.*,
       f.kind        AS fb_kind,
       f.ts          AS fb_ts,
       f.revisions   AS fb_revisions,
       f.intent      AS fb_intent,
       f.action      AS fb_action,
       f.variables   AS fb_variables,
       f.location    AS fb_location,
       f.time_raw    AS fb_time_raw,
       f.error_type  AS fb_error_type,
       f.note        AS fb_note
FROM turns t LEFT JOIN feedback f ON f.turn_id = t.id
WHERE date(t.ts) IN ({marks})
ORDER BY t.id
"""


def collapse_retries(rows: list[dict]) -> list[dict]:
    """Drop the `need_location` stub of a question that was re-asked and answered."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["chat_id"], row["turn"], row["clean_text"]), []).append(row)
    kept = []
    for group in groups.values():
        final = max(group, key=lambda r: r["id"])
        final["retry_of"] = " ".join(str(r["id"]) for r in group if r is not final)
        kept.append(final)
    return sorted(kept, key=lambda r: r["id"])


def export(db: str, days: int, unique: bool, keep_retries: bool = False) -> tuple[list[dict], list[str]]:
    # read-only: the app may well be running against this file right now
    connection = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    dates = [row[0] for row in connection.execute(
        "SELECT DISTINCT date(ts) d FROM turns ORDER BY d DESC LIMIT ?", (days,))]
    if not dates:
        return [], dates
    marks = ",".join("?" * len(dates))
    rows = [dict(row) for row in connection.execute(QUERY.format(marks=marks), dates)]
    for row in rows:
        row["clean_text"] = TAG.sub("", (row["text"] or "").strip())
        row["model"] = (match.group(0).strip("[] ") if (match := TAG.match(row["text"] or ""))
                        else "legacy")
        row["retry_of"] = ""

    if not keep_retries:
        rows = collapse_retries(rows)

    if unique:
        # keep the newest turn of each distinct question, and say how often it was asked
        seen: dict[str, dict] = {}
        for row in rows:
            key = row["clean_text"].lower()
            if not key:
                continue
            row["asked"] = seen.get(key, {}).get("asked", 0) + 1
            row["turn_ids"] = f"{seen.get(key, {}).get('turn_ids', '')} {row['id']}".strip()
            seen[key] = row
        rows = sorted(seen.values(), key=lambda r: r["id"])
    return rows, dates


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=10, help="days that have turns (default 10)")
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--out", default=str(ROOT / "data" / "turns_export.csv"))
    parser.add_argument("--unique", action="store_true", help="one row per distinct question")
    parser.add_argument("--keep-retries", action="store_true",
                        help="keep the need_location stub of a re-asked question")
    args = parser.parse_args()

    rows, dates = export(args.db, args.days, args.unique, args.keep_retries)
    if not rows:
        print("export: no turns in that window")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    rated = sum(1 for row in rows if row["fb_kind"])
    dropped = sum(len(row["retry_of"].split()) for row in rows if row["retry_of"])
    print(f"export: {len(rows)} rows over {len(dates)} days ({dates[-1]} .. {dates[0]}), "
          f"{rated} with feedback"
          + (f", {dropped} need_location retries collapsed" if dropped else "")
          + f" -> {out}")


if __name__ == "__main__":
    main()
