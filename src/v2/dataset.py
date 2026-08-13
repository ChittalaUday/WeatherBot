"""
v2 dataset - one SQLite table instead of three CSVs.

    python -m src.v2.dataset --build          # rebuild data/v2_dataset.db from scratch
    python -m src.v2.dataset --stats
    python -m src.v2.dataset --sample 10

Why a database rather than more CSVs:

  - one row shape for train / test / eval, separated by a `split` column instead of by file,
    so a row can never drift between them or be counted twice
  - `source` records where a row came from (v1 carry-over, multi-variable generator, or a
    real user turn that a human labelled), which makes "what did the new data buy us?"
    a query rather than an archaeology exercise
  - labelled production turns land in the same table as everything else, so the retraining
    loop is an INSERT, not a file merge

Three sources feed it:
  v1        the existing generated splits, mapped onto the v2 shape (14 intents -> 1 slot)
  multivar  freshly generated prompts that name several variables, places or times at once
  users     rows exported from data/conversations.db (choice/correction feedback)
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_intents_csv
from src.tagger import normalize_time
from src.v2.schema import PAST_TIMES, PRESENT_TIMES, V1_TO_VARIABLE, Intent, Variable

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "v2_dataset.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    text        TEXT NOT NULL UNIQUE,
    intent      TEXT NOT NULL,
    variables   TEXT NOT NULL,     -- json list, multi-label
    aggregation TEXT NOT NULL DEFAULT 'RAW',
    locations   TEXT NOT NULL,     -- json list of raw spans
    times       TEXT NOT NULL,     -- json list of raw spans
    split       TEXT NOT NULL,     -- train | test | eval
    source      TEXT NOT NULL,     -- v1 | multivar | users
    lang        TEXT NOT NULL DEFAULT 'en'
);
CREATE INDEX IF NOT EXISTS examples_split ON examples(split);
CREATE INDEX IF NOT EXISTS examples_source ON examples(source);
"""

# --- multi-variable generation ----------------------------------------------
# v1 could not express "rain and temperature", so no template ever produced it. These do.

VARIABLE_WORDS = {
    Variable.TEMPERATURE: ["temperature", "temp", "how hot it is"],
    Variable.TEMPERATURE_MIN: ["minimum temperature", "min temp", "overnight low"],
    Variable.TEMPERATURE_MAX: ["maximum temperature", "max temp", "daytime high"],
    Variable.RAIN: ["rain", "rainfall", "precipitation", "showers"],
    Variable.HUMIDITY: ["humidity", "relative humidity"],
    Variable.DEW_POINT: ["dew point", "dewpoint"],
    Variable.WIND_SPEED: ["wind speed", "wind"],
    Variable.WIND_DIRECTION: ["wind direction", "wind heading"],
    Variable.SUNSHINE: ["sunshine", "sunlight", "sunshine hours"],
    Variable.CLOUD_COVER: ["cloud cover", "cloudiness"],
    Variable.SOIL_MOISTURE: ["soil moisture", "field moisture"],
    Variable.SOIL_TEMPERATURE: ["soil temperature", "soil temp"],
    Variable.GENERAL: ["weather", "conditions", "weather conditions"],
}
PAIR_FRAMES = [
    "{a} and {b} in {loc} {t}",
    "what is the {a} and {b} in {loc} {t}?",
    "give me {a} and {b} for {loc} {t}",
    "{loc} {a} and {b} {t}",
    "check {a} and {b} in {loc} {t}",
    "i need {a} and {b} for {loc} {t}",
    "{t} {a} and {b} in {loc}",
    "show {a}, {b} for {loc} {t}",
]
TRIPLE_FRAMES = [
    "{a}, {b} and {c} in {loc} {t}",
    "{a}, {b}, {c} for {loc} {t}",
    "i want {a}, {b} and {c} in {loc} {t}",
    "send {a}, {b} and {c} for {loc} {t}",
    "what is the {a}, {b} and {c} in {loc} {t}?",
    "give me {a}, {b} and {c} for {loc} {t}",
    "{loc} {t}: {a}, {b}, {c}",
]
MULTI_LOCATION_FRAMES = [
    "{a} in {loc} and {loc2} {t}",
    "compare {a} in {loc} and {loc2} {t}",
    "{a} for {loc}, {loc2} and {loc3} {t}",
    "{a} in {loc} vs {loc2} {t}",
]
MULTI_TIME_FRAMES = [
    "{a} in {loc} {t} and {t2}",
    "{a} in {loc} {t}, {t2}",
    "compare {a} in {loc} between {t} and {t2}",
]
TIMES = ["today", "tomorrow", "tonight", "this evening", "next week", "this weekend",
         "day after tomorrow", "on Friday", "next 3 days", "at 6 PM", "tomorrow morning", ""]
LOCATIONS = ["Guntur", "Kakinada", "Hyderabad", "Vizag", "Nokha", "Rajahmundry", "Warangal",
             "Tirupati", "Kochi", "Bhopal", "Angara", "Kalimpong", "Barpeta", "Kasaragod",
             "Chikmagalur", "Kollam", "Pithoragarh", "Mandla", "Ziro", "Tawang"]


def _intent_for(time_span: str, action: str) -> Intent:
    """v1's action plus the time decides v2's coarse intent."""
    if action == "COMPARE":
        return Intent.COMPARE
    if action == "ALERT":
        return Intent.ALERT
    canonical = normalize_time(time_span) if time_span else ""
    if canonical in PAST_TIMES:
        return Intent.HISTORICAL
    if not canonical or canonical in PRESENT_TIMES:
        return Intent.CURRENT
    return Intent.FORECAST


def _clean(text: str) -> str:
    return " ".join(text.split()).replace(" ?", "?").replace(" ,", ",").strip()


def generate_multivariable(rng: random.Random, count: int) -> list[dict]:
    """Prompts naming several variables, places or times at once."""
    rows, seen = [], set()
    variables = [v for v in Variable if v != Variable.GENERAL]
    guard = 0
    while len(rows) < count and guard < count * 40:
        guard += 1
        shape = rng.choices(["pair", "triple", "places", "times"], weights=[38, 25, 22, 15])[0]
        time_span = rng.choice(TIMES)
        place = rng.choice(LOCATIONS)

        if shape in {"pair", "triple"}:
            picked = rng.sample(variables, 3 if shape == "triple" else 2)
            words = [rng.choice(VARIABLE_WORDS[v]) for v in picked]
            frame = rng.choice(TRIPLE_FRAMES if shape == "triple" else PAIR_FRAMES)
            fields = dict(zip(("a", "b", "c"), words))
            text = frame.format(loc=place, t=time_span, **fields)
            row = {"variables": picked, "locations": [place],
                   "times": [time_span] if time_span else [],
                   "intent": _intent_for(time_span, "GET")}
        elif shape == "places":
            picked = [rng.choice(variables)]
            others = rng.sample([p for p in LOCATIONS if p != place], 2)
            frame = rng.choice(MULTI_LOCATION_FRAMES)
            text = frame.format(a=rng.choice(VARIABLE_WORDS[picked[0]]), loc=place,
                                loc2=others[0], loc3=others[1], t=time_span)
            places = [place, others[0]] + ([others[1]] if "{loc3}" in frame else [])
            row = {"variables": picked, "locations": places,
                   "times": [time_span] if time_span else [], "intent": Intent.COMPARE}
        else:
            picked = [rng.choice(variables)]
            second = rng.choice([t for t in TIMES if t and t != time_span])
            time_span = time_span or "today"
            text = rng.choice(MULTI_TIME_FRAMES).format(
                a=rng.choice(VARIABLE_WORDS[picked[0]]), loc=place, t=time_span, t2=second)
            row = {"variables": picked, "locations": [place], "times": [time_span, second],
                   "intent": Intent.COMPARE}

        text = _clean(text)
        if text.lower() in seen:
            continue
        seen.add(text.lower())
        # spans must stay verbatim, exactly as in v1 (Rules 4.1 / 4.2)
        if not all(span in text for span in row["locations"] + row["times"]):
            continue
        rows.append({**row, "text": text, "aggregation": "RAW", "source": "multivar"})
    return rows


def from_v1(split: str) -> list[dict]:
    """Carry a v1 CSV split over to the v2 shape."""
    from src.build_dataset import EVAL_MANUAL, SPLITS

    path = ROOT / SPLITS[split][2] if split in SPLITS else EVAL_MANUAL
    frame = load_intents_csv(path)
    rows = []
    for record in frame.to_dict("records"):
        variable = V1_TO_VARIABLE[record["weather_intent"]]
        time_spans = record["time"]
        rows.append({
            "text": record["text"],
            "intent": _intent_for(time_spans[0] if time_spans else "", record["action"]),
            "variables": [variable],
            "aggregation": record.get("aggregation", "RAW"),
            "locations": record["location"],
            "times": time_spans,
            "source": "v1",
            "lang": record.get("lang", "en"),
        })
    return rows


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA)
    return connection


def insert(connection, rows: list[dict], split: str) -> int:
    written = 0
    for row in rows:
        try:
            connection.execute(
                """INSERT INTO examples (text, intent, variables, aggregation, locations,
                                         times, split, source, lang)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (row["text"],
                 row["intent"].value if hasattr(row["intent"], "value") else row["intent"],
                 json.dumps([v.value if hasattr(v, "value") else v for v in row["variables"]]),
                 row.get("aggregation", "RAW"),
                 json.dumps(row["locations"]), json.dumps(row["times"]),
                 split, row.get("source", "v1"), row.get("lang", "en")),
            )
            written += 1
        except sqlite3.IntegrityError:
            continue                    # UNIQUE(text): a prompt belongs to exactly one split
    connection.commit()
    return written


def load(connection, split: str | None = None, lang: str | None = None) -> list[dict]:
    query = "SELECT * FROM examples"
    clauses, params = [], []
    if split:
        clauses.append("split = ?")
        params.append(split)
    if lang:
        clauses.append("lang = ?")
        params.append(lang)
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    return [{**dict(row),
             "variables": json.loads(row["variables"]),
             "locations": json.loads(row["locations"]),
             "times": json.loads(row["times"])}
            for row in connection.execute(query, params)]


def build(db_path: Path = DB_PATH, multivar: int = 2500, seed: int = 11) -> dict:
    """Rebuild the whole dataset. Idempotent: the table is dropped first."""
    connection = connect(db_path)
    connection.execute("DROP TABLE IF EXISTS examples")
    connection.executescript(SCHEMA)
    rng = random.Random(seed)

    written = {}
    written["train:v1"] = insert(connection, from_v1("train"), "train")
    written["test:v1"] = insert(connection, from_v1("test"), "test")
    written["eval:v1"] = insert(connection, from_v1("eval"), "eval")

    generated = generate_multivariable(rng, multivar)
    cut = int(len(generated) * 0.85)
    written["train:multivar"] = insert(connection, generated[:cut], "train")
    written["test:multivar"] = insert(connection, generated[cut:], "test")

    # labelled production turns, if any exist yet
    try:
        from backend import store

        conversations = store.connect()
        user_rows = []
        for record in store.training_rows(conversations):
            variable = V1_TO_VARIABLE.get(record["weather_intent"])
            if not variable:
                continue
            times = json.loads(record["time"])
            user_rows.append({
                "text": record["text"],
                "intent": _intent_for(times[0] if times else "", record["action"]),
                "variables": [variable], "aggregation": "RAW",
                "locations": json.loads(record["location"]), "times": times, "source": "users",
            })
        written["train:users"] = insert(connection, user_rows, "train")
    except Exception:                    # noqa: BLE001 - no conversations yet is normal
        written["train:users"] = 0

    return written


def stats(connection) -> dict:
    rows = load(connection)
    return {
        "total": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "intents": dict(Counter(r["intent"] for r in rows)),
        "multi_variable": sum(len(r["variables"]) > 1 for r in rows),
        "multi_location": sum(len(r["locations"]) > 1 for r in rows),
        "multi_time": sum(len(r["times"]) > 1 for r in rows),
        "variables": dict(Counter(v for r in rows for v in r["variables"]).most_common()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DB_PATH))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--multivar", type=int, default=2500)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--sample", type=int, metavar="N")
    args = parser.parse_args()

    if args.build:
        for key, count in build(Path(args.db), args.multivar).items():
            print(f"  {key:18s} {count}")

    connection = connect(args.db)
    if args.stats or args.build:
        for key, value in stats(connection).items():
            print(f"  {key:16s} {value}")
    if args.sample:
        for row in load(connection)[:0] or []:
            pass
        rows = connection.execute(
            "SELECT * FROM examples WHERE source='multivar' ORDER BY RANDOM() LIMIT ?",
            (args.sample,)).fetchall()
        print()
        for row in rows:
            print(f"  {row['intent']:11s} {json.loads(row['variables'])} "
                  f"loc={json.loads(row['locations'])} time={json.loads(row['times'])}")
            print(f"     {row['text']}")


if __name__ == "__main__":
    main()
