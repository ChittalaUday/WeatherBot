"""
v2 dataset - CSV, and it contains conversations, not just isolated sentences.

    python -m src.v2.dataset --build      # -> data/v2_dataset.csv
    python -m src.v2.dataset --stats
    python -m src.v2.dataset --chats 3    # print a few conversations end to end

Every row is one *turn*, and turns are grouped by `chat_id`:

    chat_id  turn  text                                    operation  variables
    c0041    0     rain and temperature in Guntur tomorrow  SET        RAIN|TEMPERATURE
    c0041    1     what about Vizag?                        REPLACE    RAIN|TEMPERATURE
    c0041    2     and there next week?                     MODIFY     RAIN|TEMPERATURE

The follow-up rows carry the slots the user *means*, which is not what their words say -
"what about Vizag?" names no measurement at all. That makes the file two things at once:
per-turn training data for the heads, and a replay script for the context engine, which is
the only part that can turn turn 1 into the right question.

Sources, recorded per row so "what did the new data buy us?" stays answerable:
  v1        the existing generated splits, mapped onto the v2 shape
  multivar  prompts naming several variables, places or times at once
  chats     generated multi-turn conversations
  users     labelled turns exported from data/conversations.db (the runtime store)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data_loader import load_intents_csv
from src.tagger import normalize_time
from src.v2.schema import PAST_TIMES, PRESENT_TIMES, V1_TO_VARIABLE, Intent, Variable

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "v2_dataset.csv"

FIELDS = ["chat_id", "turn", "text", "intent", "variables", "aggregation", "locations",
          "times", "operation", "ctx_locations", "ctx_times", "split", "source", "lang"]

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
    "{a} and {b} in {loc} {t}", "what is the {a} and {b} in {loc} {t}?",
    "give me {a} and {b} for {loc} {t}", "{loc} {a} and {b} {t}",
    "check {a} and {b} in {loc} {t}", "i need {a} and {b} for {loc} {t}",
    "{t} {a} and {b} in {loc}", "show {a}, {b} for {loc} {t}",
]
TRIPLE_FRAMES = [
    "{a}, {b} and {c} in {loc} {t}", "{a}, {b}, {c} for {loc} {t}",
    "i want {a}, {b} and {c} in {loc} {t}", "send {a}, {b} and {c} for {loc} {t}",
    "what is the {a}, {b} and {c} in {loc} {t}?", "{loc} {t}: {a}, {b}, {c}",
]
MULTI_LOCATION_FRAMES = [
    "{a} in {loc} and {loc2} {t}", "compare {a} in {loc} and {loc2} {t}",
    "{a} for {loc}, {loc2} and {loc3} {t}", "{a} in {loc} vs {loc2} {t}",
]
MULTI_TIME_FRAMES = [
    "{a} in {loc} {t} and {t2}", "{a} in {loc} {t}, {t2}",
    "compare {a} in {loc} between {t} and {t2}",
]

# --- conversation shapes ----------------------------------------------------
# Each follow-up says what it changes and nothing else, which is the whole point.
OPENERS = [
    "{a} in {loc} {t}", "what is the {a} in {loc} {t}?", "{a} and {b} in {loc} {t}",
    "how is the {a} in {loc} {t}?", "check {a} for {loc} {t}",
]
FOLLOW_TIME = ["what about {t}?", "and {t}?", "what about {t} then?", "{t}?",
               "and how about {t}?", "same place {t}?"]
FOLLOW_PLACE = ["what about {loc}?", "and {loc}?", "how about {loc}?", "and in {loc}?",
                "same for {loc}?", "what about {loc} then?"]
FOLLOW_REFERENCE = ["and there?", "what about there?", "same place?", "and that place?",
                    "there?", "how about there?"]
FOLLOW_VARIABLE = ["what about the {a}?", "and the {a}?", "{a} too?", "also the {a}?"]

TIMES = ["today", "tomorrow", "tonight", "this evening", "next week", "this weekend",
         "day after tomorrow", "on Friday", "next 3 days", "at 6 PM", "tomorrow morning", ""]
LOCATIONS = ["Guntur", "Kakinada", "Hyderabad", "Vizag", "Nokha", "Rajahmundry", "Warangal",
             "Tirupati", "Kochi", "Bhopal", "Angara", "Kalimpong", "Barpeta", "Kasaragod",
             "Chikmagalur", "Kollam", "Pithoragarh", "Mandla", "Ziro", "Tawang"]


def _intent_for(time_span: str, action: str) -> Intent:
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


def _row(chat_id, turn, text, intent, variables, locations, times, operation,
         ctx_locations, ctx_times, split, source, aggregation="RAW", lang="en") -> dict:
    return {
        "chat_id": chat_id, "turn": turn, "text": text,
        "intent": intent.value if hasattr(intent, "value") else intent,
        "variables": "|".join(v.value if hasattr(v, "value") else v for v in variables),
        "aggregation": aggregation,
        "locations": json.dumps(locations), "times": json.dumps(times),
        "operation": operation,
        "ctx_locations": json.dumps(ctx_locations), "ctx_times": json.dumps(ctx_times),
        "split": split, "source": source, "lang": lang,
    }


def generate_conversations(rng: random.Random, count: int, split: str) -> list[dict]:
    """Multi-turn chats: an opener, then follow-ups that each change one slot."""
    rows = []
    for index in range(count):
        chat_id = f"gen-{split[:2]}-{index:05d}"
        variables = rng.sample([v for v in Variable if v != Variable.GENERAL],
                               rng.choice([1, 1, 2]))
        place, time_span = rng.choice(LOCATIONS), rng.choice(TIMES)
        words = [rng.choice(VARIABLE_WORDS[v]) for v in variables]

        frame = rng.choice([f for f in OPENERS if ("{b}" in f) == (len(variables) > 1)])
        text = _clean(frame.format(a=words[0], b=words[-1], loc=place, t=time_span))
        times = [time_span] if time_span else []
        if not all(span in text for span in [place] + times):
            continue

        state_places, state_times = [place], times
        rows.append(_row(chat_id, 0, text, _intent_for(time_span, "GET"), variables,
                         [place], times, "SET", state_places, state_times, split, "chats"))

        for turn in range(1, rng.choice([2, 3, 3, 4])):
            shape = rng.choices(["time", "place", "reference", "variable"],
                                weights=[35, 30, 20, 15])[0]
            if shape == "time":
                new_time = rng.choice([t for t in TIMES if t and t != state_times[:1]])
                text = _clean(rng.choice(FOLLOW_TIME).format(t=new_time))
                turn_row = _row(chat_id, turn, text, _intent_for(new_time, "GET"), variables,
                                [], [new_time], "MODIFY", state_places, [new_time], split, "chats")
                state_times = [new_time]
            elif shape == "place":
                new_place = rng.choice([p for p in LOCATIONS if p not in state_places])
                text = _clean(rng.choice(FOLLOW_PLACE).format(loc=new_place))
                turn_row = _row(chat_id, turn, text,
                                _intent_for(state_times[0] if state_times else "", "GET"),
                                variables, [new_place], [], "REPLACE", [new_place],
                                state_times, split, "chats")
                state_places = [new_place]
            elif shape == "reference":
                text = rng.choice(FOLLOW_REFERENCE)
                turn_row = _row(chat_id, turn, text,
                                _intent_for(state_times[0] if state_times else "", "GET"),
                                variables, [], [], "INHERIT", state_places, state_times,
                                split, "chats")
            else:
                new_variable = rng.choice([v for v in Variable
                                           if v not in variables and v != Variable.GENERAL])
                text = _clean(rng.choice(FOLLOW_VARIABLE).format(
                    a=rng.choice(VARIABLE_WORDS[new_variable])))
                variables = [new_variable]
                turn_row = _row(chat_id, turn, text,
                                _intent_for(state_times[0] if state_times else "", "GET"),
                                variables, [], [], "INHERIT", state_places, state_times,
                                split, "chats")
            rows.append(turn_row)
    return rows


def generate_multivariable(rng: random.Random, count: int, split: str) -> list[dict]:
    """Single-turn prompts naming several variables, places or times at once."""
    rows, seen, guard = [], set(), 0
    variables = [v for v in Variable if v != Variable.GENERAL]
    while len(rows) < count and guard < count * 40:
        guard += 1
        shape = rng.choices(["pair", "triple", "places", "times"], weights=[38, 25, 22, 15])[0]
        time_span, place = rng.choice(TIMES), rng.choice(LOCATIONS)

        if shape in {"pair", "triple"}:
            picked = rng.sample(variables, 3 if shape == "triple" else 2)
            words = [rng.choice(VARIABLE_WORDS[v]) for v in picked]
            frame = rng.choice(TRIPLE_FRAMES if shape == "triple" else PAIR_FRAMES)
            text = frame.format(loc=place, t=time_span, **dict(zip(("a", "b", "c"), words)))
            places, times = [place], [time_span] if time_span else []
            intent = _intent_for(time_span, "GET")
        elif shape == "places":
            picked = [rng.choice(variables)]
            others = rng.sample([p for p in LOCATIONS if p != place], 2)
            frame = rng.choice(MULTI_LOCATION_FRAMES)
            text = frame.format(a=rng.choice(VARIABLE_WORDS[picked[0]]), loc=place,
                                loc2=others[0], loc3=others[1], t=time_span)
            places = [place, others[0]] + ([others[1]] if "{loc3}" in frame else [])
            times, intent = [time_span] if time_span else [], Intent.COMPARE
        else:
            picked = [rng.choice(variables)]
            second = rng.choice([t for t in TIMES if t and t != time_span])
            time_span = time_span or "today"
            text = rng.choice(MULTI_TIME_FRAMES).format(
                a=rng.choice(VARIABLE_WORDS[picked[0]]), loc=place, t=time_span, t2=second)
            places, times, intent = [place], [time_span, second], Intent.COMPARE

        text = _clean(text)
        if text.lower() in seen or not all(span in text for span in places + times):
            continue
        seen.add(text.lower())
        rows.append(_row(f"mv-{split[:2]}-{len(rows):05d}", 0, text, intent, picked,
                         places, times, "SET", places, times, split, "multivar"))
    return rows


def from_v1(split: str) -> list[dict]:
    """Carry a v1 CSV split over to the v2 shape - one chat per row."""
    from src.build_dataset import EVAL_MANUAL, SPLITS

    path = ROOT / SPLITS[split][2] if split in SPLITS else EVAL_MANUAL
    rows = []
    for index, record in enumerate(load_intents_csv(path).to_dict("records")):
        times = record["time"]
        rows.append(_row(f"v1-{split[:2]}-{index:05d}", 0, record["text"],
                         _intent_for(times[0] if times else "", record["action"]),
                         [V1_TO_VARIABLE[record["weather_intent"]]],
                         record["location"], times, "SET", record["location"], times,
                         split, "v1", record.get("aggregation", "RAW"),
                         record.get("lang", "en")))
    return rows


def from_users() -> list[dict]:
    """Labelled turns from the runtime store - the retraining loop's input."""
    try:
        from backend import store
    except ImportError:
        return []

    connection = store.connect()
    rows = []
    for index, record in enumerate(store.training_rows(connection)):
        variable = V1_TO_VARIABLE.get(record["weather_intent"])
        if not variable:
            continue
        times = json.loads(record["time"])
        rows.append(_row(f"user-{index:05d}", 0, record["text"],
                         _intent_for(times[0] if times else "", record["action"]), [variable],
                         json.loads(record["location"]), times, "SET",
                         json.loads(record["location"]), times, "train", "users"))
    return rows


def load(path: Path | str = CSV_PATH, split: str | None = None,
         lang: str | None = None, source: str | None = None) -> list[dict]:
    """Rows as dicts, with the JSON columns already parsed."""
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if split and record["split"] != split:
                continue
            if lang and record["lang"] != lang:
                continue
            if source and record["source"] != source:
                continue
            rows.append({
                **record,
                "turn": int(record["turn"]),
                "variables": [v for v in record["variables"].split("|") if v],
                "locations": json.loads(record["locations"]),
                "times": json.loads(record["times"]),
                "ctx_locations": json.loads(record["ctx_locations"]),
                "ctx_times": json.loads(record["ctx_times"]),
            })
    return rows


def chats(rows: list[dict]) -> dict[str, list[dict]]:
    """Group turns back into conversations, in order."""
    grouped: dict[str, list[dict]] = {}
    for row in sorted(rows, key=lambda r: (r["chat_id"], r["turn"])):
        grouped.setdefault(row["chat_id"], []).append(row)
    return grouped


def build(path: Path = CSV_PATH, multivar: int = 2500, conversations: int = 1200,
          seed: int = 11) -> dict:
    rng = random.Random(seed)
    rows: list[dict] = []
    counts = {}

    for split in ("train", "test", "eval"):
        carried = from_v1(split)
        rows += carried
        counts[f"{split}:v1"] = len(carried)

    generated = generate_multivariable(rng, multivar, "train")
    cut = int(len(generated) * 0.85)
    for row in generated[cut:]:
        row["split"], row["chat_id"] = "test", row["chat_id"].replace("-tr-", "-te-")
    rows += generated
    counts["train:multivar"], counts["test:multivar"] = cut, len(generated) - cut

    chat_rows = generate_conversations(rng, conversations, "train")
    eval_chats = generate_conversations(rng, max(conversations // 8, 40), "eval")
    rows += chat_rows + eval_chats
    counts["train:chats"], counts["eval:chats"] = len(chat_rows), len(eval_chats)

    user_rows = from_users()
    rows += user_rows
    counts["train:users"] = len(user_rows)

    seen, unique = set(), []
    for row in rows:
        key = (row["chat_id"], row["turn"])
        if key not in seen:
            seen.add(key)
            unique.append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(unique)
    counts["rows"] = len(unique)
    return counts


def stats(rows: list[dict]) -> dict:
    grouped = chats(rows)
    return {
        "rows": len(rows),
        "chats": len(grouped),
        "multi_turn_chats": sum(len(turns) > 1 for turns in grouped.values()),
        "splits": dict(Counter(r["split"] for r in rows)),
        "sources": dict(Counter(r["source"] for r in rows)),
        "operations": dict(Counter(r["operation"] for r in rows)),
        "intents": dict(Counter(r["intent"] for r in rows)),
        "multi_variable": sum(len(r["variables"]) > 1 for r in rows),
        "multi_location": sum(len(r["locations"]) > 1 for r in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(CSV_PATH))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--multivar", type=int, default=2500)
    parser.add_argument("--conversations", type=int, default=1200)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--chats", type=int, metavar="N", help="print N conversations")
    args = parser.parse_args()

    if args.build:
        for key, value in build(Path(args.path), args.multivar, args.conversations).items():
            print(f"  {key:18s} {value}")

    rows = load(args.path)
    if args.stats or args.build:
        for key, value in stats(rows).items():
            print(f"  {key:18s} {value}")
    if args.chats:
        multi = [turns for turns in chats(rows).values() if len(turns) > 1]
        for turns in random.Random(3).sample(multi, min(args.chats, len(multi))):
            print(f"\n  chat {turns[0]['chat_id']}")
            for turn in turns:
                print(f"    [{turn['operation']:8s}] {turn['text']}")
                print(f"       means: {turn['variables']} @ {turn['ctx_locations']} "
                      f"{turn['ctx_times']}")


if __name__ == "__main__":
    main()
