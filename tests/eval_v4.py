"""Score the v4 model against the hand-written evaluation set.

    python tests/eval_v4.py

data/eval_v4.csv is authored by hand, one row at a time, and no template in
src/v4/dataset.py produced any of it. That is the entire point: train, test and eval are all
generated from the same frames, so the numbers they give measure template memorisation. This
file is the only measurement in the project that a real user could have written.
"""
import csv
import json
from collections import Counter, defaultdict

from _root import ROOT

from src.v4.model import V4Model
from src.v4.schema import NO_DATA_NEEDED, Intent

EVAL = ROOT / "data" / "eval_v4.csv"


def main():
    rows = list(csv.DictReader(EVAL.open()))
    model = V4Model.load()
    norm = lambda spans: sorted(s.lower() for s in spans)
    hits, seen = Counter(), Counter()
    misses = defaultdict(list)

    for r in rows:
        p = model.predict(r["text"])
        gold_loc, gold_time = json.loads(r["locations"]), json.loads(r["times"])
        checks = {
            "intent": p.intent.value == r["intent"],
            "weather_intent": p.weather_intent.value == r["weather_intent"],
            "variables": {v.value for v in p.slots.variables} == {v for v in r["variables"].split("|") if v},
            "locations": norm(p.slots.locations) == norm(gold_loc),
            "times": norm(p.slots.times) == norm(gold_time),
        }
        if r["intent"] == Intent.ADVICE.value:
            checks["activity"] = p.activity.value == r["activity"]
        # the predicted value lives on p.slots for three of these, not on p
        got_of = {
            "intent": p.intent.value,
            "weather_intent": p.weather_intent.value,
            "activity": p.activity.value,
            "variables": "|".join(sorted(v.value for v in p.slots.variables)) or "-",
            "locations": p.slots.locations or "-",
            "times": p.slots.times or "-",
        }
        gold_of = {**{k: r.get(k) for k in ("intent", "weather_intent", "activity")},
                   "variables": "|".join(sorted(v for v in r["variables"].split("|") if v)) or "-",
                   "locations": gold_loc or "-", "times": gold_time or "-"}
        for k, ok in checks.items():
            hits[k] += ok; seen[k] += 1
            if not ok:
                misses[k].append((r["text"], gold_of[k], got_of[k]))
        every = all(checks.values())
        hits["everything"] += every; seen["everything"] += 1
        if not every:
            misses["everything"].append((r["text"], "", ""))

    print(f"hand-written evaluation set: {len(rows)} rows\n")
    for k in ("intent", "weather_intent", "variables", "activity", "locations", "times", "everything"):
        if seen[k]:
            print(f"  {k:16s} {hits[k]/seen[k]:6.1%}   ({hits[k]}/{seen[k]})")

    print("\nworst component failures")
    for k in ("intent", "variables", "locations", "times", "activity", "weather_intent"):
        if misses[k]:
            print(f"\n  {k} — {len(misses[k])} wrong")
            for text, gold, got in misses[k][:7]:
                print(f"     {text[:52]:54s} gold={gold}  got={got}")

    # the split that matters most: does a non-weather turn ever reach the weather path?
    leak = sum(1 for r in rows if r["intent"] in {i.value for i in NO_DATA_NEEDED}
               and model.predict(r["text"]).intent not in NO_DATA_NEEDED)
    print(f"\nnon-weather turns misrouted into the weather path: {leak}")


if __name__ == "__main__":
    main()
