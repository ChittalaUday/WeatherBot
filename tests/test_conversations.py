"""Replays generated conversations through the real pipeline. Run: python tests/test_conversations.py

A per-utterance model score cannot judge a follow-up: "and there?" contains no place, no time
and no measurement, so any per-utterance metric over it is noise. What matters is the state the
pipeline holds *after* the turn, which is what this replays - model, then normalizer, then the
context engine, exactly as the socket does it.

The fixture is data/v3_dataset.csv, the only shipped file with multi-turn conversations and
gold *context* slots in it. Read here rather than through a dataset module: it is a fixture for
this test now, not a training set for a model that still exists. Its `variables` column uses
the retired label set, so it is not scored - `operation`, `locations` and `times` are what this
file is actually measuring, and they are label-set independent.
"""

import csv
import json
from collections import Counter

from _root import ROOT

from backend import nlu as models
from backend.nlu import context
from backend.pipeline import places as place_index
from src.normalize import normalize
from src.schema import ConversationState
from src.v2 import dataset as v2_dataset  # chats() - grouping helper, version-neutral

FIXTURE = ROOT / "data" / "v3_dataset.csv"
# Measured, not aspirational: the trained classifier scores operation 99.8%, locations 89.5%,
# this fixture. Each floor sits a few points under that, so a regression trips it and normal
# drift does not.
FLOORS = {"operation": 0.95, "locations": 0.85, "times": 0.95}
VERSION = "v4"
MIN_CONFIDENCE = 0.45          # same gate the socket uses


def load_chat_rows() -> list[dict]:
    """The conversation turns out of the fixture, slots parsed. [] if it is not there."""
    if not FIXTURE.exists():
        return []
    with FIXTURE.open(newline="", encoding="utf-8") as handle:
        return [{**record, "turn": int(record["turn"]),
                 "variables": [v for v in record["variables"].split("|") if v],
                 "locations": json.loads(record["locations"]),
                 "times": json.loads(record["times"]),
                 "ctx_locations": json.loads(record["ctx_locations"]),
                 "ctx_times": json.loads(record["ctx_times"])}
                for record in csv.DictReader(handle)
                if record["split"] == "eval" and record["source"] == "chats"]


def replay(registry, chat_turns, version=VERSION):
    """Run one conversation and report what the pipeline believed after each turn."""
    state, results = ConversationState(), []
    for turn in chat_turns:
        cleaned = normalize(turn["text"])
        understanding = registry.understand(cleaned.normalized, version)
        reference = context.detect_reference(cleaned.normalized)
        follow_up = context.is_follow_up(cleaned.normalized)

        named = [name for name in understanding.locations
                 if not place_index.is_relative(name)]
        state, operation = context.apply(
            state,
            weather_intent=understanding.intent, action=understanding.action,
            aggregation=understanding.aggregation, location=named,
            time_raw=understanding.times[0] if understanding.times else None,
            time_normalized=(understanding.times_normalized[0]
                             if understanding.times_normalized else None),
            reference=reference, follow_up=follow_up,
            confident=understanding.confidence >= MIN_CONFIDENCE,
            text=cleaned.normalized, variables=understanding.variables,
        )
        results.append({
            "operation": operation.value,
            "locations": [name.lower() for name in state.location],
            "times": [state.time_normalized] if state.time_normalized else [],
            "variables": sorted(state.variables),
        })
    return results


def main():
    registry = models.Registry()
    rows = load_chat_rows()
    if not rows:
        print(f"SKIP: {FIXTURE.relative_to(ROOT)} absent - the multi-turn fixture is not "
              f"shipped. Rebuild it with: python -m src.v2.dataset --build")
        return
    conversations = [turns for turns in v2_dataset.chats(rows).values() if len(turns) > 1]
    assert conversations, "no multi-turn conversations in the eval split"

    hits = Counter()
    total = 0
    confusion = Counter()
    for turns in conversations:
        predicted = replay(registry, turns, VERSION)
        for turn, got in zip(turns, predicted):
            total += 1
            wanted_locations = sorted(name.lower() for name in turn["ctx_locations"])
            wanted_times = sorted(
                filter(None, (__import__("src.tagger", fromlist=["normalize_time"])
                              .normalize_time(span) for span in turn["ctx_times"])))
            hits["operation"] += got["operation"] == turn["operation"]
            hits["locations"] += sorted(got["locations"]) == wanted_locations
            hits["times"] += sorted(got["times"]) == wanted_times
            if got["operation"] != turn["operation"]:
                confusion[f"{turn['operation']} -> {got['operation']}"] += 1

    print(f"replayed {len(conversations)} conversations, {total} turns "
          f"through {models.MODELS[VERSION]['name']} ({VERSION})")
    for name, floor in FLOORS.items():
        score = hits[name] / total
        print(f"  {name:11s} {score:.1%}" + (f"   (floor {floor:.0%})" if score < floor else ""))
    if confusion:
        print("  operation misreads:", dict(confusion.most_common(4)))
    for name, floor in FLOORS.items():
        assert hits[name] / total >= floor, f"{name} {hits[name] / total:.1%} below floor {floor:.0%}"
    print(f"OK: context survives multi-turn conversations ({models.MODELS[VERSION]['name']})")


if __name__ == "__main__":
    main()
