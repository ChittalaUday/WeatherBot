"""Replays generated conversations through the real pipeline. Run: python test_conversations.py

The per-turn model scores in models/metrics_v2.json cannot judge a follow-up: "and there?"
contains no place, no time and no measurement, so any per-utterance metric over it is noise.
What matters is the state the pipeline holds *after* the turn, which is what this replays -
model, then normalizer, then the context engine, exactly as the socket does it.
"""

from collections import Counter

from backend import registry as models
from backend import state as context
from src.normalize import normalize
from src.schema import ConversationState
from src.v2 import dataset as v2_dataset

FLOORS = {"operation": 0.85, "locations": 0.90, "times": 0.85, "variables": 0.80}
MIN_CONFIDENCE = 0.45          # same gate the socket uses


def replay(registry, chat_turns, version="v2"):
    """Run one conversation and report what the pipeline believed after each turn."""
    state, results = ConversationState(), []
    for turn in chat_turns:
        cleaned = normalize(turn["text"])
        understanding = registry.understand(cleaned.normalized, version)
        reference = context.detect_reference(cleaned.normalized)
        follow_up = context.is_follow_up(cleaned.normalized)

        named = [name for name in understanding.locations
                 if not models.locations_module.is_relative(name)
                 and not models.locations_module.is_probably_not_a_place(name)]
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
    rows = [r for r in v2_dataset.load(split="eval") if r["source"] == "chats"]
    conversations = [turns for turns in v2_dataset.chats(rows).values() if len(turns) > 1]
    assert conversations, "no multi-turn conversations in the eval split"

    hits = Counter()
    total = 0
    confusion = Counter()
    for turns in conversations:
        predicted = replay(registry, turns)
        for turn, got in zip(turns, predicted):
            total += 1
            wanted_locations = sorted(name.lower() for name in turn["ctx_locations"])
            wanted_times = sorted(
                filter(None, (__import__("src.tagger", fromlist=["normalize_time"])
                              .normalize_time(span) for span in turn["ctx_times"])))
            hits["operation"] += got["operation"] == turn["operation"]
            hits["locations"] += sorted(got["locations"]) == wanted_locations
            hits["times"] += sorted(got["times"]) == wanted_times
            hits["variables"] += got["variables"] == sorted(turn["variables"])
            if got["operation"] != turn["operation"]:
                confusion[f"{turn['operation']} -> {got['operation']}"] += 1

    print(f"replayed {len(conversations)} conversations, {total} turns")
    for name, floor in FLOORS.items():
        score = hits[name] / total
        print(f"  {name:11s} {score:.1%}" + (f"   (floor {floor:.0%})" if score < floor else ""))
    if confusion:
        print("  operation misreads:", dict(confusion.most_common(4)))
    for name, floor in FLOORS.items():
        assert hits[name] / total >= floor, f"{name} {hits[name] / total:.1%} below floor {floor:.0%}"
    print("OK: context survives multi-turn conversations")


if __name__ == "__main__":
    main()
