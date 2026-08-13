"""Self-check for Model 1. Run: python test_model.py

Fails if the bundle is missing, if a smoke query breaks, if a predicted span is not verbatim
in its prompt (Rules 4.1 / 4.2), or if accuracy on the hand-written English evaluation set
drops below the floors below. Retrain with `python -m src.v3.model --export` first.

Model 1 predicts eight things per turn: intent, aggregation, a set of variables, location and
time spans, and the three presentation decisions (detail, chart, insights). This checks all
of them - a model that reads the question right but picks the wrong chart is still wrong.
"""

from pathlib import Path

from src.v3.dataset import CSV_PATH
from src.v3.model import BUNDLE_PATH, V3Model, evaluate
from src.v3.schema import Detail

ROOT = Path(__file__).parent

# Floors, not targets: set below the measured numbers so this catches regressions without
# going red on run-to-run noise. Raise them when the model genuinely improves.
# Measured on the English eval utterances at export time (models/metrics_v3.json).
FLOORS = {
    "intent": 0.92,          # measured 0.959
    "aggregation": 0.92,     # measured 0.954
    "detail": 0.97,          # measured 1.000
    "chart": 0.80,           # measured 0.840
    "variables": 0.88,       # measured 0.922
    "locations": 0.90,       # measured 0.945
    "times": 0.92,           # measured 0.959
    "everything": 0.62,      # measured 0.680 - all eight targets right on the same turn
    "insights_f1": 0.86,     # measured 0.907
}

# (query, intent, variables, locations, times) - unambiguous English.
SMOKE = [
    ("Will it rain in Rajahmundry tomorrow morning?",
     "FORECAST", ["RAIN"], ["Rajahmundry"], ["tomorrow morning"]),
    ("Compare the maximum temperature in Hyderabad and Vizag next week",
     "COMPARE", ["TEMPERATURE_MAX"], ["Hyderabad", "Vizag"], ["next week"]),
    ("Alert me if the wind speed crosses 40 kmph in Kakinada tonight",
     "ALERT", ["WIND_SPEED"], ["Kakinada"], ["tonight"]),
    ("What is the soil moisture in my field right now?",
     "CURRENT", ["SOIL_MOISTURE"], ["my field"], ["right now"]),
    ("rain and temperature in Guntur tomorrow",
     "FORECAST", ["RAIN", "TEMPERATURE"], ["Guntur"], ["tomorrow"]),
]

# (query, aggregation) - the model, not a keyword rule, decides the reduction (Rule 2.3)
AGGREGATION_SMOKE = [
    ("total rainfall in Nokha next 7 days", "SUM"),
    ("average humidity in Tarora next 3 days", "AVG"),
    ("peak wind speed in Kakinada tomorrow", "MAX"),
    ("lowest soil temperature in Vaghan this week", "MIN"),
    ("when will the rain stop in Remta today?", "TREND"),
    ("will it rain in Guntur tomorrow?", "RAW"),
]

# (query, detail, chart) - the presentation decisions Python used to make (Rules 7.1 / 7.2).
# "just tell me" asks for one number, so it gets no chart; a range gets a line; two places
# side by side get grouped bars.
PRESENTATION_SMOKE = [
    ("just tell me the temperature in Guntur tomorrow", "MINIMAL", "NONE"),
    ("full details of the weather in Guntur next 5 days", "FULL", "LINE"),
    ("compare rainfall in Guntur and Vizag next 3 days", "NORMAL", "GROUPED_BAR"),
    ("rainfall in Guntur next 7 days", "NORMAL", "LINE"),
]


def check_freshness():
    """A bundle older than the training data is a stale export - warn, do not fail."""
    if CSV_PATH.exists() and CSV_PATH.stat().st_mtime > BUNDLE_PATH.stat().st_mtime:
        print(f"WARNING: {CSV_PATH.name} is newer than the bundle - "
              "rerun python -m src.v3.model --export")


def check_smoke(model):
    for text, intent, variables, locations, times in SMOKE:
        out = model.predict(text)
        got_variables = sorted(v.value for v in out.slots.variables)
        assert out.intent.value == intent, f"{text!r} -> {out.intent.value}, want {intent}"
        assert got_variables == sorted(variables), \
            f"{text!r} -> variables {got_variables}, want {sorted(variables)}"
        assert out.slots.locations == locations, \
            f"{text!r} -> location {out.slots.locations}, want {locations}"
        assert sorted(out.slots.times) == sorted(times), \
            f"{text!r} -> time {out.slots.times}, want {times}"
    print(f"OK smoke   : {len(SMOKE)} queries predicted exactly")


def check_presentation(model):
    """Model 1's job is to decide, not to ask: every turn gets a detail level and a chart."""
    for text, detail, chart in PRESENTATION_SMOKE:
        presentation = model.predict(text).presentation
        if detail:
            assert presentation.detail.value == detail, \
                f"{text!r} -> detail {presentation.detail.value}, want {detail}"
        if chart:
            assert presentation.chart.value == chart, \
                f"{text!r} -> chart {presentation.chart.value}, want {chart}"
    print(f"OK present : {len(PRESENTATION_SMOKE)} detail/chart decisions correct")


def check_always_decides(model, rows):
    """No turn may come back undecided - that is the whole point of Model 1."""
    for row in rows:
        presentation = model.predict(row["text"]).presentation
        assert isinstance(presentation.detail, Detail), row["text"]
        assert presentation.chart is not None, row["text"]
    print(f"OK decides : a detail level and a chart on all {len(rows)} prompts, none deferred")


def check_time_normalized(model, rows):
    """Every raw time span gets exactly one canonical twin, in the same order (Rule 4.3)."""
    for row in rows:
        slots = model.predict(row["text"]).slots
        assert len(slots.times_normalized) == len(slots.times), \
            f"{row['text']!r} -> {len(slots.times)} spans but {len(slots.times_normalized)} normalized"
        for value in slots.times_normalized:
            assert value == value.strip().lower(), f"{value!r} is not canonical-cased"

    canonical = {
        "will it rain tommorrow": ["tomorrow"],
        "temp in Nokha at 6:45 pm": ["18:45"],
        "humidity from 7 AM to 11 AM": ["07:00-11:00"],
        "rain on Friday": ["friday"],
        "wind rn": ["now"],
        "forecast for the next 3 days": ["next 3 days"],
    }
    for query, expected in canonical.items():
        actual = model.predict(query).slots.times_normalized
        assert actual == expected, f"{query!r} -> {actual}, want {expected}"
    print(f"OK time    : {len(canonical)} spellings folded to canonical form, "
          f"aligned 1:1 across {len(rows)} prompts")


def check_spans_verbatim(model, rows):
    """Rules 4.1 / 4.2: every returned span must be raw text lifted from the prompt."""
    for row in rows:
        slots = model.predict(row["text"]).slots
        for span in list(slots.locations) + list(slots.times):
            assert span in row["text"], f"span {span!r} is not verbatim in {row['text']!r}"
    print(f"OK spans   : every predicted span verbatim across {len(rows)} prompts")


def check_accuracy(model, rows, label):
    scores = evaluate(model, rows)
    for metric, floor in FLOORS.items():
        assert scores[metric] >= floor, \
            f"[{label}] {metric} {scores[metric]:.3f} below floor {floor}"
    print(f"OK accuracy: {label} ({scores['rows']} rows) " +
          "  ".join(f"{k} {scores[k]:.3f}" for k in FLOORS))
    return scores


def main():
    assert BUNDLE_PATH.exists(), \
        f"{BUNDLE_PATH} missing - run: python -m src.v3.model --export"
    check_freshness()

    model = V3Model.load()
    print(f"OK loaded  : {BUNDLE_PATH.name} ({BUNDLE_PATH.stat().st_size / 1e6:.1f} MB)")

    check_smoke(model)
    check_presentation(model)
    for query, wanted in AGGREGATION_SMOKE:
        got = model.predict(query).aggregation.value
        assert got == wanted, f"{query!r} -> {got}, want {wanted}"
    print(f"OK agg     : {len(AGGREGATION_SMOKE)} reductions chosen correctly")

    from src.v3 import dataset as v3_dataset

    # single utterances only - the multi-turn chats are test_conversations.py's job
    english = [r for r in v3_dataset.load(split="eval", lang="en") if r["source"] != "chats"]
    assert english, f"no English eval rows in {CSV_PATH.name}"
    check_spans_verbatim(model, english)
    check_time_normalized(model, english)
    check_always_decides(model, english)
    check_accuracy(model, english, "eval English")

    # Code-mixed rows are a diagnostic, not a gate: reported, never asserted.
    mixed = [r for r in v3_dataset.load(split="eval", lang="mixed") if r["source"] != "chats"]
    if mixed:
        scores = evaluate(model, mixed)
        print(f"INFO       : code-mixed ({scores['rows']} rows) intent {scores['intent']:.3f}  "
              f"variables {scores['variables']:.3f}  everything {scores['everything']:.3f}  "
              "(not a Model 1 target)")


if __name__ == "__main__":
    main()
