"""Self-check for the exported model. Run: python test_model.py

Fails if the bundle is missing, if a smoke query breaks, if a predicted span is not verbatim
in its prompt (Rules 4.1 / 4.2), or if accuracy on the hand-written English evaluation set
drops below the floors below. Retrain with `python src/nlu.py --export` first.
"""

from pathlib import Path

from src.build_dataset import EVAL_MANUAL, SPLITS
from src.data_loader import load_intents_csv
from src.nlu import BUNDLE_PATH, NLUModel, evaluate

ROOT = Path(__file__).parent

# Floors, not targets: set below the measured numbers so this catches regressions without
# going red on run-to-run noise. Raise them when the model genuinely improves.
FLOORS = {
    "weather_intent_accuracy": 0.92,   # measured 0.953
    "action_accuracy": 0.95,           # measured 0.979
    "aggregation_accuracy": 0.90,      # measured 0.950
    "location_f1": 0.90,               # measured 0.949
    "time_f1": 0.94,                   # measured 0.976
    "all_targets": 0.70,               # measured 0.753 - now 5 targets, harder eval rows
}

# (query, intent, action, locations, times) - unambiguous English, the V1 target language.
SMOKE = [
    ("Will it rain in Rajahmundry tomorrow morning?",
     "RAIN", "GET", ["Rajahmundry"], ["tomorrow morning"]),
    ("Compare the maximum temperature in Hyderabad and Vizag next week",
     "TEMPERATURE_MAX", "COMPARE", ["Hyderabad", "Vizag"], ["next week"]),
    ("Alert me if the wind speed crosses 40 kmph in Kakinada tonight",
     "WIND_SPEED", "ALERT", ["Kakinada"], ["tonight"]),
    ("What is the soil moisture in my field right now?",
     "SOIL_MOISTURE", "GET", ["my field"], ["right now"]),
    ("whats the temprature in Peddapuram, East Godavari at 6:45 pm",
     "TEMPERATURE", "GET", ["Peddapuram, East Godavari"], ["6:45 pm"]),
    ("humidity", "HUMIDITY", "GET", [], []),
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


def check_freshness():
    """A bundle older than the training data is a stale export - warn, do not fail."""
    train_csv = ROOT / SPLITS["train"][2]
    if train_csv.exists() and train_csv.stat().st_mtime > BUNDLE_PATH.stat().st_mtime:
        print("WARNING: data/processed/nlu_dataset.csv is newer than the bundle - "
              "rerun python src/nlu.py --export")


def check_smoke(model):
    for text, intent, action, locations, times in SMOKE:
        out = model.predict(text)
        assert out.weather_intent.value == intent, f"{text!r} -> {out.weather_intent.value}, want {intent}"
        assert out.action.value == action, f"{text!r} -> {out.action.value}, want {action}"
        assert out.entities.location == locations, f"{text!r} -> location {out.entities.location}, want {locations}"
        assert sorted(out.entities.time) == sorted(times), f"{text!r} -> time {out.entities.time}, want {times}"
    print(f"OK smoke   : {len(SMOKE)} queries predicted exactly")


def check_time_normalized(model, df):
    """Every raw time span gets exactly one canonical twin, in the same order (Rule 6.1)."""
    for text in df["text"]:
        entities = model.predict(text).entities
        assert len(entities.time_normalized) == len(entities.time), \
            f"{text!r} -> {len(entities.time)} spans but {len(entities.time_normalized)} normalized"
        for value in entities.time_normalized:
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
        actual = model.predict(query).entities.time_normalized
        assert actual == expected, f"{query!r} -> {actual}, want {expected}"
    print(f"OK time    : {len(canonical)} spellings folded to canonical form, "
          f"aligned 1:1 across {len(df)} prompts")


def check_spans_verbatim(model, df):
    """Rules 4.1 / 4.2: every returned span must be raw text lifted from the prompt."""
    for text in df["text"]:
        out = model.predict(text)
        for span in out.entities.location + out.entities.time:
            assert span in text, f"span {span!r} is not verbatim in {text!r}"
    print(f"OK spans   : every predicted span verbatim across {len(df)} prompts")


def check_accuracy(model, df, label):
    scores = evaluate(model, df)
    actual = {
        "weather_intent_accuracy": scores["weather_intent_accuracy"],
        "action_accuracy": scores["action_accuracy"],
        "aggregation_accuracy": scores["aggregation_accuracy"],
        "location_f1": scores["location_span"]["f1"],
        "time_f1": scores["time_span"]["f1"],
        "all_targets": scores["all_targets"],
    }
    for metric, floor in FLOORS.items():
        assert actual[metric] >= floor, f"[{label}] {metric} {actual[metric]:.3f} below floor {floor}"
    print(f"OK accuracy: {label} ({len(df)} rows) " +
          "  ".join(f"{k.replace('_accuracy', '').replace('_', ' ')} {v:.3f}" for k, v in actual.items()))
    return scores


def main():
    assert BUNDLE_PATH.exists(), f"{BUNDLE_PATH} missing - run: python src/nlu.py --export"
    check_freshness()

    model = NLUModel.load()
    print(f"OK loaded  : {BUNDLE_PATH.name} ({BUNDLE_PATH.stat().st_size / 1e6:.1f} MB)")

    check_smoke(model)
    for query, wanted in AGGREGATION_SMOKE:
        got = model.predict(query).aggregation.value
        assert got == wanted, f"{query!r} -> {got}, want {wanted}"
    print(f"OK agg     : {len(AGGREGATION_SMOKE)} reductions chosen correctly")

    evaluation = load_intents_csv(EVAL_MANUAL)
    english = evaluation[evaluation["lang"] == "en"]
    check_spans_verbatim(model, evaluation)
    check_time_normalized(model, evaluation)
    check_accuracy(model, english, "eval English")

    # Code-mixed rows are a diagnostic, not a gate: reported, never asserted.
    mixed = evaluate(model, evaluation[evaluation["lang"] == "mixed"])
    print(f"INFO       : code-mixed ({mixed['rows']} rows) intent {mixed['weather_intent_accuracy']:.3f}  "
          f"action {mixed['action_accuracy']:.3f}  all 4 {mixed['all_targets']:.3f}  (not a V1 target)")


if __name__ == "__main__":
    main()
