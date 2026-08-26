"""
v4 model - four heads, one tagger, and everything else derived.

    python -m src.v4.model --export
    python -m src.v4.model "should i spray fertilizer on my cotton field in Guntur tomorrow"

    text
      |
    TF-IDF (word 1-2 + char_wb 3-5)          shared encoder, one fit
      |
      +-- intent        11 classes                    what kind of request
      +-- variables     10 labels, multi-label        which measurements
      +-- activity      12 classes, ADVICE turns only what to decide
      +-- aggregation   6 classes                     which reduction
      |
    span tagger (src/tagger.py)                       LOCATION / TIME
      |
    gazetteer (src/v4/entities.py)                    sport / crop / material / ...
      |
    lookups                                           weather_intent, action,
                                                      sub_activity, time_bucket

v3 trained six heads and scored them with one `everything` number. Half of what it predicted
did not need a model at all: weather_intent is a function of the time span, action is a
function of the activity, and entities are closed vocabularies. Predicting those is how you
get a model that says FORECAST while its own time slot says "yesterday".

So v4 trains only what is genuinely ambiguous in the text, and derives the rest from what it
trained. The report is per component, because "overall 91%" never told anyone what to fix.

Two constraints are enforced at predict time rather than hoped for, because they are dataset
invariants and a classifier has no reason to respect them:

    intent is conversational or declined  ->  no variables, no window, no activity
    intent is not ADVICE                  ->  activity NONE
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.nlu import build_vectorizer, clean_text
from src.tagger import GENERIC_PLACE_WORDS, SpanTagger, normalize_time, tokenize
from src.v4 import dataset as v4_dataset
from src.v4.entities import extract as extract_entities
from src.v4.schema import (
    NO_DATA_NEEDED,
    Activity,
    Aggregation,
    Intent,
    Slots,
    V4Result,
    Variable,
    WeatherIntent,
    sub_activity_for,
    weather_intent_for,
)

BUNDLE_PATH = ROOT / "models" / "nlu_v4.joblib"
METRICS_PATH = ROOT / "models" / "metrics_v4.json"


def min_word_freq(texts, location_spans, candidates=(20, 40, 80, 120, 160, 200, 300, 500)) -> int:
    """Smallest frequency cut at which no *real* place name survives in the word vocabulary.

    Below the cut the tagger can satisfy training by memorising village names and then fails
    on the first village it has not met; above it, only context and word shape are left, which
    is what transfers.

    The share test is the difference from src.tagger.choose_min_word_freq. One row reading
    "Kinshasa, Democratic Republic of the Congo" puts "the" inside a location span, and a
    membership test then treats the commonest word in English as a place name - no cut clears
    it, the ceiling is returned, and the vocabulary collapses to 48 words. A word only counts
    as a place name if a real share of its uses are inside a span.
    """
    counts = Counter(w.lower() for text in texts for _, _, w in tokenize(text))
    # Only the head of a qualified address is the name worth protecting. "Guntur, Andhra
    # Pradesh" carries a village nobody can enumerate and a state from a closed set of 36 -
    # memorising "pradesh" is correct behaviour, and counting it as a place name pins the cut
    # at its ceiling for the same reason "the" did.
    inside = Counter(w.lower() for spans in location_spans for span in spans
                     for _, _, w in tokenize(span.split(",")[0]))
    names = {w for w, n in inside.items()
             if w.isalpha() and w not in GENERIC_PLACE_WORDS and n / counts[w] >= 0.5}
    for cut in candidates:
        if not ({w for w, n in counts.items() if n >= cut} & names):
            return cut
    return candidates[-1]


# A duration is tenseless without its qualifier, and the tagger keeps dropping it: "last 7
# days" comes back as "7 days", which normalises to "next 7 days" - the past becomes the
# future. This is a lexical rule, not a judgment, so it is applied here rather than hoped for
# from more training rows (which measurably made the span boundary worse).
TIME_QUALIFIERS = ("last", "past", "previous", "prior", "next", "coming", "this", "these",
                   "recent", "upcoming", "following")

# "history of rainfall", "rainfall history" - the word is a noun here, and the tagger will not
# mark it: the per-template cap starves the pattern (42 rows out of 24,000). It is a fixed
# vocabulary, so it is matched rather than learned - the same call as the qualifier above.
HISTORY_NOUNS = ("historical data", "historical records", "past records", "previous records",
                 "history data", "past data", "history", "historical")


def _with_qualifier(text: str, span: str) -> str:
    """Extend a time span left over "last" / "next" / "the past" and friends."""
    match = re.search(rf"((?:the\s+)?(?:{'|'.join(TIME_QUALIFIERS)})\s+){re.escape(span)}",
                      text, re.I)
    return match.group(0).strip() if match else span


def _history_noun(text: str) -> str:
    """The history word in the sentence, if any. Longest first so "historical data" wins."""
    for word in HISTORY_NOUNS:
        m = re.search(rf"\b{re.escape(word)}\b", text, re.I)
        if m:
            return m.group(0)
    return ""


def _merge_adjacent(text: str, spans: list) -> list:
    """Join time spans separated only by a connector - a split date range is still one range.

    "11 jan 2026 and 17 jan 2026" came back as ["11", "2026 and 17 jan 2026"], and only the
    first span is normalised, so the window was read off "11".
    """
    if len(spans) < 2:
        return spans
    merged, current = [], spans[0]
    for nxt in spans[1:]:
        start = text.lower().find(current.lower())
        gap_from = start + len(current) if start >= 0 else -1
        gap_to = text.lower().find(nxt.lower(), gap_from) if gap_from >= 0 else -1
        between = text[gap_from:gap_to].strip().lower() if 0 <= gap_from <= gap_to else None
        # a short, purely-alphabetic gap is part of the same expression, not a clause break:
        # "11 [jan ]2026 and 17 jan 2026" splits on a month name, not on a connector
        joinable = between is not None and len(between) <= 12 and (
            between in ("", "-") or re.fullmatch(r"[a-z ]*", between))
        if joinable:
            current = text[start:gap_to + len(nxt)].strip()
        else:
            merged.append(current)
            current = nxt
    merged.append(current)
    return merged


def _multi_label_head(features, labels_per_row, classes):
    binarizer = MultiLabelBinarizer(classes=classes)
    targets = binarizer.fit_transform(labels_per_row)
    # liblinear, not the default lbfgs. Each one-vs-rest sub-problem here is close to
    # separable, and lbfgs's line search overflows computing the L2 penalty on the trial step -
    # 24 numeric warnings per fit. The step is rejected and the model is fine, but liblinear
    # reaches the same micro-F1 five times faster with none of the noise.
    head = OneVsRestClassifier(
        LogisticRegression(max_iter=1000, C=4.0, solver="liblinear"), n_jobs=-1)
    head.fit(features, targets)
    return head, list(binarizer.classes_)


def _calibrate(head, features, rows, classes, key,
               candidates=(0.2, 0.25, 0.3, 0.35, 0.4, 0.5)) -> float:
    """Pick the multi-label cut that maximises micro-F1 on held-out rows."""
    probabilities = head.predict_proba(features)
    best = (0.35, 0.0)
    for threshold in candidates:
        tp = fp = fn = 0
        for row, scores in zip(rows, probabilities):
            predicted = {label for label, score in zip(classes, scores) if score >= threshold}
            if not predicted:
                predicted = {classes[int(scores.argmax())]}
            gold = set(row[key])
            tp += len(predicted & gold)
            fp += len(predicted - gold)
            fn += len(gold - predicted)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        if f1 > best[1]:
            best = (threshold, f1)
    return best[0]


class V4Model:
    """Four heads, a span tagger, and the lookups that finish the turn."""

    version = "v4"

    def __init__(self, vectorizer, heads, labels, thresholds, tagger):
        self.vectorizer = vectorizer
        self.heads = heads              # intent | variables | activity | aggregation
        self.labels = labels
        self.thresholds = thresholds
        self.tagger = tagger

    def predict(self, text: str) -> V4Result:
        features = self.vectorizer.transform([clean_text(text)])
        spans = self.tagger.predict(text)

        intent_scores = dict(zip(self.heads["intent"].classes_,
                                 self.heads["intent"].predict_proba(features)[0]))
        intent = Intent(str(max(intent_scores, key=lambda k: intent_scores[k])))
        # str(): sklearn hands back numpy.str_, which is a str subclass the enum accepts
        # but pydantic warns about on every serialisation
        aggregation = Aggregation(str(self.heads["aggregation"].predict(features)[0]))

        # A greeting has no window, no measurement and nothing to decide. Letting the other
        # heads answer anyway is how "hi" ends up fetching soil moisture for a village.
        if intent in NO_DATA_NEEDED:
            # ...with one exception: "set my location to Guntur" is the whole point of the
            # CHANGE_LOCATION turn, and blanking its slots throws away the only thing the
            # backend needs from it.
            keep = Slots(locations=spans["location"]) if intent is Intent.CHANGE_LOCATION else Slots()
            return V4Result(
                text=text, intent=intent, weather_intent=WeatherIntent.NONE,
                activity=Activity.NONE, aggregation=Aggregation.RAW, slots=keep,
                confidence={"intent": round(float(intent_scores[intent]), 4)},
                scores={k: round(float(v), 4) for k, v in intent_scores.items()})

        spans["time"] = _merge_adjacent(text, [_with_qualifier(text, span)
                                               for span in spans["time"]])
        if not spans["time"]:
            spans["time"] = [found] if (found := _history_noun(text)) else []
        variables = self._multi("variables", features)
        # A span that normalises to nothing is a preposition the tagger swept up ("for" in
        # "for whole day"). Keeping it would be harmless except that weather_intent is derived
        # from times_normalized[0] - so one stray word silently moves the whole window.
        timed = [(raw, normalize_time(raw)) for raw in spans["time"]]
        timed = [pair for pair in timed if pair[1]] or []
        times_raw = [raw for raw, _ in timed]
        times_normalized = [norm for _, norm in timed]
        entities = extract_entities(text)

        activity = Activity.NONE
        if intent is Intent.ADVICE:
            activity = Activity(str(self.heads["activity"].predict(features)[0]))

        return V4Result(
            text=text, intent=intent,
            # derived from the time slot, so the window can never contradict the span
            weather_intent=weather_intent_for(times_normalized[0] if times_normalized else None),
            activity=activity, aggregation=aggregation,
            slots=Slots(
                variables=[Variable(name) for name in variables],
                locations=spans["location"], times=times_raw,
                times_normalized=times_normalized, entities=entities,
            ),
            confidence={"intent": round(float(intent_scores[intent]), 4)},
            scores={k: round(float(v), 4) for k, v in intent_scores.items()},
        )

    def _multi(self, name: str, features) -> list[str]:
        probabilities = self.heads[name].predict_proba(features)[0]
        classes = self.labels[name]
        chosen = [label for label, score in zip(classes, probabilities)
                  if score >= self.thresholds[name]]
        if not chosen:
            chosen = [classes[int(probabilities.argmax())]]
        return sorted(chosen, key=lambda label: -probabilities[classes.index(label)])

    def save(self, path=BUNDLE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=BUNDLE_PATH):
        return joblib.load(path)


def train(rows: list[dict], verbose: bool = True) -> V4Model:
    texts = [row["text"] for row in rows]
    vectorizer = build_vectorizer()
    features = vectorizer.fit_transform([clean_text(text) for text in texts])

    heads, labels, thresholds = {}, {}, {}

    # isotonic, not the default sigmoid. Platt scaling fits a sigmoid by Newton steps, and on
    # near-perfectly separated classes with only ~100 rows each (UNCLEAR, GOODBYE) those steps
    # divide by zero - 1,446 numeric warnings per fit, and a slightly worse model. Isotonic is
    # a monotonic step fit with no exponentials: 0 warnings, and it scored better here.
    heads["intent"] = CalibratedClassifierCV(LinearSVC(), cv=3, method="isotonic")
    heads["intent"].fit(features, [row["intent"] for row in rows])

    # Shuffle before slicing. The rows come out of the builder grouped by source, so a plain
    # rows[:cut] / rows[cut:] split hands the calibrator a holdout that is 70% chitchat - rows
    # with no variables at all - and the threshold gets tuned against noise. Seeded, so the
    # chosen threshold is reproducible.
    cut = int(len(rows) * 0.85)
    order = list(range(len(rows)))
    random.Random(13).shuffle(order)
    dev, hold = order[:cut], order[cut:]
    classes = [v.value for v in Variable]
    dev_head, dev_classes = _multi_label_head(features[dev],
                                              [rows[i]["variables"] for i in dev], classes)
    thresholds["variables"] = _calibrate(dev_head, features[hold], [rows[i] for i in hold],
                                         dev_classes, "variables")
    heads["variables"], labels["variables"] = _multi_label_head(
        features, [r["variables"] for r in rows], classes)

    heads["aggregation"] = LinearSVC(class_weight="balanced")
    heads["aggregation"].fit(features, [row["aggregation"] for row in rows])

    # The activity head is trained on ADVICE rows only. Trained on everything it would spend
    # its capacity learning that a greeting is NONE - which the intent head already knows, and
    # which predict() enforces outright.
    advice = [row for row in rows if row["intent"] == Intent.ADVICE.value]
    advice_features = vectorizer.transform([clean_text(row["text"]) for row in advice])
    heads["activity"] = LinearSVC(class_weight="balanced")
    heads["activity"].fit(advice_features, [row["activity"] for row in advice])

    tagger = SpanTagger(
        metric_nouns=[word for words in v4_dataset.VARIABLE_WORDS.values() for word in words],
        min_word_freq=min_word_freq(texts, [row["locations"] for row in rows]),
    ).fit(texts, [{"location": r["locations"], "time": r["times"]} for r in rows])

    if verbose:
        print(f"trained on {len(rows)} rows ({len(advice)} ADVICE) | "
              f"variable threshold {thresholds['variables']} | tagger vocab {len(tagger.vocab)}")
    return V4Model(vectorizer, heads, labels, thresholds, tagger)


def evaluate(model: V4Model, rows: list[dict]) -> dict:
    """Per component, including the derived ones - those measure the tagger and the lookups
    end to end, which is the only way a derived field can be wrong."""
    keys = ("intent", "weather_intent", "variables", "activity", "aggregation",
            "locations", "times", "sub_activity", "everything")
    hits = {key: 0 for key in keys}
    seen = {key: 0 for key in keys}
    ent_tp = ent_fp = ent_fn = 0
    var_tp = var_fp = var_fn = 0
    normalise = lambda spans: sorted(s.lower() for s in spans)

    for row in rows:
        prediction = model.predict(row["text"])
        got = {
            "intent": prediction.intent.value == row["intent"],
            "weather_intent": prediction.weather_intent.value == row["weather_intent"],
            "variables": {v.value for v in prediction.slots.variables} == set(row["variables"]),
            "aggregation": prediction.aggregation.value == row["aggregation"],
            "locations": normalise(prediction.slots.locations) == normalise(row["locations"]),
            "times": normalise(prediction.slots.times) == normalise(row["times"]),
            "sub_activity": sub_activity_for(prediction.activity, prediction.slots.entities,
                                             row["text"]) == row["sub_activity"],
        }
        # activity is only meaningful on ADVICE turns; scoring NONE everywhere else would
        # report 90% for a head that had not been asked a question
        if row["intent"] == Intent.ADVICE.value:
            got["activity"] = prediction.activity.value == row["activity"]

        for key, value in got.items():
            hits[key] += value
            seen[key] += 1
        hits["everything"] += all(got.values())
        seen["everything"] += 1

        predicted_vars = {v.value for v in prediction.slots.variables}
        gold_vars = set(row["variables"])
        var_tp += len(predicted_vars & gold_vars)
        var_fp += len(predicted_vars - gold_vars)
        var_fn += len(gold_vars - predicted_vars)

        predicted_ents = {(k, v.lower()) for k, vs in prediction.slots.entities.items() for v in vs}
        gold_ents = {(k, v.lower()) for k, vs in row["entities"].items() for v in vs}
        ent_tp += len(predicted_ents & gold_ents)
        ent_fp += len(predicted_ents - gold_ents)
        ent_fn += len(gold_ents - predicted_ents)

    f1 = lambda tp, fp, fn: round(2 * tp / (2 * tp + fp + fn), 4) if tp else 0.0
    report = {key: round(hits[key] / seen[key], 4) if seen[key] else None for key in keys}
    report["variables_f1"] = f1(var_tp, var_fp, var_fn)
    report["entities_f1"] = f1(ent_tp, ent_fp, ent_fn)
    report["rows"] = len(rows)
    report["advice_rows"] = seen["activity"]
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.export:
        train_rows = v4_dataset.load(split="train")
        if not train_rows:
            raise SystemExit("build the dataset first: python -m src.v4.dataset --build")
        model = train(train_rows)
        report = {
            "train_rows": len(train_rows),
            "test": evaluate(model, v4_dataset.load(split="test")),
            "eval": evaluate(model, v4_dataset.load(split="eval")),
        }
        for name in ("implicit", "confusion"):
            rows = [r for r in v4_dataset.load(split="test") if r["source"] == name]
            if rows:
                report[name] = evaluate(model, rows)
        model.save()
        METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nsaved {BUNDLE_PATH}\nsaved {METRICS_PATH}\n")
        for name, scores in report.items():
            if not isinstance(scores, dict):
                continue
            print(f"{name:11s} intent {scores['intent']:.1%}  weather {scores['weather_intent']:.1%}  "
                  f"vars {scores['variables']:.1%}  activity "
                  f"{scores['activity']:.1%}  agg {scores['aggregation']:.1%}  "
                  f"loc {scores['locations']:.1%}  time {scores['times']:.1%}  "
                  f"sub {scores['sub_activity']:.1%}  ent F1 {scores['entities_f1']:.3f}  "
                  f"all {scores['everything']:.1%}")
        return

    model = V4Model.load()
    print(model.predict(" ".join(args.text)).model_dump_json(indent=2))


if __name__ == "__main__":
    from src.v4.model import main as packaged_main  # pickle as src.v4.model.V4Model

    packaged_main()
