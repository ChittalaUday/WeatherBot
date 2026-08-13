"""
v2 model - one encoder, three heads, and the v1 span tagger reused unchanged.

    python -m src.v2.model --export        # train from data/v2_dataset.csv -> models/nlu_v2.joblib
    python -m src.v2.model --info
    python -m src.v2.model "rain and temperature in Guntur and Vizag tomorrow"

    text
      |
    TF-IDF (word 1-2 + char_wb 3-5)          shared, fit once
      |
      +-- intent head        LinearSVC, 6 coarse classes
      +-- variables head     one-vs-rest logistic regression, 13 labels, MULTI-LABEL
      +-- aggregation head   LinearSVC, 6 classes
      |
    span tagger (src/tagger.py)              LOCATION / TIME, unchanged from v1

The variables head is what v1 could not do: a query naming rain *and* temperature produces
two labels from one pass, instead of forcing a single winner.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.nlu import build_vectorizer, clean_text
from src.tagger import SpanTagger, choose_min_word_freq, normalize_time
from src.v2 import dataset as v2_dataset
from src.v2.schema import Aggregation, Intent, Slots, V2Result, Variable

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "models" / "nlu_v2.joblib"
METRICS_PATH = ROOT / "models" / "metrics_v2.json"

# Starting point only - train() calibrates the real threshold on a held-out slice of the
# training data. A hand-picked 0.35 dropped TEMPERATURE at 0.205 from "temperature, humidity
# and rainfall", which is exactly the multi-variable case v2 exists to get right.
VARIABLE_THRESHOLD = 0.35


class V2Model:
    """Coarse intent + multi-value slots behind one predict() call."""

    version = "v2"

    def __init__(self, vectorizer, intent_head, variable_head, variable_labels,
                 aggregation_head, tagger, variable_threshold: float = VARIABLE_THRESHOLD):
        self.vectorizer = vectorizer
        self.intent_head = intent_head
        self.variable_head = variable_head
        self.variable_labels = variable_labels
        self.aggregation_head = aggregation_head
        self.tagger = tagger
        self.variable_threshold = variable_threshold

    def predict(self, text: str) -> V2Result:
        features = self.vectorizer.transform([clean_text(text)])
        spans = self.tagger.predict(text)

        intent_scores = dict(zip(self.intent_head.classes_,
                                 self.intent_head.predict_proba(features)[0]))
        intent = max(intent_scores, key=intent_scores.get)

        probabilities = self.variable_head.predict_proba(features)[0]
        threshold = getattr(self, "variable_threshold", VARIABLE_THRESHOLD)
        chosen = [label for label, probability in zip(self.variable_labels, probabilities)
                  if probability >= threshold]
        if not chosen:                       # never return nothing: take the single best
            chosen = [self.variable_labels[int(probabilities.argmax())]]
        # keep the order stable and the strongest first, so the caller can truncate safely
        chosen.sort(key=lambda label: -probabilities[self.variable_labels.index(label)])

        aggregation = self.aggregation_head.predict(features)[0]
        return V2Result(
            text=text,
            intent=Intent(intent),
            aggregation=Aggregation(aggregation),
            slots=Slots(
                variables=[Variable(label) for label in chosen],
                locations=spans["location"],
                times=spans["time"],
                times_normalized=[normalize_time(span) for span in spans["time"]],
            ),
            confidence={
                "intent": round(float(intent_scores[intent]), 4),
                "variables": round(float(max(probabilities[self.variable_labels.index(label)]
                                             for label in chosen)), 4),
            },
            scores={label: round(float(score), 4) for label, score in intent_scores.items()},
        )

    def save(self, path=BUNDLE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=BUNDLE_PATH):
        return joblib.load(path)


def _fit_variable_head(features, rows, labels):
    binarizer = MultiLabelBinarizer(classes=labels)
    targets = binarizer.fit_transform([row["variables"] for row in rows])
    head = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=4.0), n_jobs=-1)
    head.fit(features, targets)
    return head


def calibrate_threshold(head, features, rows, labels,
                        candidates=(0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5)) -> tuple[float, float]:
    """Pick the per-label cut that maximises micro-F1 on held-out rows."""
    probabilities = head.predict_proba(features)
    best = (VARIABLE_THRESHOLD, 0.0)
    for threshold in candidates:
        true_positive = false_positive = false_negative = 0
        for row, scores in zip(rows, probabilities):
            predicted = {label for label, score in zip(labels, scores) if score >= threshold}
            if not predicted:
                predicted = {labels[int(scores.argmax())]}
            gold = set(row["variables"])
            true_positive += len(predicted & gold)
            false_positive += len(predicted - gold)
            false_negative += len(gold - predicted)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        if f1 > best[1]:
            best = (threshold, f1)
    return best


def train(rows: list[dict], verbose: bool = True) -> V2Model:
    texts = [row["text"] for row in rows]
    vectorizer = build_vectorizer()
    # a list, not a generator: FeatureUnion feeds the input to each transformer in turn,
    # and the second one would find the generator already exhausted
    features = vectorizer.fit_transform([clean_text(text) for text in texts])

    intent_head = CalibratedClassifierCV(LinearSVC(), cv=3)    # LinearSVC + calibration is
    intent_head.fit(features, [row["intent"] for row in rows])  # far cheaper than SVC(probability)

    labels = [v.value for v in Variable]
    # calibrate the multi-label cut on a held-out slice, then refit on everything
    cut = int(len(rows) * 0.85)
    dev_head = _fit_variable_head(features[:cut], rows[:cut], labels)
    threshold, dev_f1 = calibrate_threshold(dev_head, features[cut:], rows[cut:], labels)
    variable_head = _fit_variable_head(features, rows, labels)

    aggregation_head = LinearSVC(class_weight="balanced")
    aggregation_head.fit(features, [row.get("aggregation", "RAW") for row in rows])

    spans = [{"location": row["locations"], "time": row["times"]} for row in rows]
    tagger = SpanTagger(
        metric_nouns=[word for words in _variable_words() for word in words],
        min_word_freq=choose_min_word_freq(texts, [row["locations"] for row in rows]),
    ).fit(texts, spans)

    if verbose:
        print(f"trained on {len(rows)} rows | {len(labels)} variable labels | "
              f"threshold {threshold} (dev F1 {dev_f1:.3f}) | tagger vocab {len(tagger.vocab)}")
    return V2Model(vectorizer, intent_head, variable_head, labels, aggregation_head, tagger,
                   variable_threshold=threshold)


def _variable_words():
    return list(v2_dataset.VARIABLE_WORDS.values())


def evaluate(model: V2Model, rows: list[dict]) -> dict:
    """Per-head accuracy, plus multi-label precision/recall - a single accuracy would hide
    whether the second variable of a two-variable question was found."""
    intent_hits = aggregation_hits = exact_variables = location_hits = time_hits = whole = 0
    true_positive = false_positive = false_negative = 0
    normalize = lambda spans: sorted(s.lower() for s in spans)

    for row in rows:
        prediction = model.predict(row["text"])
        predicted_variables = {v.value for v in prediction.slots.variables}
        gold_variables = set(row["variables"])

        intent_ok = prediction.intent.value == row["intent"]
        aggregation_ok = prediction.aggregation.value == row.get("aggregation", "RAW")
        variables_ok = predicted_variables == gold_variables
        location_ok = normalize(prediction.slots.locations) == normalize(row["locations"])
        time_ok = normalize(prediction.slots.times) == normalize(row["times"])

        intent_hits += intent_ok
        aggregation_hits += aggregation_ok
        exact_variables += variables_ok
        location_hits += location_ok
        time_hits += time_ok
        whole += intent_ok and variables_ok and location_ok and time_ok and aggregation_ok

        true_positive += len(predicted_variables & gold_variables)
        false_positive += len(predicted_variables - gold_variables)
        false_negative += len(gold_variables - predicted_variables)

    total = len(rows) or 1
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return {
        "rows": len(rows),
        "intent_accuracy": round(intent_hits / total, 4),
        "aggregation_accuracy": round(aggregation_hits / total, 4),
        "variables_exact_set": round(exact_variables / total, 4),
        "variables_precision": round(precision, 4),
        "variables_recall": round(recall, 4),
        "variables_f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        "location_exact": round(location_hits / total, 4),
        "time_exact": round(time_hits / total, 4),
        "all_slots": round(whole / total, 4),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*")
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--info", action="store_true")
    args = parser.parse_args()

    if args.export:
        train_rows = v2_dataset.load(split="train")
        model = train(train_rows)
        test_rows = v2_dataset.load(split="test")
        report = {
            "train_rows": len(train_rows),
            "test": evaluate(model, test_rows),
            # standalone utterances only: a follow-up carries no variable in its words, so
            # per-utterance scoring of "and there?" measures nothing
            "eval_utterances": evaluate(model, [r for r in v2_dataset.load(split="eval", lang="en")
                                                if r["source"] != "chats"]),
            "eval_conversation_openers": evaluate(
                model, [r for r in v2_dataset.load(split="eval") if r["source"] == "chats"
                        and r["turn"] == 0]),
            "multivariable_only": evaluate(model, [r for r in test_rows if len(r["variables"]) > 1]),
            # follow-up turns are scored by test_conversations.py, which replays them through
            # the context engine - judging them per-utterance would be meaningless
            "first_turns_only": evaluate(model, [r for r in test_rows if r["turn"] == 0]),
        }
        model.save()
        METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nsaved {BUNDLE_PATH}\nsaved {METRICS_PATH}\n")
        for name, scores in report.items():
            if isinstance(scores, dict):
                print(f"{name:20s} intent {scores['intent_accuracy']:.1%}  "
                      f"vars F1 {scores['variables_f1']:.3f} (exact {scores['variables_exact_set']:.1%})  "
                      f"loc {scores['location_exact']:.1%}  time {scores['time_exact']:.1%}  "
                      f"all {scores['all_slots']:.1%}")
        return

    model = V2Model.load()
    if args.info:
        import pickle

        size = lambda obj: len(pickle.dumps(obj)) / 1e6
        print(f"bundle        {BUNDLE_PATH.stat().st_size / 1e6:.1f} MB")
        print(f"  vectorizer  {size(model.vectorizer):6.2f} MB")
        print(f"  intent      {size(model.intent_head):6.2f} MB  {len(model.intent_head.classes_)} classes")
        print(f"  variables   {size(model.variable_head):6.2f} MB  "
              f"{len(model.variable_labels)} labels, multi-label")
        print(f"  aggregation {size(model.aggregation_head):6.2f} MB")
        print(f"  tagger      {size(model.tagger):6.2f} MB")
        return

    print(model.predict(" ".join(args.text)).model_dump_json(indent=2))


if __name__ == "__main__":
    # Re-enter through the package so pickled objects record src.v2.model.V2Model rather
    # than __main__.V2Model - otherwise the bundle only loads back in this exact script.
    from src.v2.model import main as packaged_main

    packaged_main()
