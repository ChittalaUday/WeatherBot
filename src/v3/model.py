"""
v3 model - one encoder, five heads, and no questions asked.

    python -m src.v3.model --export
    python -m src.v3.model "temperature in Guntur tomorrow in detail"

    text
      |
    TF-IDF (word 1-2 + char_wb 3-5)
      |
      +-- intent        6 coarse classes            (from v2)
      +-- variables     13 labels, multi-label      (from v2)
      +-- aggregation   6 classes                   (from v2)
      +-- detail        MINIMAL / NORMAL / FULL     <- new: how many columns
      +-- chart         5 kinds                     <- new: which picture
      +-- insights      9 labels, multi-label       <- new: which observations
      |
    span tagger (src/tagger.py)                      LOCATION / TIME, unchanged

The last three are what Python used to decide. Predicting them makes a bad chart a labelled
example rather than an argument about an if-statement - and lets the wording carry the
decision, so "in detail" and "full breakdown" both widen the table without either being
enumerated anywhere.

v3 also commits rather than asking: `assumed` records what it filled in, so the answer can
say "showing Angara, Jharkhand" instead of stopping to ask which Angara.
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
from src.v2.schema import Aggregation, Intent, Slots, Variable
from src.v3 import dataset as v3_dataset
from src.v3.schema import ChartKind, Detail, Insight, Presentation, V3Result

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = ROOT / "models" / "nlu_v3.joblib"
METRICS_PATH = ROOT / "models" / "metrics_v3.json"


def _multi_label_head(features, labels_per_row, classes):
    binarizer = MultiLabelBinarizer(classes=classes)
    targets = binarizer.fit_transform(labels_per_row)
    head = OneVsRestClassifier(LogisticRegression(max_iter=1000, C=4.0), n_jobs=-1)
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


class V3Model:
    """Extraction and presentation in one pass."""

    version = "v3"

    def __init__(self, vectorizer, heads, labels, thresholds, tagger):
        self.vectorizer = vectorizer
        self.heads = heads              # intent | variables | aggregation | detail | chart | insights
        self.labels = labels            # class lists for the multi-label heads
        self.thresholds = thresholds
        self.tagger = tagger

    def _multi(self, name: str, features) -> list[str]:
        probabilities = self.heads[name].predict_proba(features)[0]
        classes = self.labels[name]
        chosen = [label for label, score in zip(classes, probabilities)
                  if score >= self.thresholds[name]]
        if not chosen:
            chosen = [classes[int(probabilities.argmax())]]
        return sorted(chosen, key=lambda label: -probabilities[classes.index(label)])

    def predict(self, text: str) -> V3Result:
        features = self.vectorizer.transform([clean_text(text)])
        spans = self.tagger.predict(text)

        intent_scores = dict(zip(self.heads["intent"].classes_,
                                 self.heads["intent"].predict_proba(features)[0]))
        intent = max(intent_scores, key=intent_scores.get)
        variables = self._multi("variables", features)
        insights = self._multi("insights", features)

        return V3Result(
            text=text,
            intent=Intent(intent),
            aggregation=Aggregation(self.heads["aggregation"].predict(features)[0]),
            slots=Slots(
                variables=[Variable(name) for name in variables],
                locations=spans["location"],
                times=spans["time"],
                times_normalized=[normalize_time(span) for span in spans["time"]],
            ),
            presentation=Presentation(
                detail=Detail(self.heads["detail"].predict(features)[0]),
                chart=ChartKind(self.heads["chart"].predict(features)[0]),
                insights=[Insight(name) for name in insights],
            ),
            confidence={"intent": round(float(intent_scores[intent]), 4)},
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


def train(rows: list[dict], verbose: bool = True) -> V3Model:
    texts = [row["text"] for row in rows]
    vectorizer = build_vectorizer()
    features = vectorizer.fit_transform([clean_text(text) for text in texts])

    heads, labels, thresholds = {}, {}, {}

    heads["intent"] = CalibratedClassifierCV(LinearSVC(), cv=3)
    heads["intent"].fit(features, [row["intent"] for row in rows])

    for name, column, classes in (
        ("variables", "variables", [v.value for v in Variable]),
        ("insights", "insights", [i.value for i in Insight]),
    ):
        cut = int(len(rows) * 0.85)
        dev_head, dev_classes = _multi_label_head(features[:cut], [r[column] for r in rows[:cut]], classes)
        thresholds[name] = _calibrate(dev_head, features[cut:], rows[cut:], dev_classes, column)
        heads[name], labels[name] = _multi_label_head(features, [r[column] for r in rows], classes)

    for name, column in (("aggregation", "aggregation"), ("detail", "detail"), ("chart", "chart")):
        heads[name] = LinearSVC(class_weight="balanced")
        heads[name].fit(features, [row[column] for row in rows])

    tagger = SpanTagger(
        metric_nouns=[word for words in v2_dataset.VARIABLE_WORDS.values() for word in words],
        min_word_freq=choose_min_word_freq(texts, [row["locations"] for row in rows]),
    ).fit(texts, [{"location": r["locations"], "time": r["times"]} for r in rows])

    if verbose:
        print(f"trained on {len(rows)} rows | thresholds {thresholds} | "
              f"tagger vocab {len(tagger.vocab)}")
    return V3Model(vectorizer, heads, labels, thresholds, tagger)


def evaluate(model: V3Model, rows: list[dict]) -> dict:
    hits = {key: 0 for key in ("intent", "aggregation", "detail", "chart", "variables",
                               "locations", "times", "everything")}
    insight_tp = insight_fp = insight_fn = 0
    normalize = lambda spans: sorted(s.lower() for s in spans)

    for row in rows:
        prediction = model.predict(row["text"])
        got = {
            "intent": prediction.intent.value == row["intent"],
            "aggregation": prediction.aggregation.value == row.get("aggregation", "RAW"),
            "detail": prediction.presentation.detail.value == row["detail"],
            "chart": prediction.presentation.chart.value == row["chart"],
            "variables": {v.value for v in prediction.slots.variables} == set(row["variables"]),
            "locations": normalize(prediction.slots.locations) == normalize(row["locations"]),
            "times": normalize(prediction.slots.times) == normalize(row["times"]),
        }
        for key, value in got.items():
            hits[key] += value
        hits["everything"] += all(got.values())

        predicted = {i.value for i in prediction.presentation.insights}
        gold = set(row["insights"])
        insight_tp += len(predicted & gold)
        insight_fp += len(predicted - gold)
        insight_fn += len(gold - predicted)

    total = len(rows) or 1
    precision = insight_tp / (insight_tp + insight_fp) if insight_tp + insight_fp else 0
    recall = insight_tp / (insight_tp + insight_fn) if insight_tp + insight_fn else 0
    report = {key: round(value / total, 4) for key, value in hits.items()}
    report["rows"] = len(rows)
    report["insights_f1"] = round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()

    if args.export:
        train_rows = v3_dataset.load(split="train")
        if not train_rows:
            raise SystemExit("build the dataset first: python -m src.v3.dataset --build")
        model = train(train_rows)
        test_rows = v3_dataset.load(split="test")
        report = {
            "train_rows": len(train_rows),
            "test": evaluate(model, test_rows),
            "detail_phrasings": evaluate(model, [r for r in test_rows if r["source"] == "detail"]),
            "eval_utterances": evaluate(model, [r for r in v3_dataset.load(split="eval", lang="en")
                                                if r["source"] != "chats"]),
        }
        model.save()
        METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nsaved {BUNDLE_PATH}\nsaved {METRICS_PATH}\n")
        for name, scores in report.items():
            if isinstance(scores, dict):
                print(f"{name:18s} intent {scores['intent']:.1%}  vars {scores['variables']:.1%}  "
                      f"detail {scores['detail']:.1%}  chart {scores['chart']:.1%}  "
                      f"insights F1 {scores['insights_f1']:.3f}  all {scores['everything']:.1%}")
        return

    model = V3Model.load()
    print(model.predict(" ".join(args.text)).model_dump_json(indent=2))


if __name__ == "__main__":
    from src.v3.model import main as packaged_main   # pickle as src.v3.model.V3Model

    packaged_main()
