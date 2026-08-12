"""
Train, export and serve the WeatherBot NLU model.

One module owns the model so training and inference cannot drift apart: the notebook
imports the same `train()` / `NLUModel` used by the exported bundle.

    python src/nlu.py --export                       # train, evaluate, write models/
    python src/nlu.py "will it rain in Guntur at 6pm tomorrow?"

English is the target language. Hindi/Telugu code-mixed prompts are carried in the
evaluation set as a diagnostic only (`lang == "mixed"`), and are reported separately so they
never flatter or drag the headline English number.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.build_dataset import EVAL_MANUAL, NOUNS, SPLITS
from src.data_loader import load_intents_csv
from src.schema import Entities, NLUOutput
from src.tagger import SpanTagger, choose_min_word_freq, normalize_time

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_PATH = ROOT / "models/nlu_pipeline.joblib"
METRICS_PATH = ROOT / "models/metrics.json"
TARGETS = ["weather_intent", "action", "location", "time"]

import re


def clean_text(text: str) -> str:
    """Classifier input. Entity spans are read off the RAW text (Rules 4.1 / 4.2)."""
    text = re.sub(r"[^a-z0-9:\s]", " ", text.lower())  # keep ':' so "5:30 am" survives
    return re.sub(r"\s+", " ", text).strip()


def build_vectorizer():
    """Word n-grams for phrasing, character n-grams for spelling.

    Character n-grams are what let "temprature" reach TEMPERATURE: word n-grams treat it as a
    token they have never seen, while "tempr"/"ratu" overlap the correct spelling heavily.
    """
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2))),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ])


class NLUModel:
    """The 4 targets of MODEL_RULES.md behind one predict() call."""

    def __init__(self, vectorizer, intent_model, action_model, tagger):
        self.vectorizer = vectorizer
        self.intent_model = intent_model
        self.action_model = action_model
        self.tagger = tagger

    def predict(self, text: str) -> NLUOutput:
        features = self.vectorizer.transform([clean_text(text)])
        spans = self.tagger.predict(text)
        return NLUOutput(
            weather_intent=self.intent_model.predict(features)[0],
            action=self.action_model.predict(features)[0],
            entities=Entities(
                location=spans["location"],
                time=spans["time"],
                time_normalized=[normalize_time(span) for span in spans["time"]],
            ),
        )

    def confidence(self, text: str) -> float:
        """Max intent probability - the hook for a downstream 'ask the user' fallback."""
        return float(self.intent_model.predict_proba(
            self.vectorizer.transform([clean_text(text)]))[0].max())

    def top_intents(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Best k intents with probabilities - what the UI offers when it has to ask."""
        features = self.vectorizer.transform([clean_text(text)])
        probabilities = self.intent_model.predict_proba(features)[0]
        ranked = sorted(zip(self.intent_model.classes_, probabilities), key=lambda p: -p[1])
        return [(intent, float(p)) for intent, p in ranked[:k]]

    def save(self, path=BUNDLE_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=BUNDLE_PATH):
        """Needs src/ importable: the tagger's feature builder lives in src.tagger, so
        training and inference stay on byte-identical features."""
        return joblib.load(path)


def train(df, verbose=True):
    """Fit vectorizer, both classifiers and the span tagger on one annotated dataframe."""
    vectorizer = build_vectorizer()
    features = vectorizer.fit_transform(df["text"].apply(clean_text))

    # Linear SVC wins model selection in the notebook every run; probability=True buys the
    # confidence score for the downstream fallback.
    models = {}
    for target in ("weather_intent", "action"):
        models[target] = SVC(kernel="linear", probability=True, random_state=42)
        models[target].fit(features, df[target])

    spans = [{"location": loc, "time": time} for loc, time in zip(df["location"], df["time"])]
    min_freq = choose_min_word_freq(df["text"], df["location"])
    tagger = SpanTagger(metric_nouns=[n for nouns in NOUNS.values() for n in nouns],
                        min_word_freq=min_freq).fit(list(df["text"]), spans)
    if verbose:
        print(f"trained on {len(df)} rows | tagger min_word_freq={min_freq} "
              f"({len(tagger.vocab)} words kept)")
    return NLUModel(vectorizer, models["weather_intent"], models["action"], tagger)


def span_scores(gold_spans, predicted_spans):
    tp = fp = fn = 0
    for gold, predicted in zip(gold_spans, predicted_spans):
        g, p = Counter(s.lower() for s in gold), Counter(s.lower() for s in predicted)
        hit = sum((g & p).values())
        tp, fp, fn = tp + hit, fp + sum(p.values()) - hit, fn + sum(g.values()) - hit
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}


def evaluate(model, df):
    """Per-target scores plus the number that matters: all 4 targets right on one prompt."""
    predictions = [model.predict(text) for text in df["text"]]
    norm = lambda spans: sorted(s.lower() for s in spans)

    correct = {
        "weather_intent": [p.weather_intent.value == g for p, g in zip(predictions, df["weather_intent"])],
        "action": [p.action.value == g for p, g in zip(predictions, df["action"])],
        "location": [norm(p.entities.location) == norm(g) for p, g in zip(predictions, df["location"])],
        "time": [norm(p.entities.time) == norm(g) for p, g in zip(predictions, df["time"])],
    }
    return {
        "rows": len(df),
        "weather_intent_accuracy": round(sum(correct["weather_intent"]) / len(df), 4),
        "action_accuracy": round(sum(correct["action"]) / len(df), 4),
        "location_span": span_scores(df["location"], [p.entities.location for p in predictions]),
        "time_span": span_scores(df["time"], [p.entities.time for p in predictions]),
        "all_four_targets": round(sum(all(c[i] for c in correct.values()) for i in range(len(df))) / len(df), 4),
    }


def describe(model, path=BUNDLE_PATH):
    """What is actually inside the exported bundle, and what it costs to run."""
    import pickle
    import platform
    import time

    import sklearn

    size = lambda obj: len(pickle.dumps(obj)) / 1e6
    word, char = (t[1] for t in model.vectorizer.transformer_list)
    print(f"bundle          {Path(path).stat().st_size / 1e6:.1f} MB  ({path})")
    print(f"  vectorizer    {size(model.vectorizer):6.2f} MB  "
          f"{len(word.vocabulary_):,} word + {len(char.vocabulary_):,} char features")
    for name, clf in (("intent", model.intent_model), ("action", model.action_model)):
        print(f"  {name:<12s}  {size(clf):6.2f} MB  SVC linear, "
              f"{clf.support_vectors_.shape[0]:,} support vectors, {len(clf.classes_)} classes")
    print(f"  tagger        {size(model.tagger):6.2f} MB  "
          f"{len(model.tagger.vectorizer.feature_names_):,} token features, "
          f"{len(model.tagger.model.classes_)} BIO labels, {len(model.tagger.vocab)} words kept")

    query = "compare the max temp in Peddapuram, East Godavari and Nokha at 6:45 pm tomorrow"
    model.predict(query)                                   # warm up
    start = time.perf_counter()
    for _ in range(50):
        model.predict(query)
    print(f"  latency       {(time.perf_counter() - start) / 50 * 1000:.1f} ms per query, single core")
    print(f"  runtime       python {platform.python_version()}, scikit-learn {sklearn.__version__}")
    if METRICS_PATH.exists():
        scores = json.loads(METRICS_PATH.read_text())["eval (English)"]
        print(f"  eval English  intent {scores['weather_intent_accuracy']:.1%}, "
              f"action {scores['action_accuracy']:.1%}, "
              f"all 4 targets {scores['all_four_targets']:.1%}")


def repl(model):
    """Type a query, get the answer. The bundle loads once, so each query costs ~5 ms."""
    import time

    print("WeatherBot NLU - type a query, blank line or Ctrl-D to quit\n")
    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not query or query.lower() in {"quit", "exit", "q"}:
            return
        start = time.perf_counter()
        out = model.predict(query)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  intent    {out.weather_intent.value}  ({model.confidence(query):.0%} confident)")
        print(f"  action    {out.action.value}")
        print(f"  location  {out.entities.location or '-'}")
        print(f"  time      {out.entities.time or '-'}"
              f"{'  ->  ' + str(out.entities.time_normalized) if out.entities.time else ''}")
        print(f"  {out.model_dump_json()}   [{elapsed:.0f} ms]\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text", nargs="*", help="predict this query (no arguments: interactive)")
    parser.add_argument("--export", action="store_true", help="train, evaluate and save models/")
    parser.add_argument("--info", action="store_true", help="describe the exported bundle")
    args = parser.parse_args()

    if args.info:
        describe(NLUModel.load())
        return

    if not args.export:
        model = NLUModel.load()
        if not args.text:
            repl(model)
            return
        query = " ".join(args.text)
        print(model.predict(query).model_dump_json(indent=2))
        print(f"# intent confidence {model.confidence(query):.1%}")
        return

    train_df = load_intents_csv(ROOT / SPLITS["train"][2])
    model = train(train_df)

    evaluation = load_intents_csv(EVAL_MANUAL)
    english = evaluation[evaluation["lang"] == "en"]
    report = {
        "train_rows": len(train_df),
        "test_generated": evaluate(model, load_intents_csv(ROOT / SPLITS["test"][2])),
        "eval_english": evaluate(model, english),                                  # headline
        "eval_code_mixed": evaluate(model, evaluation[evaluation["lang"] == "mixed"]),
        "eval_all": evaluate(model, evaluation),
    }

    bundle = model.save()
    METRICS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nsaved {bundle}\nsaved {METRICS_PATH}\n")
    for name, scores in report.items():
        if isinstance(scores, dict):
            print(f"{name:16s} intent {scores['weather_intent_accuracy']:.1%}  "
                  f"action {scores['action_accuracy']:.1%}  "
                  f"loc F1 {scores['location_span']['f1']:.3f}  "
                  f"time F1 {scores['time_span']['f1']:.3f}  "
                  f"all-4 {scores['all_four_targets']:.1%}")


if __name__ == "__main__":
    # Re-enter through the package so pickled objects record src.nlu.NLUModel rather than
    # __main__.NLUModel - otherwise the bundle only loads back in this exact script.
    from src.nlu import main as packaged_main

    packaged_main()
