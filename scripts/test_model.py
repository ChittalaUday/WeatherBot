"""The trained classifier's raw output for one sentence, or a REPL, or the eval suite.

Usage:
    python scripts/test_model.py                                     # interactive console
    python scripts/test_model.py "Will it rain in Guntur tomorrow?"  # one query, pure JSON
    python scripts/test_model.py --eval                              # eval suite
    python scripts/test_model.py --demo                              # self-check

One bundle is served, so there is no version flag and no compare mode here. Side-by-side
against the prompted classifiers lives at POST /api/compare, which runs the whole pipeline
per column.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"

# If invoked with system Python, automatically switch to the project's .venv Python
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    os.execv(str(VENV_PYTHON), [str(VENV_PYTHON)] + sys.argv)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.nlu.registry import DEFAULT_VERSION, MODELS, Registry
from src.v4.model import V4Model

_MODEL = None


def predict_pure_json(text: str) -> str:
    """The model's own output object, serialized. No Understanding, no pipeline."""
    global _MODEL
    if _MODEL is None:
        _MODEL = V4Model.load()
    result = _MODEL.predict(text)
    if hasattr(result, "model_dump_json"):
        return result.model_dump_json(indent=2)
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(), indent=2)
    return json.dumps(result, indent=2, default=str)


def run_eval(registry: Registry):
    """Run evaluation benchmark against data/eval_v4.csv."""
    eval_csv = ROOT / "data" / "eval_v4.csv"
    if not eval_csv.exists():
        print(f"Error: {eval_csv} not found")
        return

    rows = list(csv.DictReader(eval_csv.open()))
    print(f"\n--- Running evaluation benchmark on {len(rows)} hand-written rows ---\n")
    hits, seen = Counter(), Counter()

    for r in rows:
        u = registry.understand(r["text"])
        gold_loc, gold_time = json.loads(r["locations"]), json.loads(r["times"])
        norm = lambda spans: sorted(s.lower() for s in spans)

        checks = {
            "intent": u.intent == r["intent"],
            "variables": set(u.variables) == {v for v in r["variables"].split("|") if v},
            "locations": norm(u.locations) == norm(gold_loc),
            "times": norm(u.times) == norm(gold_time),
        }
        if r["intent"] == "ADVICE":
            checks["activity"] = u.activity == r["activity"]

        for k, ok in checks.items():
            hits[k] += int(ok)
            seen[k] += 1
        every = all(checks.values())
        hits["everything"] += int(every)
        seen["everything"] += 1

    for k in ("intent", "variables", "activity", "locations", "times", "everything"):
        if seen[k]:
            pct = hits[k] / seen[k]
            print(f"  {k:15s} : {pct:6.1%} ({hits[k]}/{seen[k]})")
    print()


def interactive_loop():
    """Interactive terminal session, pure JSON per line."""
    name = MODELS[DEFAULT_VERSION]["name"]
    print("=" * 65)
    print("WeatherBot NLU Terminal Console (Pure Model Output)")
    print(f"Model: {DEFAULT_VERSION} ({name})   'exit' to quit")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input("> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break
        print(predict_pure_json(query))
        print()


def demo():
    """Self-check assertion for pure model test script (Rule 10)."""
    said = predict_pure_json("Will it rain in Guntur tomorrow?")
    assert '"intent"' in said, "Missing intent in prediction"
    assert '"RAIN"' in said, "Missing RAIN in prediction"
    assert '"Guntur"' in said, "Missing Guntur in prediction"
    print("OK: test_model pure prediction demo checks passed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Test the WeatherBot NLU model, raw output")
    parser.add_argument("query", nargs="*", help="Query string to process (optional)")
    parser.add_argument("-e", "--eval", action="store_true", help="Run evaluation benchmark suite")
    parser.add_argument("--demo", action="store_true", help="Run self-check assertions")
    args = parser.parse_args()

    if args.demo:
        demo()
    elif args.eval:
        run_eval(Registry())
    elif args.query:
        print(predict_pure_json(" ".join(args.query)))
    else:
        interactive_loop()


if __name__ == "__main__":
    main()
