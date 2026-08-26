"""Simple CLI and interactive terminal script to test WeatherBot models (v4 / v3) with pure model output.

Usage:
    python scripts/test_model.py                                          # Interactive terminal session
    python scripts/test_model.py "Will it rain in Guntur tomorrow?"      # Direct CLI query (pure JSON)
    python scripts/test_model.py -v v3 "Will it rain in Guntur tomorrow?" # Force model version (v3)
    python scripts/test_model.py --compare "Will it rain in Guntur?"      # Compare pure outputs side-by-side
    python scripts/test_model.py --eval                                   # Run eval suite
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

from backend.nlu.registry import MODELS, Registry, Understanding
from src.v3.model import V3Model
from src.v4.model import V4Model



class ModelRunner:
    """Lazy loader and wrapper for pure model prediction outputs."""

    def __init__(self):
        self._models = {}

    def get_model(self, version: str):
        version = version.lower()
        if version not in {"v3", "v4"}:
            version = "v4"
        if version not in self._models:
            if version == "v4":
                self._models["v4"] = V4Model.load()
            else:
                self._models["v3"] = V3Model.load()
        return self._models[version]

    def predict_pure_json(self, text: str, version: str = "v4") -> str:
        """Return the pure model output object serialized as indented JSON string."""
        model = self.get_model(version)
        result = model.predict(text)
        if hasattr(result, "model_dump_json"):
            return result.model_dump_json(indent=2)
        elif hasattr(result, "model_dump"):
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
        u = registry.understand(r["text"], version="v4")
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


def interactive_loop(runner: ModelRunner, initial_version: str = "v4"):
    """Interactive terminal session to test models with pure JSON output."""
    current_version = initial_version.lower()
    print("=" * 65)
    print("WeatherBot NLU Terminal Console (Pure Model Output)")
    print(f"Active Model: {current_version} ({MODELS.get(current_version, {}).get('name', '')})")
    print("Commands:")
    print("  'v3' or 'v4'        : Switch active model version")
    print("  'compare <query>'   : Run query against both v4 and v3")
    print("  'exit' or 'quit'    : Exit console")
    print("=" * 65 + "\n")

    while True:
        try:
            query = input(f"[{current_version}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not query:
            continue

        cmd = query.lower()
        if cmd in {"exit", "quit", "q"}:
            break

        if cmd in {"v3", "v4"}:
            current_version = cmd
            name = MODELS.get(current_version, {}).get("name", current_version)
            print(f"--> Switched active model to {current_version} ({name})\n")
            continue

        if cmd.startswith("compare "):
            text = query[8:].strip()
            print("\n=== Model 2 (v4) Output ===")
            print(runner.predict_pure_json(text, "v4"))
            print("\n=== Model 1 (v3) Output ===")
            print(runner.predict_pure_json(text, "v3"))
            print()
            continue

        # Pure model output JSON
        print(runner.predict_pure_json(query, current_version))
        print()


def demo():
    """Self-check assertion for pure model test script (Rule 10)."""
    runner = ModelRunner()

    v4_json = runner.predict_pure_json("Will it rain in Guntur tomorrow?", "v4")
    assert '"intent"' in v4_json, "Missing intent in v4 prediction"
    assert '"RAIN"' in v4_json, "Missing RAIN in v4 prediction"
    assert '"Guntur"' in v4_json, "Missing Guntur in v4 prediction"

    v3_json = runner.predict_pure_json("Will it rain in Guntur tomorrow?", "v3")
    assert '"intent"' in v3_json, "Missing intent in v3 prediction"

    print("OK: test_model pure prediction demo checks passed cleanly.")


def main():
    parser = argparse.ArgumentParser(description="Test WeatherBot NLU models with pure model output")
    parser.add_argument("query", nargs="*", help="Query string to process (optional)")
    parser.add_argument("-v", "--version", choices=["v3", "v4"], default="v4", help="Model version (default: v4)")
    parser.add_argument("-c", "--compare", action="store_true", help="Compare v4 and v3 predictions for given query")
    parser.add_argument("-e", "--eval", action="store_true", help="Run evaluation benchmark suite")
    parser.add_argument("--demo", action="store_true", help="Run self-check assertions")
    args = parser.parse_args()

    if args.demo:
        demo()
        return

    if args.eval:
        registry = Registry()
        run_eval(registry)
        return

    runner = ModelRunner()

    if args.query:
        query_text = " ".join(args.query)
        if args.compare:
            print("=== Model 2 (v4) Output ===")
            print(runner.predict_pure_json(query_text, "v4"))
            print("\n=== Model 1 (v3) Output ===")
            print(runner.predict_pure_json(query_text, "v3"))
        else:
            print(runner.predict_pure_json(query_text, args.version))
        return

    # Interactive REPL mode
    interactive_loop(runner, args.version)


if __name__ == "__main__":
    main()

