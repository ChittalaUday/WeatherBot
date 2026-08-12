"""Self-check for the NLU datasets. Run: python test_dataset.py"""

import csv
import difflib
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

from src.build_dataset import (ACTIONS, EVAL_MANUAL, INDIA_NAMES, INTENTS, LOCATIONS_CSV,
                               SPLITS, india_share, load_seed, validate_row)

ROOT = Path(__file__).parent
MIN_EVAL_ROWS = 200


def check_rows(name, rows):
    """Rules every split obeys, generated or hand-written."""
    assert rows, f"{name} split missing - run: python src/build_dataset.py --split {name}"

    for row in rows:
        err = validate_row(row)
        assert not err, f"[{name}] {err} -> {row}"

    texts = [r["text"].strip().lower() for r in rows]
    assert len(set(texts)) == len(texts), f"[{name}] duplicate prompts"

    assert any(not r["location"] for r in rows), f"[{name}] no examples with empty location"
    assert any(not r["time"] for r in rows), f"[{name}] no examples with empty time"


def check_generated(name, rows):
    check_rows(name, rows)
    cells = Counter((r["weather_intent"], r["action"]) for r in rows)
    assert len(cells) == len(INTENTS) * len(ACTIONS), f"[{name}] missing cells: {len(cells)}"
    assert min(cells.values()) >= 0.8 * max(cells.values()), f"[{name}] unbalanced: {cells.most_common()}"
    print(f"OK {name:5s}: {len(rows)} rows, {len(cells)} balanced cells "
          f"({min(cells.values())}-{max(cells.values())} each)")


def check_manual(rows):
    """The evaluation set is hand-written (data/eval_manual.csv), so it is deliberately not
    balanced per cell - it is judged on coverage and on carrying the edge cases no template
    produces. Never regenerate this file from a script."""
    check_rows("eval", rows)
    assert len(rows) >= MIN_EVAL_ROWS, f"eval has {len(rows)} rows, want >= {MIN_EVAL_ROWS}"

    intents = {r["weather_intent"] for r in rows}
    actions = {r["action"] for r in rows}
    assert intents == set(INTENTS), f"eval misses intents: {sorted(set(INTENTS) - intents)}"
    assert actions == set(ACTIONS), f"eval misses actions: {sorted(set(ACTIONS) - actions)}"

    # English is the target language; code-mixed rows ride along as a diagnostic only.
    langs = Counter(r.get("lang", "") for r in rows)
    assert set(langs) == {"en", "mixed"}, f"eval lang column is {sorted(langs)}, want en/mixed"
    assert langs["en"] >= 0.8 * len(rows), f"only {langs['en']}/{len(rows)} English eval rows"
    assert langs["mixed"] >= 10, f"only {langs['mixed']} code-mixed rows to diagnose with"

    edge = {
        "multi-location": sum(len(r["location"]) > 1 for r in rows),
        "address form": sum(any("," in s for s in r["location"]) for r in rows),
        "two time spans": sum(len(r["time"]) > 1 for r in rows),
        "clock time": sum(any(re.search(r"\d[:\d]* ?(am|pm)", s, re.I) for s in r["time"]) for r in rows),
        "no entities": sum(not r["location"] and not r["time"] for r in rows),
    }
    for label, count in edge.items():
        assert count >= 5, f"eval has only {count} '{label}' rows"
    print(f"OK eval : {len(rows)} hand-written rows, {len(intents)} intents, edge cases {edge}")


def spans(rows):
    return {s.lower() for r in rows for s in r["location"] + r["time"]}


def main():
    rows = {name: load_seed(ROOT / path) for name, (_, _, path, _) in SPLITS.items()}
    for name in SPLITS:
        check_generated(name, rows[name])

    rows["eval"] = load_seed(EVAL_MANUAL)
    check_manual(rows["eval"])

    for a, b in combinations(rows, 2):
        shared = {r["text"].strip().lower() for r in rows[a]} & {r["text"].strip().lower() for r in rows[b]}
        assert not shared, f"{a}/{b} share {len(shared)} prompts: {sorted(shared)[:3]}"

    # The eval set only measures generalization while it keeps entity spans train never saw.
    unseen = spans(rows["eval"]) - (spans(rows["train"]) | spans(rows["test"]))
    assert len(unseen) >= 40, f"eval barely holds anything out: {len(unseen)} unseen spans"
    print(f"OK: {len(rows)} splits disjoint by prompt, "
          f"{len(unseen)}/{len(spans(rows['eval']))} eval entity spans unseen in train/test")

    if not INDIA_NAMES:  # built-in fallback vocabulary, nothing geographic to assert
        print("SKIP: data/locations.csv absent - run python src/fetch_locations.py")
        return

    for name in SPLITS:  # generated splits draw from the sampled vocabulary
        share = india_share(rows[name])
        assert share >= 0.8, f"[{name}] only {share:.1%} of location spans inside India"
        print(f"OK {name:5s}: {share:.1%} of location spans inside India")

    vocab = list(csv.DictReader(LOCATIONS_CSV.open()))
    levels = Counter(row["level"] for row in vocab)
    inside = sum(row["in_india"] == "1" for row in vocab)
    assert len(vocab) >= 1000, f"location vocabulary is only {len(vocab)} names"
    assert levels.most_common(1)[0][0] == "village", f"vocabulary is not village-led: {levels}"
    assert inside / len(vocab) >= 0.8, f"only {inside / len(vocab):.1%} of the vocabulary is in India"
    print(f"OK: {len(vocab)} DB locations ({inside / len(vocab):.1%} in India), {dict(levels)}")

    # Addresses ("Angara, East Godavari"), not only bare names.
    addressed = {s for r in rows["train"] for s in r["location"] if "," in s}
    assert len(addressed) >= 30, f"only {len(addressed)} qualified addresses in train"

    # Spelling noise: near-miss variants of common words must survive into the prompts,
    # otherwise the model only ever sees perfectly typed queries.
    common = ["compare", "notify", "warning", "alert", "temperature", "weather", "forecast",
              "moisture", "between", "advisory", "humidity", "sunshine"]
    tokens = Counter(t for r in rows["train"] for t in re.findall(r"[a-z]{5,}", r["text"].lower()))
    misspelt = {t for t in tokens
                if t not in common and difflib.get_close_matches(t, common, n=1, cutoff=0.85)}
    assert len(misspelt) >= 10, f"barely any spelling mistakes in train: {sorted(misspelt)}"

    # COMPARE is the most template-like action, so it must be the noisiest, not the cleanest.
    compare = [r for r in rows["train"] if r["action"] == "COMPARE"]
    compare_misspelt = sum(any(t in misspelt for t in re.findall(r"[a-z]{5,}", r["text"].lower()))
                           for r in compare)
    assert compare_misspelt >= 0.05 * len(compare), f"COMPARE barely misspelt: {compare_misspelt}"

    # Ungrammatical typing: chat fillers and prompts that dropped their question mark.
    fillers = sum(bool(re.search(r"\b(pls|plz|kindly|asap|sir|bro|urgent|quickly)\b", r["text"], re.I))
                  for r in rows["train"])
    assert fillers >= 100, f"only {fillers} prompts with chat fillers"
    print(f"OK: {len(misspelt)} misspelt word variants (e.g. {sorted(misspelt)[:4]}), "
          f"{compare_misspelt} noisy COMPARE rows, {fillers} with fillers")

    # Clock times must carry real numbers: round hours dominate, odd minutes still show up.
    minutes = Counter(m.group(1) for r in rows["train"] for s in r["time"]
                      if (m := re.search(r":(\d\d)", s)))
    assert len(minutes) >= 5, f"clock minutes barely vary: {minutes}"
    assert set(minutes) - {"00", "30"}, "no odd-minute clock times"
    print(f"OK: {len(addressed)} address forms in train, clock minutes {sorted(minutes)}")


if __name__ == "__main__":
    main()
