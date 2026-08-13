"""
v3 dataset - the v2 turns, relabelled with the three presentation decisions.

    python -m src.v3.dataset --build     # -> data/v3_dataset.csv
    python -m src.v3.dataset --stats
    python -m src.v3.dataset --samples 8

Where the labels come from, stated plainly: the *detail* label is read off explicit wording
("in detail", "just the number", "everything"), and where the wording says nothing, a teacher
rule assigns the choice a careful engineer would make. So v3 begins as a distillation of the
rules it replaces. That is still worth doing:

  - it decides from the phrasing rather than from a lookup keyed on intent, so "temperature in
    detail" and "full temperature breakdown" both reach FULL without either being enumerated
  - the decision becomes a measurable target, so a wrong chart is a labelled example instead
    of an argument about an if-statement
  - user corrections land in the same columns, which is the only way it ever beats the teacher

Detail-bearing phrasings are also generated fresh, because no v1/v2 template ever said
"in detail" - the models could not have learned it from the existing data.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v2 import dataset as v2_dataset
from src.v2.schema import Variable
from src.v3.schema import ChartKind, Detail, Insight

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "data" / "v3_dataset.csv"
FIELDS = v2_dataset.FIELDS + ["detail", "chart", "insights"]

# Wording that names the detail level outright.
FULL_CUES = ("in detail", "detailed", "full", "everything", "breakdown", "all the",
             "complete", "in depth", "expand", "elaborate", "more detail")
MINIMAL_CUES = ("just the", "only the", "one number", "quick", "briefly", "in short",
                "single", "summary", "tl;dr", "short answer")

# Phrasings that carry a detail cue. No v1/v2 template produced any of these.
DETAIL_FRAMES = {
    "FULL": [
        "{a} in {loc} {t} in detail",
        "give me the full {a} breakdown for {loc} {t}",
        "detailed {a} for {loc} {t}",
        "everything about the {a} in {loc} {t}",
        "{a} in {loc} {t} - all the numbers",
        "complete {a} report for {loc} {t}",
        "i want the {a} in depth for {loc} {t}",
        "expand on the {a} in {loc} {t}",
        "full {a} for {loc} {t}",
        "{a} breakdown {loc} {t}",
    ],
    "MINIMAL": [
        "just the {a} in {loc} {t}",
        "only the {a} for {loc} {t}",
        "quick {a} for {loc} {t}",
        "{a} in {loc} {t} in short",
        "one number: {a} in {loc} {t}",
        "brief {a} for {loc} {t}",
        "short answer, {a} in {loc} {t}?",
        "{a} {loc} {t} - summary only",
    ],
}


def detail_from_text(text: str) -> Detail:
    """Explicit wording wins; everything else is NORMAL."""
    lowered = text.lower()
    if any(cue in lowered for cue in FULL_CUES):
        return Detail.FULL
    if any(cue in lowered for cue in MINIMAL_CUES):
        return Detail.MINIMAL
    return Detail.NORMAL


def teach_chart(variables, locations, times, detail: Detail, aggregation: str) -> ChartKind:
    """The chart a careful engineer would pick, given the shape of the question.

    This is the teacher signal, and it is deliberately about the *question*, not the row
    count: how many places, how many variables, how wide the time range, and whether the
    user asked for a single number.
    """
    if detail is Detail.MINIMAL or aggregation in {"SUM", "AVG", "MAX", "MIN"}:
        return ChartKind.STAT if aggregation != "RAW" else ChartKind.NONE
    if len(locations) > 1 or len(times) > 1:
        return ChartKind.GROUPED_BAR
    if len(variables) > 1:
        return ChartKind.MULTI_LINE
    if not times or any(word in " ".join(times).lower()
                        for word in ("week", "days", "month", "weekend")):
        return ChartKind.LINE                 # a range worth seeing as a curve
    return ChartKind.NONE                     # a single day: the table says it better


def teach_insights(variables, locations, times, detail: Detail, aggregation: str) -> list[Insight]:
    """Which observations earn their place. Fewer for a one-number question, more for a
    comparison across a week."""
    picked: list[Insight] = []
    wide = (not times) or any(word in " ".join(times).lower()
                              for word in ("week", "days", "month", "weekend"))

    if aggregation == "SUM":
        picked.append(Insight.TOTAL)
    elif aggregation == "AVG":
        picked.append(Insight.AVERAGE)
    elif aggregation == "MAX":
        picked.append(Insight.PEAK)
    elif aggregation == "MIN":
        picked.append(Insight.LOW)
    elif aggregation == "TREND":
        picked.append(Insight.TREND)

    if detail is Detail.MINIMAL:
        return picked or [Insight.RANGE]

    if wide:
        picked.append(Insight.RANGE)
        if Variable.RAIN in variables or "RAIN" in [str(v) for v in variables]:
            picked += [Insight.TOTAL, Insight.DRY_SPELL]
        if detail is Detail.FULL:
            picked += [Insight.PEAK, Insight.LOW]
    if len(locations) > 1 or len(times) > 1:
        picked.append(Insight.COMPARISON)
    if detail is Detail.FULL:
        picked.append(Insight.THRESHOLD)

    seen, ordered = set(), []
    for insight in picked:
        if insight.value not in seen:
            seen.add(insight.value)
            ordered.append(insight)
    return ordered or [Insight.RANGE]


def _label(row: dict) -> dict:
    """Attach the three presentation columns to a v2 row."""
    variables = [Variable(name) for name in row["variables"]]
    detail = detail_from_text(row["text"])
    aggregation = row.get("aggregation", "RAW")
    chart = teach_chart(variables, row["locations"], row["times"], detail, aggregation)
    insights = teach_insights(variables, row["locations"], row["times"], detail, aggregation)
    return {**row,
            "variables": "|".join(v.value for v in variables),
            "locations": json.dumps(row["locations"]), "times": json.dumps(row["times"]),
            "ctx_locations": json.dumps(row["ctx_locations"]),
            "ctx_times": json.dumps(row["ctx_times"]),
            "detail": detail.value, "chart": chart.value,
            "insights": "|".join(i.value for i in insights)}


def generate_detail_rows(rng: random.Random, count: int, split: str) -> list[dict]:
    """Prompts that say how much detail they want - a phrasing v1/v2 never produced."""
    rows, seen = [], set()
    variables = [v for v in Variable if v != Variable.GENERAL]
    guard = 0
    while len(rows) < count and guard < count * 30:
        guard += 1
        level = rng.choice(["FULL", "MINIMAL"])
        variable = rng.choice(variables)
        place = rng.choice(v2_dataset.LOCATIONS)
        time_span = rng.choice(v2_dataset.TIMES)
        text = v2_dataset._clean(rng.choice(DETAIL_FRAMES[level]).format(
            a=rng.choice(v2_dataset.VARIABLE_WORDS[variable]), loc=place, t=time_span))
        if text.lower() in seen or not all(span in text for span in [place] + ([time_span] if time_span else [])):
            continue
        seen.add(text.lower())
        times = [time_span] if time_span else []
        base = v2_dataset._row(f"d3-{split[:2]}-{len(rows):05d}", 0, text,
                               v2_dataset._intent_for(time_span, "GET"), [variable],
                               [place], times, "SET", [place], times, split, "detail")
        base.update({"variables": [variable.value], "locations": [place], "times": times,
                     "ctx_locations": [place], "ctx_times": times})
        rows.append(_label(base))
    return rows


def build(path: Path = CSV_PATH, detail_rows: int = 1800, seed: int = 13) -> dict:
    """Relabel every v2 row, then add the detail-bearing phrasings."""
    rng = random.Random(seed)
    v2_rows = v2_dataset.load()
    if not v2_rows:
        raise SystemExit("build the v2 dataset first: python -m src.v2.dataset --build")

    rows = [_label(row) for row in v2_rows]
    counts = {f"relabelled:{split}": sum(r["split"] == split for r in rows)
              for split in ("train", "test", "eval")}

    generated = generate_detail_rows(rng, detail_rows, "train")
    cut = int(len(generated) * 0.85)
    for row in generated[cut:]:
        row["split"] = "test"
    rows += generated
    counts["detail:train"], counts["detail:test"] = cut, len(generated) - cut

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts["rows"] = len(rows)
    return counts


def load(path: Path | str = CSV_PATH, split: str | None = None, lang: str | None = None,
         source: str | None = None) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if (split and record["split"] != split) or (lang and record["lang"] != lang):
                continue
            if source and record["source"] != source:
                continue
            rows.append({
                **record,
                "turn": int(record["turn"]),
                "variables": [v for v in record["variables"].split("|") if v],
                "insights": [i for i in record["insights"].split("|") if i],
                "locations": json.loads(record["locations"]),
                "times": json.loads(record["times"]),
                "ctx_locations": json.loads(record["ctx_locations"]),
                "ctx_times": json.loads(record["ctx_times"]),
            })
    return rows


def stats(rows: list[dict]) -> dict:
    return {
        "rows": len(rows),
        "splits": dict(Counter(r["split"] for r in rows)),
        "detail": dict(Counter(r["detail"] for r in rows)),
        "chart": dict(Counter(r["chart"] for r in rows)),
        "insights": dict(Counter(i for r in rows for i in r["insights"]).most_common()),
        "mean_insights": round(sum(len(r["insights"]) for r in rows) / max(len(rows), 1), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(CSV_PATH))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--detail-rows", type=int, default=1800)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--samples", type=int, metavar="N")
    args = parser.parse_args()

    if args.build:
        for key, value in build(Path(args.path), args.detail_rows).items():
            print(f"  {key:20s} {value}")
    rows = load(args.path)
    if args.stats or args.build:
        for key, value in stats(rows).items():
            print(f"  {key:20s} {value}")
    if args.samples:
        for row in random.Random(5).sample(rows, min(args.samples, len(rows))):
            print(f"\n  {row['text'][:72]}")
            print(f"    detail={row['detail']:8s} chart={row['chart']:12s} "
                  f"insights={row['insights']}")


if __name__ == "__main__":
    main()
