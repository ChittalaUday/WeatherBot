"""
Aggregations, charts and insights - everything that turns rows into an answer.

The model decides *which* reduction the question asks for (Rule 2.3: RAW / SUM / AVG / MAX /
MIN / TREND). Applying it, picking a chart and reading the numbers is deterministic: the
model never invents a value, it only says what to compute.
"""

from __future__ import annotations

from datetime import datetime

from backend.respond import LABELS, UNITS, _format, _parse

# Thresholds worth mentioning unprompted, in the API's own units.
NOTABLE = {
    "Rainfall": (10.0, "heavy rain", "mm"),
    "Tmax": (40.0, "heat above 40°C", "°C"),
    "Tmin": (10.0, "cold below 10°C", "°C"),
    "Wind_Speed": (11.0, "strong wind above ~40 kmph", "m/s"),
    "Wind_max": (14.0, "gusts above ~50 kmph", "m/s"),
    "RH": (90.0, "very humid air", "%"),
}
# Fields that add up over a range; everything else averages.
ADDITIVE = {"Rainfall", "SunSD"}


# A reduction is always spoken out loud - "total", "average", "peak". If none of these words
# is in the prompt, a non-RAW prediction is the classifier over-reaching on a short query.
AGG_WORDS = {
    "SUM": ("total", "cumulative", "sum", "add up", "overall", "how much", "altogether"),
    "AVG": ("average", "avg", "mean", "typical", "on average"),
    "MAX": ("peak", "highest", "max", "hottest", "strongest", "worst", "how high", "how hot",
            "warmest", "most"),
    "MIN": ("lowest", "minimum", "min", "coldest", "weakest", "how low", "least"),
    "TREND": ("start", "starts", "rise", "rising", "drop", "dropping", "fall", "falling",
              "change", "changing", "increase", "increasing", "decrease", "stop", "pick up",
              "when will", "what time"),
}


def confirm_aggregation(text: str, aggregation: str) -> str:
    """Keep the model's reduction only when the wording supports it."""
    if aggregation == "RAW":
        return aggregation
    lowered = text.lower()
    return aggregation if any(word in lowered for word in AGG_WORDS.get(aggregation, ())) else "RAW"


def _values(rows: list[dict], field: str) -> list[float]:
    return [float(r[field]) for r in rows if r.get(field) is not None]


def apply_aggregation(rows: list[dict], field: str, aggregation: str) -> dict | None:
    """The single number the question asked for, or None when it wants the rows themselves."""
    values = _values(rows, field)
    if not values or aggregation == "RAW":
        return None
    unit = UNITS.get(field, "")
    label = LABELS.get(field, field)

    if aggregation == "SUM":
        return {"kind": "SUM", "value": round(sum(values), 2), "unit": unit,
                "text": f"Total {label.lower()}: {sum(values):.1f}{unit} across {len(values)} readings"}
    if aggregation == "AVG":
        mean = sum(values) / len(values)
        return {"kind": "AVG", "value": round(mean, 2), "unit": unit,
                "text": f"Average {label.lower()}: {mean:.1f}{unit}"}
    if aggregation in {"MAX", "MIN"}:
        pick = max if aggregation == "MAX" else min
        best = pick(rows, key=lambda r: float(r[field]) if r.get(field) is not None else
                    (float("-inf") if aggregation == "MAX" else float("inf")))
        when = _parse(best).strftime("%d %b %H:%M" if _parse(best).hour else "%d %b")
        word = "Highest" if aggregation == "MAX" else "Lowest"
        return {"kind": aggregation, "value": round(float(best[field]), 2), "unit": unit,
                "at": best["Date_time"],
                "text": f"{word} {label.lower()}: {float(best[field]):.1f}{unit} at {when}"}
    if aggregation == "TREND":
        return _trend(rows, field)
    return None


def _trend(rows: list[dict], field: str) -> dict | None:
    """Where the series turns, in plain words - "starts dropping after 14:00"."""
    points = [(r["Date_time"], float(r[field])) for r in rows if r.get(field) is not None]
    if len(points) < 3:
        return None
    label = LABELS.get(field, field).lower()
    unit = UNITS.get(field, "")
    peak_at, peak = max(points, key=lambda p: p[1])
    low_at, low = min(points, key=lambda p: p[1])
    first, last = points[0][1], points[-1][1]
    direction = "rising" if last > first else "falling" if last < first else "flat"

    stamp = lambda iso: datetime.fromisoformat(iso).strftime("%d %b %H:%M"
                                                             if datetime.fromisoformat(iso).hour else "%d %b")
    turn = ""
    if points.index((peak_at, peak)) < len(points) - 1 and direction != "rising":
        turn = f", starts dropping after {stamp(peak_at)}"
    elif points.index((low_at, low)) < len(points) - 1 and direction == "rising":
        turn = f", starts climbing after {stamp(low_at)}"
    return {"kind": "TREND", "value": round(last - first, 2), "unit": unit,
            "text": f"{label.capitalize()} is {direction} ({first:.1f} to {last:.1f}{unit}){turn}",
            "peak_at": peak_at, "low_at": low_at}


def build_chart(selected: list[list[dict]], places: list[dict], field: str, hourly: bool,
                kind: str | None = None, fields: list[str] | None = None) -> dict | None:
    """Line for a series over time, grouped bars for a short comparison, nothing for one point.

    `kind` is v3's decision and wins when given: the model read the question, whereas this
    function only ever saw row counts. MULTI_LINE plots several variables for one place,
    which the row-count heuristic could not express at all.
    """
    if kind == "NONE":
        return None
    if kind == "MULTI_LINE" and fields and len(places) == 1 and selected:
        multi = [{"name": LABELS.get(name, name),
                  "points": [{"t": row["Date_time"], "v": float(row[name])}
                             for row in selected[0] if row.get(name) is not None]}
                 for name in fields[:3]]
        multi = [entry for entry in multi if len(entry["points"]) > 1]
        if multi:
            return {"type": "line", "field": field, "label": "Readings",
                    "unit": UNITS.get(field, ""),
                    "granularity": "hourly" if hourly else "daily", "series": multi}

    series = []
    for place, rows in zip(places, selected):
        points = [{"t": r["Date_time"], "v": float(r[field])} for r in rows if r.get(field) is not None]
        if points:
            series.append({"name": place["name"], "points": points})
    if not series or all(len(s["points"]) < 2 for s in series):
        return None
    if kind in {"GROUPED_BAR", "STAT"}:
        shape = "bar"
    elif kind in {"LINE", "MULTI_LINE"}:
        shape = "line"
    else:
        shape = "bar" if len(series) > 1 and max(len(s["points"]) for s in series) <= 8 else "line"
    return {
        "type": shape,
        "field": field,
        "label": LABELS.get(field, field),
        "unit": UNITS.get(field, ""),
        "granularity": "hourly" if hourly else "daily",
        "series": series,
    }


def build_insights(selected: list[list[dict]], places: list[dict], fields: list[str],
                   aggregation: str, hourly: bool, wanted: list[str] | None = None) -> list[str]:
    """Two to four things worth saying that the table does not say by itself.

    `wanted` is v3's selection. Without it every applicable observation is emitted, which is
    the old behaviour before the model chose for itself.
    """
    allow = set(wanted) if wanted else None
    keep = lambda kind: allow is None or kind in allow
    field = fields[0]
    label = LABELS.get(field, field).lower()
    unit = UNITS.get(field, "")
    out: list[str] = []

    for place, rows in zip(places, selected):
        values = _values(rows, field)
        if not values:
            continue
        where = place["name"] if len(places) > 1 else ""
        prefix = f"{where}: " if where else ""

        if len(values) > 1 and aggregation == "RAW" and keep("RANGE"):
            total = sum(values) if field in ADDITIVE else sum(values) / len(values)
            word = "total" if field in ADDITIVE else "average"
            out.append(f"{prefix}{word} {label} {total:.1f}{unit}, "
                       f"range {min(values):.1f}-{max(values):.1f}{unit}")

        if keep("THRESHOLD") and (threshold := NOTABLE.get(field)) and \
                (crossings := [v for v in values if v >= threshold[0]]):
            unit_word = "readings" if hourly else "days"
            out.append(f"{prefix}{threshold[1]} on {len(crossings)} of {len(values)} {unit_word} "
                       f"(peak {max(crossings):.1f}{threshold[2]})")

        if keep("DRY_SPELL") and field == "Rainfall" and len(values) > 2:
            dry = sum(1 for v in values if v < 1.0)
            if dry:
                out.append(f"{prefix}{dry} of {len(values)} dry {'hours' if hourly else 'days'} (<1mm)")

    # comparisons only make sense once both sides are in hand
    if len(places) > 1 and keep("COMPARISON"):
        totals = []
        for place, rows in zip(places, selected):
            values = _values(rows, field)
            if values:
                totals.append((place["name"],
                               sum(values) if field in ADDITIVE else sum(values) / len(values)))
        if len(totals) > 1:
            totals.sort(key=lambda t: -t[1])
            gap = totals[0][1] - totals[-1][1]
            out.append(f"{totals[0][0]} leads {totals[-1][0]} by {gap:.1f}{unit} on {label}")

    return out[:4]
