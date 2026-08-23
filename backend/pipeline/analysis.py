"""
Rows -> the reduction, the chart and the observations worth saying out loud.

    python -m backend.pipeline.analysis        # self-check

The model decides *which* reduction the question asks for (Rule 2.3: RAW / SUM / AVG / MAX /
MIN / TREND) and whether a chart was wanted. Computing any of it is deterministic: the model
never invents a value, it only says what to compute.

Observations carry their kind, not just their wording. A flat list of sentences was fine for
a bullet list and useless to everything else - the generation layer could not tell a range
from a threshold crossing from a comparison, so it re-said them in whatever order they
happened to be built. `Note.kind` is what lets the prompt group them.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.pipeline.quality import is_missing, values
from backend.pipeline.render import label, stamp, summary_stat, unit

# Thresholds worth mentioning unprompted, in the API's own units.
NOTABLE = {
    "Rainfall": (10.0, "heavy rain", "mm"),
    "Tmax": (40.0, "heat above 40°C", "°C"),
    "Tmin": (10.0, "cold below 10°C", "°C"),
    "Wind_Speed": (11.0, "strong wind above ~40 kmph", "m/s"),
    "Wind_max": (14.0, "gusts above ~50 kmph", "m/s"),
    "RH": (90.0, "very humid air", "%"),
}
# Fields whose notable event is the *low* end of the threshold. Without this every reading
# above 10°C counted as "cold below 10°C" - a 21°C night included.
BELOW = {"Tmin"}

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
# Words that ask to *see* it rather than be told it. Same idea: a chart is offered, not assumed.
CHART_WORDS = ("chart", "graph", "plot", "curve", "trend", "over time", "hour by hour",
               "visuali", "show me")

MAX_NOTES = 4          # more than this under one answer is a wall, not a summary


@dataclass(frozen=True)
class Note:
    """One observation, with the kind of observation it is."""

    kind: str          # RANGE | THRESHOLD | DRY_SPELL | COMPARISON
    text: str
    place: str = ""

    def as_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "place": self.place}


def wants_chart(text: str, kind: str = "") -> bool:
    """Did this question ask for a picture?

    v3's presentation head read the question and picked a chart kind, so it wins outright. For
    everything else the wording decides: a decision ("should I take a raincoat") and a single
    reduced figure are answers in one line, and a temperature curve under them is decoration.
    """
    if kind and kind != "NONE":
        return True
    return any(word in (text or "").lower() for word in CHART_WORDS)


# The subset of AGG_WORDS that can only mean a reduction. "how much" and "overall" are in the
# table above because they *support* a prediction; they are too loose to *make* one - "how much
# rain tomorrow" is a question about tomorrow, not a request for a sum.
EXPLICIT = {
    "SUM": ("total", "cumulative", "sum", "add up", "altogether"),
    "AVG": ("average", "avg", "mean", "on average"),
}


def confirm_aggregation(text: str, aggregation: str) -> str:
    """The reduction this question actually asked for.

    A reduction is always spoken out loud - "total", "average", "peak" - so the wording is the
    arbiter in both directions:

      - the model said SUM and no word supports it -> the classifier over-reached, drop to RAW
      - the model said RAW and the wording plainly says "total" -> it under-reached, promote

    Only the unambiguous words promote. Dropping a reduction costs a less specific answer;
    inventing one changes what the number means, so the bar for adding is higher.
    """
    lowered = (text or "").lower()
    if aggregation == "RAW":
        for kind, words in EXPLICIT.items():
            if any(word in lowered for word in words):
                return kind
        return "RAW"
    return aggregation if any(w in lowered for w in AGG_WORDS.get(aggregation, ())) else "RAW"


def apply_aggregation(rows: list[dict], field: str, aggregation: str) -> dict | None:
    """The single number the question asked for, or None when it wants the rows themselves."""
    numbers = values(rows, field)
    if not numbers or aggregation == "RAW":
        return None
    units, name = unit(field), label(field).lower()

    if aggregation == "SUM":
        total = sum(numbers)
        return {"kind": "SUM", "value": round(total, 2), "unit": units,
                "text": f"Total {name}: {total:.1f}{units} across {len(numbers)} readings"}
    if aggregation == "AVG":
        mean = sum(numbers) / len(numbers)
        return {"kind": "AVG", "value": round(mean, 2), "unit": units,
                "text": f"Average {name}: {mean:.1f}{units}"}
    if aggregation in {"MAX", "MIN"}:
        # `usable` skips sentinels, so -999 can never win a MIN. Reading the column any other
        # way is how "coldest: -999°C" gets printed under a summer forecast.
        usable = [r for r in rows if not is_missing(r.get(field))]
        best = (max if aggregation == "MAX" else min)(usable, key=lambda r: float(r[field]))
        word = "Highest" if aggregation == "MAX" else "Lowest"
        return {"kind": aggregation, "value": round(float(best[field]), 2), "unit": units,
                "at": best["Date_time"],
                "text": f"{word} {name}: {float(best[field]):.1f}{units} at {_when(best)}"}
    if aggregation == "TREND":
        return _trend(rows, field)
    return None


def _when(row: dict) -> str:
    """A row's timestamp, with the clock only when the feed carries one."""
    at = stamp(row)
    return f"{at:%d %b %H:%M}" if at.hour else f"{at:%d %b}"


def _trend(rows: list[dict], field: str) -> dict | None:
    """Where the series turns, in plain words - "starts dropping after 14:00"."""
    points = [(r["Date_time"], float(r[field])) for r in rows if not is_missing(r.get(field))]
    if len(points) < 3:
        return None
    name, units = label(field).lower(), unit(field)
    peak_at, peak = max(points, key=lambda p: p[1])
    low_at, low = min(points, key=lambda p: p[1])
    first, last = points[0][1], points[-1][1]
    direction = "rising" if last > first else "falling" if last < first else "flat"

    when = lambda iso: _when({"Date_time": iso})
    turn = ""
    if points.index((peak_at, peak)) < len(points) - 1 and direction != "rising":
        turn = f", starts dropping after {when(peak_at)}"
    elif points.index((low_at, low)) < len(points) - 1 and direction == "rising":
        turn = f", starts climbing after {when(low_at)}"
    return {"kind": "TREND", "value": round(last - first, 2), "unit": units,
            "text": f"{name.capitalize()} is {direction} ({first:.1f} to {last:.1f}{units}){turn}",
            "peak_at": peak_at, "low_at": low_at}


def build_chart(selected: list[list[dict]], places: list[dict], field: str, hourly: bool,
                kind: str | None = None, fields: list[str] | None = None) -> dict | None:
    """Line for a series over time, grouped bars for a short comparison, nothing for one point.

    `kind` is v3's decision and wins when given: the model read the question, whereas this
    function only ever saw row counts. MULTI_LINE plots several variables for one place, which
    the row-count heuristic could not express at all.
    """
    if kind == "NONE":
        return None
    granularity = "hourly" if hourly else "daily"

    if kind == "MULTI_LINE" and fields and len(places) == 1 and selected:
        multi = [{"name": label(f),
                  "points": [{"t": r["Date_time"], "v": float(r[f])} for r in selected[0]
                             if r.get(f) is not None]}
                 for f in fields[:3]]
        multi = [entry for entry in multi if len(entry["points"]) > 1]
        if multi:
            return {"type": "line", "field": field, "label": "Readings", "unit": unit(field),
                    "granularity": granularity, "series": multi}

    series = []
    for place, rows in zip(places, selected):
        points = [{"t": r["Date_time"], "v": float(r[field])} for r in rows
                  if r.get(field) is not None]
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
    return {"type": shape, "field": field, "label": label(field), "unit": unit(field),
            "granularity": granularity, "series": series}


def build_insights(selected: list[list[dict]], places: list[dict], fields: list[str],
                   aggregation: str, hourly: bool,
                   wanted: list[str] | None = None) -> list[Note]:
    """Two to four things worth saying that the table does not say by itself.

    `wanted` is v3's insight selection. Without it every applicable observation is emitted,
    which is the behaviour before the model chose for itself.
    """
    allow = set(wanted) if wanted else None
    keep = lambda kind: allow is None or kind in allow
    field = fields[0]
    name, units = label(field).lower(), unit(field)
    out: list[Note] = []

    # The comparison leads. Built last but placed first, because it is the answer to a
    # comparison question and the per-place notes below fill the cap on their own - with two
    # places their six notes pushed the one line that actually compared them off the end.
    if len(places) > 1 and keep("COMPARISON"):
        scored, word = [], ""
        for place, rows in zip(places, selected):
            if (numbers := values(rows, field)):
                value, word = summary_stat(field, numbers, aggregation)
                scored.append((place["name"], value))
        if len(scored) > 1:
            scored.sort(key=lambda s: -s[1])
            gap = scored[0][1] - scored[-1][1]
            out.append(Note("COMPARISON",
                            f"{scored[0][0]} leads {scored[-1][0]} by {gap:.1f}{units} "
                            f"on {word} {name}"))

    for place, rows in zip(places, selected):
        numbers = values(rows, field)
        if not numbers:
            continue
        where = place["name"] if len(places) > 1 else ""
        prefix = f"{where}: " if where else ""

        if len(numbers) > 1 and aggregation == "RAW" and keep("RANGE"):
            # `summary_stat` under RAW is a mean, never a total - a week of rain summed and
            # called "rainfall" is a bigger number for the same weather the longer you ask about
            value, word = summary_stat(field, numbers, aggregation)
            out.append(Note("RANGE",
                            f"{prefix}{word} {name} {value:.1f}{units}, "
                            f"range {min(numbers):.1f}-{max(numbers):.1f}{units}", where))

        if keep("THRESHOLD") and (threshold := NOTABLE.get(field)):
            low = field in BELOW
            crossings = [v for v in numbers if (v <= threshold[0] if low else v >= threshold[0])]
            if crossings:
                worst = min(crossings) if low else max(crossings)
                out.append(Note("THRESHOLD",
                                f"{prefix}{threshold[1]} on {len(crossings)} of {len(numbers)} "
                                f"{'readings' if hourly else 'days'} "
                                f"({'lowest' if low else 'peak'} {worst:.1f}{threshold[2]})",
                                where))

        if keep("DRY_SPELL") and field == "Rainfall" and len(numbers) > 2:
            dry = sum(1 for v in numbers if v < 1.0)
            if dry:
                out.append(Note("DRY_SPELL",
                                f"{prefix}{dry} of {len(numbers)} dry "
                                f"{'hours' if hourly else 'days'} (<1mm)", where))

    return out[:MAX_NOTES]


def demo():
    """Self-check: the guard, the reductions, and that notes carry their kind."""
    days = lambda n, **f: [{"Date_time": f"2026-08-{d:02d}T00:00:00",
                            **{k: (v[d - 1] if isinstance(v, list) else v) for k, v in f.items()}}
                           for d in range(1, n + 1)]

    # the guard, both ways: a reduction the wording never asked for is dropped, and one the
    # wording plainly asks for is added when the model missed it
    assert confirm_aggregation("weather in KKD", "MAX") == "RAW"
    assert confirm_aggregation("hottest day in KKD", "MAX") == "MAX"
    assert confirm_aggregation("weather in KKD", "RAW") == "RAW"
    assert confirm_aggregation("total rain in KKD this week", "RAW") == "SUM"
    assert confirm_aggregation("rain in KKD altogether", "RAW") == "SUM"
    assert confirm_aggregation("average temperature in KKD", "RAW") == "AVG"
    # ...but a loose phrase does not invent one. "how much rain tomorrow" is a question about
    # tomorrow, not a request for a sum.
    assert confirm_aggregation("how much rain in KKD tomorrow", "RAW") == "RAW"
    assert confirm_aggregation("overall weather in KKD", "RAW") == "RAW"

    rows = days(5, Rainfall=[1.0, 12.0, 0.0, 0.5, 3.0])
    assert apply_aggregation(rows, "Rainfall", "RAW") is None
    assert apply_aggregation(rows, "Rainfall", "SUM")["value"] == 16.5
    assert apply_aggregation(rows, "Rainfall", "MAX")["value"] == 12.0
    assert apply_aggregation(rows, "Rainfall", "MAX")["at"].startswith("2026-08-02")
    assert apply_aggregation([], "Rainfall", "SUM") is None
    # sentinels are not values - -999 must never become the minimum
    assert apply_aggregation(days(3, Rainfall=[-999, 4.0, 6.0]), "Rainfall", "MIN")["value"] == 4.0

    trend = apply_aggregation(days(4, Tmax=[34.0, 36.0, 33.0, 31.0]), "Tmax", "TREND")
    assert trend["kind"] == "TREND" and "dropping" in trend["text"], trend
    rise = apply_aggregation(days(4, Tmax=[31.0, 30.0, 33.0, 36.0]), "Tmax", "TREND")
    assert "climbing" in rise["text"], rise
    assert apply_aggregation(days(2, Tmax=[30.0, 31.0]), "Tmax", "TREND") is None   # too short

    # notes carry their kind, so the prompt layer can group them
    notes = build_insights([rows], [{"name": "Guntur"}], ["Rainfall"], "RAW", hourly=False)
    kinds = {n.kind for n in notes}
    assert "RANGE" in kinds and "THRESHOLD" in kinds and "DRY_SPELL" in kinds, notes
    assert all(n.place == "" for n in notes), "one place needs no prefix"
    assert len(build_insights([rows] * 5, [{"name": f"P{i}"} for i in range(5)],
                              ["Rainfall"], "RAW", False)) <= MAX_NOTES

    # the comparison survives the cap, because it is what a comparison question asked for
    two = build_insights([rows, days(5, Rainfall=0.0)],
                         [{"name": "Guntur"}, {"name": "Vizag"}], ["Rainfall"], "RAW", False)
    assert two[0].kind == "COMPARISON", two
    assert two[0].text.startswith("Guntur leads Vizag"), two[0]

    # v3's selection narrows it
    only_range = build_insights([rows], [{"name": "Guntur"}], ["Rainfall"], "RAW", False,
                                wanted=["RANGE"])
    assert {n.kind for n in only_range} == {"RANGE"}, only_range

    # a decision does not come with a graph unless the question asked for one
    assert not wants_chart("should i take raincoat?")
    assert wants_chart("show me a graph of rain this week")
    assert wants_chart("rain today", kind="LINE") and not wants_chart("rain today", kind="NONE")

    chart = build_chart([rows], [{"name": "Guntur"}], "Rainfall", hourly=False)
    assert chart["type"] == "line" and len(chart["series"][0]["points"]) == 5, chart
    assert build_chart([rows[:1]], [{"name": "Guntur"}], "Rainfall", False) is None  # one point

    print("analysis demo OK")
    for note in notes + two[:1]:
        print(f"  {note.kind:11s} {note.text}")


if __name__ == "__main__":
    demo()
