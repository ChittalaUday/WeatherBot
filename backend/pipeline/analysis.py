"""
Rows -> the reduction, the chart and the observations worth saying out loud.

    python tests/test_pipeline_units.py          # the checks for this module

The model decides *which* reduction the question asks for (Rule 2.3: RAW / SUM / AVG / MAX /
MIN / TREND) and whether a chart was wanted. Computing any of it is deterministic: the model
never invents a value, it only says what to compute.

Observations carry their kind, not just their wording. A flat list of sentences was fine for
a bullet list and useless to everything else - the generation layer could not tell a range
from a threshold crossing from a comparison, so it re-said them in whatever order they
happened to be built. `Note.kind` is what lets the prompt group them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from backend.pipeline import aggregate
from backend.pipeline.quality import is_missing, values
from backend.pipeline.render import label, stamp, summary_stat, unit
from src.v4.schema import FIELD_SETS, column_for, supports

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
    # Every statistic needs an entry, or `confirm_aggregation` reads an empty tuple, finds no
    # supporting word and drops it to RAW - which would have made the fifteen new ones
    # unreachable no matter what the model predicted.
    "MEDIAN": ("median", "middle", "midpoint"),
    "RANGE": ("range", "spread", "vary", "varies", "variation", "lowest to highest",
              "low to high", "swing"),
    "STDDEV": ("steady", "consistent", "variable", "variability", "standard deviation",
               "how much does it move", "stable"),
    "CHANGE": ("change", "changed", "difference", "how much has", "moved", "start to end",
               "since"),
    "CUMULATIVE": ("running total", "accumulated", "accumulating", "building up", "so far",
                   "cumulative", "adding up"),
    "COUNT": ("how many", "count", "number of", "how many days", "how many hours",
              "how many times"),
    "RUN_COUNT": ("how many spells", "spells", "bursts", "separate", "how many periods",
                  "how many times"),
    "FREQUENCY": ("how often", "frequency", "frequently", "what share", "percentage of time",
                  "share of the time", "how regular"),
    "INTENSITY": ("intensity", "how heavy", "how hard", "how intense", "when it comes"),
    "MODE": ("mostly", "dominant", "prevailing", "usually from", "mainly from",
             "most common direction", "where does it come from"),
    "DISTRIBUTION": ("breakdown", "spread across", "rose", "distribution", "by direction",
                     "each direction"),
    # "which day" was on both of these, so it supported whichever the model happened to say
    # and "which day is the hottest" was confirmed as LOW_DATE. Only the polarity word tells
    # them apart, so only the polarity word is a cue.
    "PEAK_DATE": ("hottest", "wettest", "windiest", "sunniest", "muggiest", "warmest",
                  "highest", "peak", "worst", "most", "at its peak", "at its highest",
                  # "the longest day" is a date, not a run of days - LONGEST_RUN's cue is
                  # "longest stretch"/"longest spell", which does not collide with this.
                  "largest day", "longest day", "biggest"),
    "LOW_DATE": ("coldest", "driest", "calmest", "coolest", "quietest", "least", "lowest",
                 "at its lowest", "cloudiest"),
    "PEAK_PERIOD": ("which stretch", "heaviest", "wettest period", "wettest stretch",
                    "at its heaviest", "worst period"),
    "LOW_PERIOD": ("which stretch", "driest", "lightest", "driest period", "quietest period",
                   "calmest period"),
    # not a bare "longest": "the longest day" is a PEAK_DATE and was landing here.
    "LONGEST_RUN": ("unbroken", "stretch", "spell", "spells", "run of", "in a row",
                    "back to back", "consecutive", "continuous", "how long is"),
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


def wants_chart(text: str) -> bool:
    """Did this question ask for a picture?

    The wording decides: a decision ("should I take a raincoat") and a single reduced figure
    are answers in one line, and a temperature curve under them is decoration.
    """
    return any(word in (text or "").lower() for word in CHART_WORDS)

EXPLICIT = {
    "SUM": ("total", "cumulative", "sum", "add up", "altogether"),
    "AVG": ("average", "avg", "mean", "on average"),
    # A superlative names one reading, and there is no other thing it could mean. Without
    # these the model's RAW stood, the profile's own default turned it into a total, and
    # "the largest day in June" was answered with 158 hours of day length added together.
    "PEAK_DATE": ("largest day", "longest day", "biggest day", "hottest day", "wettest day",
                  "windiest day", "sunniest day", "warmest day"),
    "LOW_DATE": ("shortest day", "coldest day", "driest day", "calmest day", "coolest day"),
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
            if _says(lowered, words):
                return kind
        return "RAW"
    return aggregation if _says(lowered, AGG_WORDS.get(aggregation, ())) else "RAW"


def _says(text: str, phrases) -> bool:
    """Is any of these actually a word here?

    Whole words, not substrings. "sum" lives inside "summarize", so every request to
    summarise the weather was confirmed as a request for a total - and a total of the general
    forecast is not a number, so the answer came back with the label and no figure.
    """
    return any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) for phrase in phrases)


def apply_aggregation(rows: list[dict], field: str, aggregation: str,
                      variable: str = "") -> dict | None:
    """The figure the question asked for, or None when it wants the rows themselves.

    Twenty of the twenty-two statistics are `backend.pipeline.aggregate`'s - one small
    function each, dispatched by a table. Two stay here: RAW means no reduction at all, and
    TREND is prose about the shape of a series rather than a number out of it.

    `variable` is what decides whether a statistic can be answered at all - a total of
    humidity is the sum of nine percentages and means nothing. Without it the field is mapped
    back to its variable, so an older caller keeps working.
    """
    if not rows or aggregation == "RAW":
        return None
    if aggregation == "TREND":
        return _trend(rows, field)
    return aggregate.compute(rows, field, variable or variable_of(field), aggregation)


# More than this under one answer is a wall of figures, not an answer.
MAX_REDUCTIONS = 6

# At this many separate findings a paragraph stops being the right shape. Below it, prose
# connects the facts ("most of the rain is in the afternoon, so the morning is your window");
# above it, prose is a list with the bullets filed off and a reader has to count commas.
STRUCTURE_AT = 4


def wants_structure(reductions: list, insights: list, places: list,
                    columns: int = 0) -> bool:
    """Is there enough here to be worth laying out rather than saying?

    Read off what the analysis found, not off the question: a turn that asked one thing and
    found six answers needs the layout, and a turn that asked six things and found one does
    not. Three signals, because a long answer gets long in three different ways:

        findings   several statistics, or several observations, or both
        places     a comparison across three of them
        columns    a full report - six measurements is a paragraph nobody finishes
    """
    return (len(reductions) + len(insights) >= STRUCTURE_AT
            or len(places) > 2
            or columns >= STRUCTURE_AT)


def pair_up(variables: list[str], aggregations: list[str]) -> list[tuple[str, str]]:
    """Which statistic to run on which variable, for a turn that asked for several.

    Three shapes, and the first two are the ones people actually say:

        one statistic, several variables   "the hottest and the rainiest day in july"
        several statistics, one variable   "the total and the average rainfall"
        several of both                    every combination the variable can support

    ponytail: the third case is a cross product, so "total rain and average temperature" also
    offers the average rain. Over-offering a figure is recoverable; dropping the one that was
    asked for is not. Pair them by where each was said once the model can emit ordered pairs.
    """
    variables = [v for v in variables if v] or ["GENERAL"]
    aggregations = [a for a in aggregations if a and a != "RAW"]
    if not aggregations:
        return []
    if len(aggregations) == 1:
        pairs = [(v, aggregations[0]) for v in variables]
    elif len(variables) == 1:
        pairs = [(variables[0], a) for a in aggregations]
    else:
        pairs = [(v, a) for v in variables for a in aggregations]
    return [(v, a) for v, a in pairs if supports(v, a)][:MAX_REDUCTIONS]


# Wording that names a specific column its variable would not otherwise pick. Day length and
# sunshine hours are both SUNSHINE, and "the longest day" means only one of them.
COLUMN_CUES = {"DayLength": ("day length", "daylight", "longest day", "largest day",
                             "length of the day", "how long the day", "day duration"),
               "Wind_Direction": ("direction", "which way", "where the wind", "wind rose"),
               "Wind_max": ("gust", "gusts", "gusty"),
               "RH_max": ("most humid", "muggiest"), "RH_min": ("least humid", "driest air")}


def column_named(text: str, available: list[str]) -> str:
    """The column the wording asks for by name, or "" when it names none."""
    lowered = (text or "").lower()
    for column, cues in COLUMN_CUES.items():
        if column in available and _says(lowered, cues):
            return column
    return ""


def reduce_all(rows: list[dict], variables: list[str], aggregations: list[str],
               available: list[str], text: str = "") -> list[dict]:
    """Every statistic this turn asked for, in the order it asked.

    `schema.column_for` decides which column each one reads - the high, the low, or the
    middle - so "the hottest day" is Tmax rather than whichever temperature column the fetch
    happened to put first. Nothing that cannot be computed appears at all.
    """
    out = []
    named = column_named(text, available)
    for variable, aggregation in pair_up(variables, aggregations):
        column = named or column_for(variable, aggregation, available)
        if (got := apply_aggregation(rows, column, aggregation, variable)):
            out.append(got)
    return out


def variable_of(field: str) -> str:
    """Which variable a column belongs to. Built once from the schema's own field sets, so a
    column added there needs no second edit here."""
    return _FIELD_VARIABLE.get(field, "GENERAL")


_FIELD_VARIABLE = {column: var.value
                   for var, columns in FIELD_SETS.items() for column in columns}


def _when(row: dict) -> str:
    """A row's timestamp, with the clock only when the feed carries one."""
    at = stamp(row)
    return f"{at:%d %b %H:%M}" if at.hour else f"{at:%d %b}"


def _trend(rows: list[dict], field: str) -> dict | None:
    """Where the series turns, in plain words - "starts dropping after 14:00"."""
    points = [(r["Date_time"], float(r[field])) for r in rows if not is_missing(r.get(field))]
    if len(points) < 3:
        return None
    first, last = points[0][1], points[-1][1]
    if abs(last - first) < 1.5:
        return None
    name, units = label(field).lower(), unit(field)
    peak_at, peak = max(points, key=lambda p: p[1])
    low_at, low = min(points, key=lambda p: p[1])
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


# Which picture a turn wants. Weather has its own shapes and a line does not stand in for
# them: a day's temperature is a band between two readings, a wind direction is a rose, and a
# season's rain is an accumulation. Each entry is (type, what it needs), checked in order.
CHART_TYPES = ("rose", "area", "band", "combo", "heatmap", "bar", "line")


def _points(rows: list[dict], field: str) -> list[dict]:
    return [{"t": r["Date_time"], "v": float(r[field])} for r in rows
            if not is_missing(r.get(field))]


def _series(selected, places, field) -> list[dict]:
    out = []
    for place, rows in zip(places, selected):
        if (points := _points(rows, field)):
            out.append({"name": place["name"], "points": points})
    return out


def _head(field: str) -> dict:
    return {"field": field, "label": label(field), "unit": unit(field)}


def pick_chart(fields: list[str], aggregations: list[str], hourly: bool, places: int,
               points: int, days: int) -> str:
    """The shape this answer should be drawn as.

    Ordered by how specific the claim is: a wind rose is only ever right for a direction, a
    band only for a high-and-low pair. The general shapes are last, so a specific one is never
    shadowed by "several places, therefore bars".
    """
    asked = set(aggregations)
    # Only when the direction is what was asked about - `fields` leads on the variable the
    # question named. A full summary carries a direction column among sixteen others, and
    # firing on its mere presence opened every summary on a wind rose.
    #
    # Unconditional on the statistic though, not only for a distribution: a bearing is
    # circular, so a line through 350° and 10° descends through south to join two readings
    # 20° apart. The rose is the only honest picture of a direction.
    if fields and fields[0] == "Wind_Direction":
        return "rose"
    if "CUMULATIVE" in asked:
        return "area"
    if places == 1 and {"Tmin", "Tmax"} <= set(fields):
        return "band"
    if places == 1 and "Rainfall" in fields and ({"Tavg", "Tmax"} & set(fields)) and points >= 3:
        return "combo"
    if places == 1 and hourly and days >= 2:
        return "heatmap"
    if places > 1 and points <= 8:
        return "bar"
    return "line"


def build_chart(selected: list[list[dict]], places: list[dict], fields: list[str],
                hourly: bool, aggregations: list[str] | None = None) -> dict | None:
    """The picture, in whichever shape fits what was asked. None when there is nothing to draw.

    Every shape carries the same envelope - type, label, unit, granularity - so a client
    switches on `type` and reads the one extra key that shape needs, rather than sniffing the
    payload to work out what it was given.
    """
    if not places or not fields:
        return None
    field = fields[0]
    rows = selected[0] if selected else []
    days = len({stamp(r).date() for r in rows})
    series = _series(selected, places, field)
    points = max((len(s["points"]) for s in series), default=0)
    shape = pick_chart(fields, aggregations or [], hourly, len(places), points, days)
    envelope = {"granularity": "hourly" if hourly else "daily", **_head(field)}

    if shape == "rose":
        return _rose(rows, envelope)
    if shape == "band":
        return _band(rows, envelope)
    if shape == "combo":
        return _combo(rows, fields, envelope)
    if shape == "heatmap":
        return _heatmap(rows, field, envelope)
    # One reading is not a series - but it is a perfectly good comparison. Three places for
    # tomorrow is three bars, and the old guard drew nothing at all for exactly the question a
    # chart answers best.
    if not series or (points < 2 and len(series) < 2):
        return None
    if shape == "area":
        return {**envelope, "type": "area", "series": _running(series)}
    return {**envelope, "type": shape, "series": series}


# More than this and the picker is a wall of buttons rather than a choice.
MAX_CHARTS = 12


def build_charts(selected: list[list[dict]], places: list[dict], fields: list[str],
                 hourly: bool, aggregations: list[str] | None = None) -> list[dict]:
    """One ready chart per field worth plotting, for a turn that measured several things.

    A summary fetches seventeen columns and draws one of them. The other sixteen were in the
    table and nowhere else, so the reader could see the numbers and never the shape.

    Each field is asked for on its own - `pick_chart` reads the whole field list, so handing
    it all of them returns the same shape every time. The temperature pair is the exception:
    a low and a high are one chart, not two, so they are offered together as a band.
    """
    if len(fields) < 2:
        return []
    out, paired = [], set()

    def offer(name: str, label_text: str, unit_text: str, wanted: list[str]) -> None:
        chart = build_chart(selected, places, wanted, hourly, aggregations)
        if chart:
            out.append({"field": name, "label": label_text, "unit": unit_text, "chart": chart})

    if {"Tmin", "Tmax"} <= set(fields):
        paired = {"Tmin", "Tmax"}
        offer("Temperature", "Temperature", unit("Tmax"),
              [f for f in ("Tmin", "Tmax", "Tavg") if f in fields])

    for field_name in fields:
        if field_name in paired or len(out) >= MAX_CHARTS:
            continue
        offer(field_name, label(field_name), unit(field_name), [field_name])
    return out[:MAX_CHARTS]


def _rose(rows, envelope) -> dict | None:
    """Wind direction as sixteen petals - the one chart that is only ever right for a bearing."""
    got = aggregate.compute(rows, "Wind_Direction", "WIND", "DISTRIBUTION")
    if not got or not got.get("buckets"):
        return None
    return {**envelope, "type": "rose", "field": "Wind_Direction",
            "label": "Wind direction", "unit": "", "buckets": got["buckets"]}


def _band(rows, envelope) -> dict | None:
    """The day's low and high as a filled band. A single temperature line hides the swing that
    the whole question is usually about."""
    band = [{"t": r["Date_time"], "lo": float(r["Tmin"]), "hi": float(r["Tmax"]),
             "v": float(r.get("Tavg") or (float(r["Tmin"]) + float(r["Tmax"])) / 2)}
            for r in rows if not is_missing(r.get("Tmin")) and not is_missing(r.get("Tmax"))]
    if len(band) < 2:
        return None
    return {**envelope, "type": "band", "field": "Tmax", "label": "Temperature",
            "unit": unit("Tmax"), "points": band}


def _combo(rows, fields, envelope) -> dict | None:
    """Rain as bars, temperature as a line over them - the shape every forecast page uses,
    because the two are read together and one axis cannot hold both."""
    warm = next((f for f in ("Tavg", "Tmax") if f in fields), "")
    bars, line = _points(rows, "Rainfall"), _points(rows, warm)
    if len(bars) < 2 or len(line) < 2:
        return None
    return {**envelope, "type": "combo", "label": "Rain and temperature", "unit": "",
            "bars": {**_head("Rainfall"), "points": bars},
            "line": {**_head(warm), "points": line}}


def _heatmap(rows, field, envelope) -> dict | None:
    """Hour of day across days. A week of hourly readings is 168 points, which is a smear as a
    line and a grid as a heatmap - and "it rains every afternoon" is only visible as a grid."""
    cells = [{"d": stamp(r).strftime("%Y-%m-%d"), "h": stamp(r).hour, "v": float(r[field])}
             for r in rows if not is_missing(r.get(field))]
    if len(cells) < 12:
        return None
    return {**envelope, "type": "heatmap", "cells": cells,
            "days": sorted({c["d"] for c in cells})}


def _running(series: list[dict]) -> list[dict]:
    """Each series as its own running total."""
    out = []
    for one in series:
        total = 0.0
        points = []
        for point in one["points"]:
            total += point["v"]
            points.append({"t": point["t"], "v": round(total, 2)})
        out.append({"name": one["name"], "points": points})
    return out


def build_insights(selected: list[list[dict]], places: list[dict], fields: list[str],
                   aggregation: str, hourly: bool) -> list[Note]:
    """Two to four things worth saying that the table does not say by itself.

    Every applicable observation is emitted; the cap below is what keeps it readable.
    """
    field = fields[0]
    name, units = label(field).lower(), unit(field)
    out: list[Note] = []

    # The comparison leads. Built last but placed first, because it is the answer to a
    # comparison question and the per-place notes below fill the cap on their own - with two
    # places their six notes pushed the one line that actually compared them off the end.
    if len(places) > 1:
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

        if len(numbers) > 1 and aggregation == "RAW":
            # `summary_stat` under RAW is a mean, never a total - a week of rain summed and
            # called "rainfall" is a bigger number for the same weather the longer you ask about
            value, word = summary_stat(field, numbers, aggregation)
            out.append(Note("RANGE",
                            f"{prefix}{word} {name} {value:.1f}{units}, "
                            f"range {min(numbers):.1f}-{max(numbers):.1f}{units}", where))

        if (threshold := NOTABLE.get(field)):
            low = field in BELOW
            crossings = [v for v in numbers if (v <= threshold[0] if low else v >= threshold[0])]
            if crossings:
                worst = min(crossings) if low else max(crossings)
                out.append(Note("THRESHOLD",
                                f"{prefix}{threshold[1]} on {len(crossings)} of {len(numbers)} "
                                f"{'readings' if hourly else 'days'} "
                                f"({'lowest' if low else 'peak'} {worst:.1f}{threshold[2]})",
                                where))

        if field == "Rainfall" and len(numbers) > 2:
            dry = sum(1 for v in numbers if v < 1.0)
            if dry:
                out.append(Note("DRY_SPELL",
                                f"{prefix}{dry} of {len(numbers)} dry "
                                f"{'hours' if hourly else 'days'} (<1mm)", where))

    return out[:MAX_NOTES]
