"""
Rows -> the table and the one-line summary the UI draws.

    python tests/test_pipeline_units.py          # the checks for this module

The deterministic half of MODEL_RULES.md Section 1: the model says *what* was asked; which
API field a variable means, how a COMPARE is laid out and how a number is written down are
plain rules, no learning. Nothing here invents a figure.

This module also owns the field vocabulary - the label and the unit for every column the
feeds serve. `analysis`, `advice` and `generation` all read those from here rather than
carrying their own copies, so a unit is written one way everywhere.
"""

from __future__ import annotations

from datetime import datetime

from backend.pipeline.quality import values

LABELS = {
    "Tavg": "Avg temp", "Tmin": "Min temp", "Tmax": "Max temp", "Rainfall": "Rainfall",
    "RH": "Humidity", "RH_max": "Humidity max", "RH_min": "Humidity min", "DPT": "Dew point",
    "Wind_Speed": "Wind speed", "Wind_max": "Wind gust", "Wind_Direction": "Wind direction",
    "SunSD": "Sunshine", "DayLength": "Day length", "Lowcloud": "Cloud cover",
    "Soilm10": "Soil moisture 10cm", "Soilm40": "Soil moisture 40cm",
    "Soilt10": "Soil temp 10cm",
    # archive-only: the climatic normals, which make "wetter than usual" answerable
    "Normal_Rainfall": "Normal rainfall", "Normal_Tmax": "Normal max temp",
    "Normal_Tmin": "Normal min temp", "Normal_RH": "Normal humidity",
}
UNITS = {
    "Tavg": "°C", "Tmin": "°C", "Tmax": "°C", "DPT": "°C", "Soilt10": "°C",
    "Rainfall": "mm", "RH": "%", "RH_max": "%", "RH_min": "%",
    "Wind_Speed": "m/s", "Wind_max": "m/s", "Wind_Direction": "°",
    "SunSD": "hrs", "DayLength": "hrs", "Lowcloud": "frac",
    "Soilm10": "m³/m³", "Soilm40": "m³/m³",
    "Normal_Rainfall": "mm", "Normal_Tmax": "°C", "Normal_Tmin": "°C", "Normal_RH": "%",
}
# Fields where adding the readings up is a meaningful thing to do at all. Note what this is
# NOT: a licence to add them up. Summing is a reduction, and Rule 2.3 says a reduction is
# reported only when the question asked for one - "total", "how much", "altogether".
ADDITIVE = {"Rainfall", "SunSD"}

COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


def label(field: str) -> str:
    return LABELS.get(field, field)


def unit(field: str) -> str:
    return UNITS.get(field, "")


def summary_stat(field: str, numbers: list[float],
                 aggregation: str = "RAW") -> tuple[float, str]:
    """The one figure that stands for a series, and the word for what that figure is.

    Adding is only ever done when the question asked for it. Rainfall used to be totalled
    whenever the field was additive, regardless of the aggregation slot - so "rain in Guntur
    this week" came back "12.5mm", which is a week's accumulation presented as though it were
    the rainfall, and "compare rain in Guntur and Vizag" ranked two places by a total neither
    person asked for. The wider the window, the bigger the number, for the same weather.

    Under RAW the answer describes the series instead: the mean per reading, with the range
    and the wet-reading count carried by the insights beside it.
    """
    if not numbers:
        return 0.0, ""
    if aggregation == "SUM" and field in ADDITIVE:
        return sum(numbers), "total"
    return sum(numbers) / len(numbers), "average"


def stamp(row: dict) -> datetime:
    return datetime.fromisoformat(row["Date_time"])


def format_value(field: str, value) -> str:
    """One cell. Compass points and percentages are written the way people read them."""
    if value is None:
        return "-"
    if field == "Wind_Direction":
        return f"{float(value):.0f}° {COMPASS[int((float(value) % 360) / 22.5 + 0.5) % 16]}"
    if field == "Lowcloud":
        return f"{float(value) * 100:.0f}%"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def build_table(rows, fields: list[str], places: list[dict], hourly: bool) -> dict:
    """One table: a column per field for one place, or a column per place for a comparison."""
    time_format = "%d %b %H:%M" if hourly else "%d %b"

    if len(places) <= 1:
        columns = [{"key": "time", "label": "When"}] + [
            {"key": f, "label": f"{label(f)} ({unit(f)})".strip()} for f in fields]
        body = [{"time": stamp(r).strftime(time_format),
                 **{f: format_value(f, r.get(f)) for f in fields}} for r in rows]
        return {"columns": columns, "rows": body}

    # COMPARE: `rows` is one list per place, zipped on the timestamp
    field = fields[0]
    columns = [{"key": "time", "label": "When"}] + [
        {"key": p["name"], "label": f"{p['name']} ({unit(field)})".strip()} for p in places]
    by_time: dict[str, dict] = {}
    for place, place_rows in zip(places, rows):
        for row in place_rows:
            when = stamp(row).strftime(time_format)
            by_time.setdefault(when, {"time": when})[place["name"]] = format_value(field,
                                                                                  row.get(field))
    return {"columns": columns, "rows": list(by_time.values())}


def summarize(action: str, rows, fields: list[str], places: list[dict], when: str,
              aggregation: str = "RAW") -> str:
    """The conclusion, in one deterministic sentence. Everything downstream re-says this."""
    if not rows:
        return "No forecast rows matched that time range."
    field = fields[0]
    name, units = label(field).lower(), unit(field)
    where = " vs ".join(p["name"] for p in places) if places else "your location"

    if action == "COMPARE" and len(places) > 1:
        scored, word = [], ""
        for place, place_rows in zip(places, rows):
            if (numbers := values(place_rows, field)):
                value, word = summary_stat(field, numbers, aggregation)
                scored.append((place["name"], value))
        if not scored:
            return f"{name.capitalize()} for {when} in {where}."
        leader = max(scored, key=lambda s: s[1])
        rest = " · ".join(f"{n} {v:.1f}{units}" for n, v in scored if n != leader[0])
        # The statistic is named, not implied. "Guntur 12.5mm" for a week is a total and reads
        # as a reading; saying which it is costs three words and stops the sentence lying.
        return (f"{leader[0]} has the {'higher' if len(scored) == 2 else 'highest'} {word} "
                f"{name} for {when}: {leader[1]:.1f}{units} against {rest}.")

    flat = rows if isinstance(rows[0], dict) else [r for group in rows for r in group]
    numbers = values(flat, field)
    if not numbers:
        return f"No {name} values for {when} in {where}."

    if action == "ALERT":
        return (f"Watching {name} in {where} for {when}. Peak in range: "
                f"{max(numbers):.1f}{units}. Alerts are reported live in this chat, not "
                f"persisted between sessions.")
    if len(numbers) == 1:
        return f"{name.capitalize()} in {where} {when}: {numbers[0]:.1f}{units}."
    if field == "Rainfall":
        # Always the series, never the total - even when a total was asked for. The total is
        # `analysis.apply_aggregation`'s job and leads the sentence already; repeating it here
        # produced "Total rainfall: 9.9mm across 7 readings. Guntur, this week: 9.9mm total".
        # A week of drizzle and one thunderstorm add to the same number and are not the same
        # forecast; the peak and the wet count are what separate them.
        wet = sum(1 for v in numbers if v >= 1.0)
        peak = max(numbers)
        if not wet:
            return (f"{where}, {when}: little to no rain, at most {peak:.1f}mm in a reading.")
        return (f"{where}, {when}: rain on {wet} of {len(numbers)} readings, "
                f"up to {peak:.1f}mm in one.")
    return (f"{name.capitalize()} in {where} for {when}: "
            f"{min(numbers):.1f}-{max(numbers):.1f}{units} "
            f"(avg {sum(numbers) / len(numbers):.1f}{units}).")


# ---------------------------------------------------------------------------
# What to put on screen
# ---------------------------------------------------------------------------
#
# The table and the chart are always in the payload. This only says which of them to *open*,
# so nothing is hidden from anyone who wants it - it just is not shoved at everyone. A client
# reads `presentation` and renders; it does not re-derive the decision from row counts, which
# is how two clients end up disagreeing about the same answer.

VIEWS = ("chart", "table", "none")

CHOOSE = """You pick how a weather answer is shown. Reply with ONLY {"show":"...","why":"..."}.

show is one of:
  "chart" - the shape over time is the point. Use it when points >= 4: a run of readings
            across hours or days, a rise, a fall, when rain starts.
  "table" - the individual values are the point. Use it when there is no series worth a
            curve but there is a grid to scan: columns >= 3, or several places side by side.
  "none"  - the sentence already answered it. Use it for a yes/no decision (verdict: True),
            a single figure (one_figure: True), or one or two rows.

why is at most eight words."""
# The thresholds are spelled out rather than left to judgement. Without the numbers a small
# model answers "none" to almost everything - it reads the "when in doubt" instruction that
# used to be here and stops thinking. With them the counts do the work and the question only
# breaks the ties.


def facts(answer) -> dict:
    """The countable half of the decision. The rule below and the model both read only this,
    so what the model is choosing between is exactly what the rule would have chosen."""
    body = (answer.table or {}).get("rows") or []
    chart = answer.chart or {}
    points = max((len(s["points"]) for s in chart.get("series") or []), default=0)
    return {
        "rows": len(body),
        # minus the time column, which is never a value anyone came for
        "columns": max(len((answer.table or {}).get("columns") or []) - 1, 0),
        "points": points,
        "places": len(answer.places),
        "has_chart": bool(answer.chart),
        "verdict": bool(answer.advice),
        "one_figure": bool(answer.reduced),
    }


def presentation(answer, view: str = "", why: str = "", decided_by: str = "rule") -> dict:
    """What this answer needs on screen, as the wire shape a client renders.

    `view` is a choice already made - the model's. Left empty the rule below decides, and it
    is also the floor under the model: a "chart" for an answer that has no chart is not a
    choice, it is a broken screen, so it is corrected here rather than trusted.
    """
    seen = facts(answer)
    if view not in VIEWS:
        view, decided_by = "", "rule"
    if not view:
        why = ""
        if seen["verdict"] or seen["one_figure"]:
            # the verdict and its reasons are the answer; a curve under "yes, spray today"
            # is decoration
            view = "none"
        elif seen["points"] >= 4:
            view = "chart"                       # enough of a series to have a shape
        elif seen["rows"] >= 6 or seen["columns"] >= 3:
            view = "table"                       # a lot of values and no single shape
        else:
            view = "none"                        # the sentence already said it
    # A view the payload cannot fill is downgraded, not shown empty.
    if view == "chart" and not seen["has_chart"]:
        view, why, decided_by = ("table" if seen["rows"] else "none"), "no series to draw", "rule"
    if view == "table" and not seen["rows"]:
        view, why, decided_by = "none", "no rows to show", "rule"

    return {
        "detail": view,
        "chart": "open" if view == "chart" else ("available" if seen["has_chart"] else "none"),
        "table": "open" if view == "table" else ("available" if seen["rows"] else "none"),
        "rows": seen["rows"], "columns": seen["columns"],
        "decided_by": decided_by,
        "why": {"chart": f"{seen['points']} points over time",
                "table": f"{seen['rows']} rows x {seen['columns']} values",
                "chose": why},
    }


async def choose(answer, question: str, spec=None, client=None) -> dict:
    """The same decision, made by the local model instead of the rule.

    Cheap on purpose - the counts and the question, nothing else - because it runs beside the
    phrasing stream, which is slower by an order of magnitude, and only stays free while it is.
    A model that is down, slow or talking nonsense falls through to `presentation()`, so the
    screen is never waiting on it.
    """
    from backend.nlu import llm

    spec = spec or llm.LOCAL
    if (answer.presentation or {}).get("decided_by") == "question":
        return answer.presentation               # the reader asked to see it; not the model's call
    seen = facts(answer)
    if not seen["rows"] and not seen["has_chart"]:
        return presentation(answer)              # nothing to show; do not spend a call saying so
    asked = f'Question: "{question}"\n' + ", ".join(f"{k}: {v}" for k, v in seen.items())
    got = await llm.chat_json(CHOOSE, asked, spec, client, max_tokens=60)
    if not got.get("ok"):
        return presentation(answer)
    said = got.get("json") or {}
    return presentation(answer, str(said.get("show", "")).strip().lower(),
                        str(said.get("why", ""))[:80], spec.name)
