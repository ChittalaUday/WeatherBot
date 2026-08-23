"""
Rows -> the table and the one-line summary the UI draws.

    python -m backend.pipeline.render        # self-check

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


def demo():
    """Self-check: a comparison names the winner, and says which way it won.

    "Min temp: A 21.3 · B 20.5. Highest: A." was true and unreadable - both a human skimming
    and the phrasing model downstream read "min ... A" as "A is the lower one", and the chat
    duly said the opposite of the data.
    """
    places = [{"name": "Hyderabad"}, {"name": "Vijawada"}]
    rows = [[{"Date_time": "2026-08-14T00:00:00", "Tmin": 21.3}],
            [{"Date_time": "2026-08-14T00:00:00", "Tmin": 20.5}]]
    said = summarize("COMPARE", rows, ["Tmin"], places, "tomorrow")
    assert said.startswith("Hyderabad has the higher"), said
    assert "21.3" in said and "Vijawada 20.5" in said, said

    # Adding happens only when a total was asked for. This is the whole rule: the same week
    # of rain is a mean under RAW and a total under SUM, and never a total by accident.
    assert summary_stat("Rainfall", [1.0, 2.0, 3.0], "SUM") == (6.0, "total")
    assert summary_stat("Rainfall", [1.0, 2.0, 3.0], "RAW") == (2.0, "average")
    assert summary_stat("Tmax", [10.0, 20.0], "SUM") == (15.0, "average")   # never additive
    assert summary_stat("Rainfall", [], "SUM") == (0.0, "")

    # ...and the sentence follows it. A week of rain under RAW describes the series; the total
    # appears only when the question said "total".
    week = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": v}
            for d, v in enumerate([0.2, 6.0, 0.0, 0.1, 3.0], start=14)]
    raw = summarize("GET", week, ["Rainfall"], places[:1], "this week")
    assert "total" not in raw and "rain on 2 of 5 readings" in raw, raw
    assert "up to 6.0mm" in raw, raw
    # the description does not change when a total was asked for - the total is said once, by
    # `analysis.apply_aggregation`, and this sentence follows it
    assert summarize("GET", week, ["Rainfall"], places[:1], "this week", "SUM") == raw
    dry = summarize("GET", [{"Date_time": "2026-08-14T00:00:00", "Rainfall": 0.1},
                            {"Date_time": "2026-08-15T00:00:00", "Rainfall": 0.0}],
                    ["Rainfall"], places[:1], "tomorrow")
    assert "little to no rain" in dry, dry

    # a comparison names which statistic it ranked on, so "12.5mm" cannot read as a reading
    compared = summarize("COMPARE", [week, week[:2]], ["Rainfall"], places, "this week")
    assert "average rainfall" in compared, compared

    # sentinels never reach a summary - `values` filters them, so this is 4.2 and nothing else
    junk = [{"Date_time": "2026-08-14T00:00:00", "Rainfall": -999},
            {"Date_time": "2026-08-15T00:00:00", "Rainfall": 4.2}]
    assert "4.2mm" in summarize("GET", junk, ["Rainfall"], places[:1], "tomorrow")

    # a column the feed never sent must not become a table column of dashes
    table = build_table([{"Date_time": "2026-08-14T16:00:00", "RH": 58.8}], ["RH"], places[:1],
                        hourly=True)
    assert [c["key"] for c in table["columns"]] == ["time", "RH"], table["columns"]
    assert table["rows"] == [{"time": "14 Aug 16:00", "RH": "58.8"}], table["rows"]

    # a comparison table is one column per place, zipped on time
    wide = build_table(rows, ["Tmin"], places, hourly=False)
    assert [c["key"] for c in wide["columns"]] == ["time", "Hyderabad", "Vijawada"], wide
    assert wide["rows"] == [{"time": "14 Aug", "Hyderabad": "21.3", "Vijawada": "20.5"}], wide

    assert format_value("Wind_Direction", 90) == "90° E", format_value("Wind_Direction", 90)
    assert format_value("Lowcloud", 0.42) == "42%"
    assert format_value("Rainfall", None) == "-"

    print("render demo OK")
    print(f"  {said}")


if __name__ == "__main__":
    demo()
