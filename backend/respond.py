"""
Turns one NLUOutput + forecast rows into the table the chat renders.

This is the deterministic half of MODEL_RULES.md Section 1: the model says *what* was asked
(intent, action, spans); everything here - which API field that intent means, which rows the
time expression selects, how a COMPARE is laid out - is plain rules, no learning.
"""

from __future__ import annotations

from datetime import datetime, timedelta

# Rule 1: weather_intent -> WeatherSnap field names. Never learned, always mapped.
INTENT_FIELDS = {
    "CURRENT_CONDITIONS": ["Tavg", "Rainfall", "RH", "Wind_Speed", "Lowcloud"],
    "FORECAST": ["Tmin", "Tmax", "Rainfall", "RH", "Wind_Speed"],
    "TEMPERATURE": ["Tavg"],
    "TEMPERATURE_MIN": ["Tmin"],
    "TEMPERATURE_MAX": ["Tmax"],
    "RAIN": ["Rainfall"],
    "HUMIDITY": ["RH", "RH_max", "RH_min"],
    "DEW_POINT": ["DPT"],
    "WIND_SPEED": ["Wind_Speed", "Wind_max"],
    "WIND_DIRECTION": ["Wind_Direction"],
    "SUNSHINE": ["SunSD", "DayLength"],
    "CLOUD_COVER": ["Lowcloud"],
    "SOIL_MOISTURE": ["Soilm10", "Soilm40"],
    "SOIL_TEMPERATURE": ["Soilt10"],
}
LABELS = {
    "Tavg": "Avg temp", "Tmin": "Min temp", "Tmax": "Max temp", "Rainfall": "Rainfall",
    "RH": "Humidity", "RH_max": "Humidity max", "RH_min": "Humidity min", "DPT": "Dew point",
    "Wind_Speed": "Wind speed", "Wind_max": "Wind gust", "Wind_Direction": "Wind direction",
    "SunSD": "Sunshine", "DayLength": "Day length", "Lowcloud": "Cloud cover",
    "Soilm10": "Soil moisture 10cm", "Soilm40": "Soil moisture 40cm", "Soilt10": "Soil temp 10cm",
}
UNITS = {
    "Tavg": "°C", "Tmin": "°C", "Tmax": "°C", "DPT": "°C", "Soilt10": "°C",
    "Rainfall": "mm", "RH": "%", "RH_max": "%", "RH_min": "%",
    "Wind_Speed": "m/s", "Wind_max": "m/s", "Wind_Direction": "°",
    "SunSD": "hrs", "DayLength": "hrs", "Lowcloud": "frac", "Soilm10": "m³/m³", "Soilm40": "m³/m³",
}
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

PART_OF_DAY = {           # canonical time -> hour window, used against the hourly endpoint
    "this morning": (6, 11), "early morning": (4, 7), "tomorrow morning": (6, 11),
    "this afternoon": (12, 16), "tomorrow afternoon": (12, 16),
    "this evening": (17, 21), "tomorrow evening": (17, 21),
    "tonight": (19, 23), "tomorrow night": (19, 23), "midnight": (0, 1),
}
DAY_OFFSET = {"today": 0, "tonight": 0, "this morning": 0, "this afternoon": 0,
              "this evening": 0, "early morning": 0, "now": 0, "midnight": 0,
              "tomorrow": 1, "tomorrow morning": 1, "tomorrow afternoon": 1,
              "tomorrow evening": 1, "tomorrow night": 1, "day after tomorrow": 2,
              "yesterday": -1}
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def _parse(row: dict) -> datetime:
    return datetime.fromisoformat(row["Date_time"])


def needs_hourly(normalized: str) -> bool:
    """Intra-day expressions need the hourly feed; whole-day ones do not."""
    if not normalized:
        return False
    return (normalized in PART_OF_DAY or normalized == "now"
            or ":" in normalized or "hours" in normalized or "minutes" in normalized)


def select_rows(rows: list[dict], normalized: str, now: datetime | None = None) -> tuple[list[dict], str]:
    """Rows the temporal expression asks for, plus a human label for the range."""
    if not rows:
        return [], "no data"
    # Real wall-clock, not rows[0]: the feed can start days in the past, and "tomorrow" must
    # mean tomorrow rather than "the day after whatever the API happened to send first".
    now = now or datetime.now()
    hourly = any(_parse(r).hour for r in rows[:24])
    # The feed starts a couple of days back; forward-looking questions must not be answered
    # with rows that already happened.
    if normalized not in {"yesterday", "last week"}:
        rows = [r for r in rows if _parse(r).date() >= now.date()] or rows

    if not normalized:                                    # no time mentioned -> the near term
        picked = rows[:7] if not hourly else rows[:12]
        return picked, "next few days" if not hourly else "next few hours"

    if normalized == "now":
        future = [r for r in rows if _parse(r) >= now]
        return (future or rows)[:1], "now"

    if "-" in normalized and ":" in normalized:           # "07:00-11:00"
        start, end = (int(part.split(":")[0]) for part in normalized.split("-"))
        picked = [r for r in rows if start <= _parse(r).hour <= end]
        return picked or rows[:1], normalized

    if ":" in normalized:                                 # "18:45" -> that hour, next occurrence
        hour = int(normalized.split(":")[0])
        at_hour = [r for r in rows if _parse(r).hour == hour]
        upcoming = [r for r in at_hour if _parse(r) >= now]
        return (upcoming or at_hour or rows)[:1], normalized

    if normalized in PART_OF_DAY:
        start, end = PART_OF_DAY[normalized]
        target = (now + timedelta(days=DAY_OFFSET.get(normalized, 0))).date()
        picked = [r for r in rows if _parse(r).date() == target and start <= _parse(r).hour <= end]
        return picked or rows[:6], normalized

    if normalized.startswith("next ") and normalized.split()[-1] in {"days", "hours", "minutes", "weeks"}:
        count = int(normalized.split()[1])
        unit = normalized.split()[-1]
        if unit == "days":
            return rows[:count], normalized
        if unit == "weeks":
            return rows[:count * 7], normalized
        return rows[:count if unit == "hours" else 1], normalized

    if normalized in DAY_OFFSET:
        target = (now + timedelta(days=DAY_OFFSET[normalized])).date()
        picked = [r for r in rows if _parse(r).date() == target]
        return picked or rows[:1], normalized

    if normalized in WEEKDAYS:
        wanted = WEEKDAYS.index(normalized)
        picked = [r for r in rows if _parse(r).weekday() == wanted]
        return picked[:24] or rows[:1], normalized

    if normalized in {"this week", "next week"}:
        offset = 0 if normalized == "this week" else 7
        return rows[offset:offset + 7], normalized
    if normalized in {"this weekend", "next weekend"}:
        picked = [r for r in rows if _parse(r).weekday() >= 5]
        return picked or rows[:7], normalized
    if normalized in {"this month", "next month"}:
        return rows[:14], normalized

    return rows[:7], normalized                            # unknown expression: show the horizon


def _format(field: str, value) -> str:
    if value is None:
        return "-"
    if field == "Wind_Direction":
        return f"{value:.0f}° {COMPASS[int((float(value) % 360) / 22.5 + 0.5) % 16]}"
    if field == "Lowcloud":
        return f"{float(value) * 100:.0f}%"
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def build_table(rows: list[dict], fields: list[str], places: list[dict], hourly: bool) -> dict:
    """One table for GET/ALERT (a column per field) or COMPARE (a column per place)."""
    time_format = "%d %b %H:%M" if hourly else "%d %b"
    if len(places) <= 1:
        columns = [{"key": "time", "label": "When"}] + [
            {"key": f, "label": f"{LABELS.get(f, f)} ({UNITS.get(f, '')})".strip()} for f in fields]
        body = [{"time": _parse(r).strftime(time_format),
                 **{f: _format(f, r.get(f)) for f in fields}} for r in rows]
        return {"columns": columns, "rows": body}

    # COMPARE: rows are (place, rows) pairs zipped on time
    field = fields[0]
    columns = [{"key": "time", "label": "When"}] + [
        {"key": p["name"], "label": f"{p['name']} ({UNITS.get(field, '')})".strip()} for p in places]
    by_time: dict[str, dict] = {}
    for place, place_rows in zip(places, rows):
        for row in place_rows:
            stamp = _parse(row).strftime(time_format)
            by_time.setdefault(stamp, {"time": stamp})[place["name"]] = _format(field, row.get(field))
    return {"columns": columns, "rows": list(by_time.values())}


def summarize(intent: str, action: str, rows: list[dict], fields: list[str],
              places: list[dict], when: str) -> str:
    """The sentence above the table. Deterministic, no model involved."""
    if not rows:
        return "No forecast rows matched that time range."
    where = " vs ".join(p["name"] for p in places) if places else "your location"
    field = fields[0]
    label = LABELS.get(field, field).lower()
    unit = UNITS.get(field, "")

    if action == "COMPARE" and len(places) > 1:
        totals = []
        for place, place_rows in zip(places, rows):
            values = [float(r[field]) for r in place_rows if r.get(field) is not None]
            if values:
                mean = sum(values) / len(values)
                totals.append((place["name"], sum(values) if field == "Rainfall" else mean))
        if totals:
            leader = max(totals, key=lambda t: t[1])
            spread = " · ".join(f"{name} {value:.1f}{unit}" for name, value in totals)
            return f"{label.capitalize()} for {when}: {spread}. Highest: {leader[0]}."
        return f"{label.capitalize()} for {when} in {where}."

    flat = rows if isinstance(rows[0], dict) else [r for group in rows for r in group]
    values = [float(r[field]) for r in flat if r.get(field) is not None]
    if not values:
        return f"No {label} values for {when} in {where}."

    if action == "ALERT":
        peak = max(values)
        return (f"Watching {label} in {where} for {when}. Peak in range: {peak:.1f}{unit}. "
                f"Alerts are reported live in this chat, not persisted between sessions.")
    if field == "Rainfall":
        total = sum(values)
        verdict = "rain expected" if total >= 1 else "little to no rain"
        return f"{where}, {when}: {total:.1f}mm total - {verdict}."
    if len(values) == 1:
        return f"{label.capitalize()} in {where} {when}: {values[0]:.1f}{unit}."
    return (f"{label.capitalize()} in {where} for {when}: "
            f"{min(values):.1f}-{max(values):.1f}{unit} (avg {sum(values) / len(values):.1f}{unit}).")
