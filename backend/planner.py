"""
Query planner - canonical time windows, and the decision to answer or ask.

Temporal resolution is rules, not a model: "day after tomorrow" is arithmetic, and a
classifier that got it wrong 2% of the time would be strictly worse than a calendar. The
model's only temporal job is to spot the span and canonicalise its wording (Rule 4.3); this
turns that wording into an absolute window.

The validator is the last gate before the weather API:

    READY    every slot the query needs is present
    CLARIFY  something is missing or ambiguous - ask, do not guess
    REJECT   nothing weather-shaped was said
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from src.schema import Operation, ResolvedQuery, Verdict

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PART_OF_DAY = {
    "early morning": (4, 7), "this morning": (6, 11), "tomorrow morning": (6, 11),
    "this afternoon": (12, 16), "tomorrow afternoon": (12, 16),
    "this evening": (17, 21), "tomorrow evening": (17, 21),
    "tonight": (19, 23), "tomorrow night": (19, 23), "midnight": (0, 1),
}
DAY_OFFSET = {
    "now": 0, "today": 0, "tonight": 0, "this morning": 0, "this afternoon": 0,
    "this evening": 0, "early morning": 0, "midnight": 0, "yesterday": -1,
    "tomorrow": 1, "tomorrow morning": 1, "tomorrow afternoon": 1, "tomorrow evening": 1,
    "tomorrow night": 1, "day after tomorrow": 2,
}
DEFAULT_HORIZON_DAYS = 7


def _window(day: date, start_hour: int, end_hour: int) -> tuple[datetime, datetime]:
    return (datetime.combine(day, time(start_hour, 0)),
            datetime.combine(day, time(min(end_hour, 23), 59)))


def resolve_time(canonical: str | None, now: datetime | None = None) -> dict:
    """Canonical wording -> {start, end, granularity, label}. Never a model's job."""
    now = now or datetime.now()
    today = now.date()
    canonical = (canonical or "").strip().lower()

    if not canonical:
        end = today + timedelta(days=DEFAULT_HORIZON_DAYS - 1)
        return {"start": datetime.combine(today, time.min), "end": datetime.combine(end, time.max),
                "granularity": "daily", "label": "next few days"}

    if canonical == "now":
        return {"start": now, "end": now + timedelta(hours=1),
                "granularity": "hourly", "label": "now"}

    if "-" in canonical and ":" in canonical:                       # "07:00-11:00"
        start_hour, end_hour = (int(part.split(":")[0]) for part in canonical.split("-"))
        day = today if end_hour >= now.hour else today + timedelta(days=1)
        start, end = _window(day, start_hour, end_hour)
        return {"start": start, "end": end, "granularity": "hourly", "label": canonical}

    if ":" in canonical:                                            # "18:45"
        hour, minute = (int(part) for part in canonical.split(":"))
        moment = datetime.combine(today, time(hour, minute))
        if moment < now:
            moment += timedelta(days=1)                             # the next occurrence
        return {"start": moment, "end": moment + timedelta(hours=1),
                "granularity": "hourly", "label": canonical}

    if canonical in PART_OF_DAY:
        start_hour, end_hour = PART_OF_DAY[canonical]
        day = today + timedelta(days=DAY_OFFSET.get(canonical, 0))
        start, end = _window(day, start_hour, end_hour)
        return {"start": start, "end": end, "granularity": "hourly", "label": canonical}

    if canonical.startswith("next ") and canonical.split()[-1] in {"days", "hours", "minutes", "weeks"}:
        count, unit = int(canonical.split()[1]), canonical.split()[-1]
        if unit == "hours":
            return {"start": now, "end": now + timedelta(hours=count),
                    "granularity": "hourly", "label": canonical}
        if unit == "minutes":
            return {"start": now, "end": now + timedelta(minutes=count),
                    "granularity": "hourly", "label": canonical}
        days = count * 7 if unit == "weeks" else count
        return {"start": datetime.combine(today, time.min),
                "end": datetime.combine(today + timedelta(days=days - 1), time.max),
                "granularity": "daily", "label": canonical}

    if canonical in DAY_OFFSET:
        day = today + timedelta(days=DAY_OFFSET[canonical])
        return {"start": datetime.combine(day, time.min), "end": datetime.combine(day, time.max),
                "granularity": "daily", "label": canonical}

    if canonical in WEEKDAYS:                                       # the next such weekday
        ahead = (WEEKDAYS.index(canonical) - today.weekday()) % 7 or 7
        day = today + timedelta(days=ahead)
        return {"start": datetime.combine(day, time.min), "end": datetime.combine(day, time.max),
                "granularity": "daily", "label": canonical}

    if canonical in {"this week", "next week", "this weekend", "next weekend",
                     "this month", "next month", "last week"}:
        offset = {"this week": 0, "next week": 7, "this weekend": 0, "next weekend": 7,
                  "this month": 0, "next month": 30, "last week": -7}[canonical]
        span = 14 if "month" in canonical else 7
        start_day = today + timedelta(days=offset)
        if "weekend" in canonical:                                  # Saturday to Sunday
            start_day += timedelta(days=(5 - start_day.weekday()) % 7)
            span = 2
        return {"start": datetime.combine(start_day, time.min),
                "end": datetime.combine(start_day + timedelta(days=span - 1), time.max),
                "granularity": "daily", "label": canonical}

    # unrecognised wording: keep it visible in the label rather than silently defaulting
    end = today + timedelta(days=DEFAULT_HORIZON_DAYS - 1)
    return {"start": datetime.combine(today, time.min), "end": datetime.combine(end, time.max),
            "granularity": "daily", "label": canonical}


def plan(state, *, places: list[dict], operation: Operation, aggregation: str,
         now: datetime | None = None) -> ResolvedQuery:
    """Build the canonical query and decide whether it can be answered."""
    window = resolve_time(state.time_normalized, now)
    granularity = "hourly" if (window["granularity"] == "hourly" or aggregation == "TREND") else "daily"

    query = ResolvedQuery(
        weather_intent=state.weather_intent,
        action=state.action,
        aggregation=aggregation,
        places=places,
        start=window["start"].isoformat(timespec="minutes"),
        end=window["end"].isoformat(timespec="minutes"),
        granularity=granularity,
        time_label=window["label"],
        operation=operation,
    )

    if not places:
        query.verdict, query.missing = Verdict.CLARIFY, ["location"]
    elif state.action == "COMPARE" and len(places) < 2:
        query.verdict, query.missing = Verdict.CLARIFY, ["second_location"]
    return query


def demo():
    """Self-check: the calendar arithmetic, on a fixed 'now'."""
    now = datetime(2026, 8, 13, 15, 30)          # a Thursday

    assert resolve_time("tomorrow", now)["start"].date() == date(2026, 8, 14)
    assert resolve_time("day after tomorrow", now)["start"].date() == date(2026, 8, 15)
    assert resolve_time("yesterday", now)["start"].date() == date(2026, 8, 12)

    monday = resolve_time("monday", now)                      # next Monday, never today
    assert monday["start"].date() == date(2026, 8, 17), monday["start"]

    evening = resolve_time("this evening", now)
    assert evening["granularity"] == "hourly" and evening["start"].hour == 17

    clock = resolve_time("18:45", now)                        # later today
    assert clock["start"] == datetime(2026, 8, 13, 18, 45), clock["start"]
    passed = resolve_time("06:00", now)                       # already gone -> tomorrow
    assert passed["start"] == datetime(2026, 8, 14, 6, 0), passed["start"]

    span = resolve_time("next 3 days", now)
    assert span["start"].date() == date(2026, 8, 13) and span["end"].date() == date(2026, 8, 15)

    weekend = resolve_time("this weekend", now)
    assert weekend["start"].date() == date(2026, 8, 15), weekend["start"]   # Saturday

    default = resolve_time(None, now)
    assert default["label"] == "next few days" and default["granularity"] == "daily"
    print("planner demo OK:", {k: str(v) for k, v in resolve_time("tomorrow morning", now).items()})


if __name__ == "__main__":
    demo()
