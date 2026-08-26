"""
The only place that knows what a time expression means.

    python tests/test_pipeline_units.py          # the checks for this module

Three jobs that used to live in three modules with three copies of the calendar:

    resolve()      canonical wording  ->  an absolute (start, end) window
    parse_date()   "11 august 2026"   ->  a date
    select_rows()  a feed's rows      ->  the ones that expression asked for

Splitting them is how "11 jun 2026" resolved to the whole of 2026 in the planner while the
row selector read it correctly, and how `tomorrow` had three different offset tables. One
calendar, one set of tables, one bug surface.

Temporal resolution is rules, not a model: "day after tomorrow" is arithmetic, and a
classifier that got it wrong 2% of the time would be strictly worse than a calendar. The
model's only temporal job is to spot the span and canonicalise its wording (Rule 4.3).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

# --- the calendar, once ------------------------------------------------------
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")

# canonical wording -> the hours of the day it covers
PART_OF_DAY = {
    "early morning": (4, 7), "this morning": (6, 11), "tomorrow morning": (6, 11),
    "this afternoon": (12, 16), "tomorrow afternoon": (12, 16),
    "this evening": (17, 21), "tomorrow evening": (17, 21),
    "tonight": (19, 23), "tomorrow night": (19, 23), "midnight": (0, 1),
}
# canonical wording -> days from today
DAY_OFFSET = {
    "now": 0, "today": 0, "tonight": 0, "this morning": 0, "this afternoon": 0,
    "this evening": 0, "early morning": 0, "midnight": 0, "yesterday": -1,
    "tomorrow": 1, "tomorrow morning": 1, "tomorrow afternoon": 1, "tomorrow evening": 1,
    "tomorrow night": 1, "day after tomorrow": 2,
}
# named ranges -> (offset in days from today, span in days)
NAMED_RANGE = {
    "this week": (0, 7), "next week": (7, 7), "last week": (-7, 7),
    "this month": (0, 14), "next month": (30, 14),
    "this weekend": (0, 2), "next weekend": (7, 2),
}

_MONTH_NAMES = ("january", "february", "march", "april", "may", "june", "july", "august",
                "september", "october", "november", "december")
# Full names AND the abbreviations people type. Without the short forms "11 jun 2026" matched
# no date branch and fell through to the bare-year rule - every abbreviated date became the
# whole of that year, which is why "11 jun" and "11 aug" both reported 225 days back.
MONTHS = {name: i for i, name in enumerate(_MONTH_NAMES, start=1)}
MONTHS.update({name[:3]: i for i, name in enumerate(_MONTH_NAMES, start=1)})
MONTHS["sept"] = 9

DEFAULT_HORIZON_DAYS = 7          # what "no time mentioned" means

_YEAR = r"(?:19|20)\d{2}"
# every shape a calendar date arrives in, with the meaning of each capture group
_DATE_PATTERNS = (
    (rf"\b({_YEAR})-(\d{{1,2}})-(\d{{1,2}})\b", ("y", "m", "d")),
    (rf"\b(\d{{1,2}})[/-](\d{{1,2}})[/-]({_YEAR})\b", ("d", "m", "y")),
    (rf"\b(\d{{1,2}})\s+([a-z]+)\.?\s+({_YEAR})\b", ("d", "M", "y")),
    (rf"\b([a-z]+)\.?\s+(\d{{1,2}}),?\s+({_YEAR})\b", ("M", "d", "y")),
)


@dataclass(frozen=True)
class Window:
    """An absolute period, and what to print above the table."""

    start: datetime
    end: datetime
    label: str
    granularity: str = "daily"        # daily | hourly - what the wording implies, not the feed

    @property
    def span_days(self) -> int:
        return max((self.end.date() - self.start.date()).days + 1, 1)


def parse_dates(text: str) -> list[date]:
    """Every calendar date in the wording, in the order written, de-duplicated."""
    found, seen, out = [], set(), []
    for pattern, order in _DATE_PATTERNS:
        for match in re.finditer(pattern, text):
            parts = dict(zip(order, match.groups()))
            month = parts.get("m") or MONTHS.get(str(parts.get("M", "")).rstrip("."))
            if month is None:
                continue
            try:
                found.append((match.start(), date(int(parts["y"]), int(month), int(parts["d"]))))
            except (ValueError, TypeError):
                continue
    for _, when in sorted(found):
        if when not in seen:
            seen.add(when)
            out.append(when)
    return out


def parse_date(text: str) -> date | None:
    """The single calendar date in the wording, or None. Used to pick rows by date."""
    found = parse_dates((text or "").strip().lower())
    return found[0] if len(found) == 1 else None


def _span(start: datetime, end: datetime, label: str, granularity: str = "daily") -> Window:
    return Window(start, end.replace(hour=23, minute=59), label, granularity)


def _day(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day)


def _relative(canonical: str, now: datetime) -> Window:
    """Wording relative to `now` - the common case, and the only one with a granularity."""
    today = now.date()
    whole_day = lambda d, label: Window(datetime.combine(d, time.min),
                                        datetime.combine(d, time.max), label)

    if canonical == "now":
        return Window(now, now + timedelta(hours=1), "now", "hourly")

    if "-" in canonical and ":" in canonical:                    # "07:00-11:00"
        start_hour, end_hour = (int(part.split(":")[0]) for part in canonical.split("-"))
        day = today if end_hour >= now.hour else today + timedelta(days=1)
        return Window(datetime.combine(day, time(start_hour, 0)),
                      datetime.combine(day, time(min(end_hour, 23), 59)), canonical, "hourly")

    if ":" in canonical:                                         # "18:45"
        hour, minute = (int(part) for part in canonical.split(":"))
        moment = datetime.combine(today, time(hour, minute))
        if moment < now:
            moment += timedelta(days=1)                          # the next occurrence
        return Window(moment, moment + timedelta(hours=1), canonical, "hourly")

    if canonical in PART_OF_DAY:
        start_hour, end_hour = PART_OF_DAY[canonical]
        day = today + timedelta(days=DAY_OFFSET.get(canonical, 0))
        return Window(datetime.combine(day, time(start_hour, 0)),
                      datetime.combine(day, time(min(end_hour, 23), 59)), canonical, "hourly")

    if canonical.startswith("next ") and canonical.split()[-1] in {"days", "hours", "minutes",
                                                                   "weeks"}:
        count, unit = int(canonical.split()[1]), canonical.split()[-1]
        if unit == "hours":
            return Window(now, now + timedelta(hours=count), canonical, "hourly")
        if unit == "minutes":
            return Window(now, now + timedelta(minutes=count), canonical, "hourly")
        days = count * 7 if unit == "weeks" else count
        return Window(datetime.combine(today, time.min),
                      datetime.combine(today + timedelta(days=days - 1), time.max), canonical)

    if canonical in DAY_OFFSET:
        return whole_day(today + timedelta(days=DAY_OFFSET[canonical]), canonical)

    if canonical in WEEKDAYS:                                    # the next such weekday
        ahead = (WEEKDAYS.index(canonical) - today.weekday()) % 7 or 7
        return whole_day(today + timedelta(days=ahead), canonical)

    if canonical in NAMED_RANGE:
        offset, span = NAMED_RANGE[canonical]
        start_day = today + timedelta(days=offset)
        if "weekend" in canonical:                               # Saturday to Sunday
            start_day += timedelta(days=(5 - start_day.weekday()) % 7)
        return Window(datetime.combine(start_day, time.min),
                      datetime.combine(start_day + timedelta(days=span - 1), time.max), canonical)

    # unrecognised wording: keep it visible in the label rather than silently defaulting
    end = today + timedelta(days=DEFAULT_HORIZON_DAYS - 1)
    return Window(datetime.combine(today, time.min), datetime.combine(end, time.max), canonical)


def resolve(canonical: str | None, now: datetime | None = None) -> Window:
    """Canonical wording -> an absolute window. Handles both relative and calendar wording."""
    now = now or datetime.now()
    text = (canonical or "").strip().lower()

    if not text:
        end = now.date() + timedelta(days=DEFAULT_HORIZON_DAYS - 1)
        return Window(datetime.combine(now.date(), time.min), datetime.combine(end, time.max),
                      "next few days")

    # "from 2010 to 2025" / "between 2015 and 2020"
    if (m := re.search(rf"({_YEAR})\s*(?:to|-|and|until)\s*({_YEAR})", text)):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        return _span(_day(a, 1, 1), _day(b, 12, 31), f"{a}-{b}")

    # "over the last 5 years" / "past 6 months" / "the last 90 days"
    if (m := re.search(r"(?:last|past)\s+(\d+)\s+(day|week|month|year)s?", text)):
        count, unit = int(m.group(1)), m.group(2)
        days = {"day": 1, "week": 7, "month": 30, "year": 365}[unit] * count
        return _span(now - timedelta(days=days), now, f"last {count} {unit}s")

    # "each year since 2018"
    if (m := re.search(rf"since\s+({_YEAR})", text)):
        return _span(_day(int(m.group(1)), 1, 1), now, f"since {m.group(1)}")

    # Every calendar date in the wording, in order. Two of them is a range - "11 jan 2026 and
    # 17 jan 2026" used to match the first date branch and answer for 11 Jan alone, dropping
    # the six days that were actually asked about.
    found = parse_dates(text)
    if len(found) >= 2:
        first, last = min(found), max(found)
        return _span(datetime.combine(first, time.min), datetime.combine(last, time.min),
                     f"{first:%d %b %Y} to {last:%d %b %Y}")
    if len(found) == 1:
        only = datetime.combine(found[0], time.min)
        return _span(only, only, f"{found[0]:%d %b %Y}")

    # "march 2022" - a whole month
    if (m := re.search(rf"([a-z]+)\s+({_YEAR})", text)) and m.group(1) in MONTHS:
        year, month = int(m.group(2)), MONTHS[m.group(1)]
        last = (_day(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)).day
        return _span(_day(year, month, 1), _day(year, month, last), m.group(0))

    # "for all of 2023" / "in 2017" - a bare year
    if (m := re.search(rf"\b({_YEAR})\b", text)):
        year = int(m.group(1))
        return _span(_day(year, 1, 1), _day(year, 12, 31), str(year))

    if "decade" in text:
        return _span(now - timedelta(days=3653), now, "the last decade")

    return _relative(text, now)


# --- picking the rows an expression asked for --------------------------------

def _stamp(row: dict) -> datetime:
    return datetime.fromisoformat(row["Date_time"])


def select_rows(rows: list[dict], canonical: str,
                now: datetime | None = None) -> tuple[list[dict], str]:
    """The rows the expression asks for, plus a human label for the range.

    A feed returns what it returns - the forecast starts a couple of days in the past and the
    archive returns exactly the window requested. This narrows it to what was asked, and the
    label it returns is what goes above the table.
    """
    if not rows:
        return [], "no data"
    # Real wall-clock, not rows[0]: the feed can start days in the past, and "tomorrow" must
    # mean tomorrow rather than "the day after whatever the API happened to send first".
    now = now or datetime.now()
    hourly = any(_stamp(r).hour for r in rows[:24])
    if canonical not in {"yesterday", "last week"}:
        rows = [r for r in rows if _stamp(r).date() >= now.date()] or rows

    if not canonical:                                     # no time mentioned -> the near term
        return (rows[:12], "next few hours") if hourly else (rows[:7], "next few days")

    if canonical == "now":
        return ([r for r in rows if _stamp(r) >= now] or rows)[:1], "now"

    if "-" in canonical and ":" in canonical:             # "07:00-11:00"
        start, end = (int(part.split(":")[0]) for part in canonical.split("-"))
        picked = [r for r in rows if start <= _stamp(r).hour <= end]
        return picked or rows[:1], canonical

    if ":" in canonical:                                  # "18:45" -> from that hour onwards
        hour = int(canonical.split(":")[0])
        at_hour = [r for r in rows if _stamp(r).hour >= hour]
        return ([r for r in at_hour if _stamp(r) >= now] or at_hour or rows)[:3], canonical

    if canonical in PART_OF_DAY:
        start, end = PART_OF_DAY[canonical]
        target = (now + timedelta(days=DAY_OFFSET.get(canonical, 0))).date()
        picked = [r for r in rows
                  if _stamp(r).date() == target and start <= _stamp(r).hour <= end]
        return picked or rows[:6], canonical

    if canonical.startswith("next ") and canonical.split()[-1] in {"days", "hours", "minutes",
                                                                   "weeks"}:
        count, unit = int(canonical.split()[1]), canonical.split()[-1]
        take = {"days": count, "weeks": count * 7, "hours": count}.get(unit, 1)
        return rows[:take], canonical

    if canonical in DAY_OFFSET:
        target = (now + timedelta(days=DAY_OFFSET[canonical])).date()
        return ([r for r in rows if _stamp(r).date() == target] or rows[:1]), canonical

    if canonical in WEEKDAYS:
        wanted = WEEKDAYS.index(canonical)
        picked = [r for r in rows if _stamp(r).weekday() == wanted]
        return picked[:24] or rows[:1], canonical

    if canonical in {"this weekend", "next weekend"}:
        return [r for r in rows if _stamp(r).weekday() >= 5] or rows[:7], canonical
    if canonical in NAMED_RANGE:
        offset, span = NAMED_RANGE[canonical]
        start = max(offset, 0)
        return rows[start:start + span], canonical

    # An explicit calendar date. Without this branch a dated question fell through to "show
    # the horizon" and answered with seven days when exactly one was asked for.
    if (target := parse_date(canonical)):
        picked = [r for r in rows if _stamp(r).date() == target]
        # in range but absent is not the same as unasked: quality reports NO_DATA for it
        return picked, canonical

    return rows[:7], canonical                            # unknown expression: show the horizon
