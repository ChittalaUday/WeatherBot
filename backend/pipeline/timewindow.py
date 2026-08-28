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

from dateutil.relativedelta import relativedelta

from src.dates import MONTHS, YEAR as _YEAR, dates_in, month_days, one_date_in

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

DEFAULT_HORIZON_DAYS = 7          # what "no time mentioned" means

# `MONTHS`, `YEAR`, and reading a written date are all `src.dates` now - one parser for this
# module and for `src/v4/schema.py`, which carried a second copy of the same four regexes.


@dataclass(frozen=True)
class Window:
    """An absolute period, and what to print above the table."""

    start: datetime
    end: datetime
    label: str
    granularity: str = "daily"        # daily | hourly - what the wording implies, not the feed
    # False when the wording was not recognised and these dates are a guess. The dates are
    # still filled in - the default horizon - because every caller expects a window, but a
    # caller that answers from an unrecognised one is answering a question nobody asked. Every
    # `time_resolution` correction in the feedback table is this: "when i asked last 7 days it
    # showed next 7days", "when i asked for history it showed next", "when i asked yesterday
    # it is showing today data". One flag, checked once, and the class of bug closes.
    understood: bool = True

    @property
    def span_days(self) -> int:
        return max((self.end.date() - self.start.date()).days + 1, 1)


def parse_dates(text: str) -> list[date]:
    """Every calendar date in the wording, in the order written, de-duplicated."""
    return dates_in(text)


def parse_date(text: str) -> date | None:
    """The single calendar date in the wording, or None. Used to pick rows by date."""
    return one_date_in((text or "").strip().lower())


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

    # Unrecognised wording. The label used to be the only thing kept - which read as "keep it
    # visible rather than silently defaulting" and was in fact silently defaulting, with the
    # user's own words printed over next week's forecast.
    end = today + timedelta(days=DEFAULT_HORIZON_DAYS - 1)
    return Window(datetime.combine(today, time.min), datetime.combine(end, time.max), canonical,
                  understood=False)


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
    #
    # relativedelta, not a fixed number of days per unit. A month is not 30 days and a year is
    # not 365: "last 12 months" used to start on 2 September for a question asked on 28 August,
    # and "last 24 months" was nine days out. Nobody would notice, which is the problem.
    if (m := re.search(r"(?:last|past)\s+(\d+)\s+(day|week|month|year)s?", text)):
        count, unit = int(m.group(1)), m.group(2)
        return _span(now - relativedelta(**{f"{unit}s": count}), now, f"last {count} {unit}s")

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
        return _span(_day(year, month, 1), _day(year, month, month_days(year, month)),
                     m.group(0))

    # "for all of 2023" / "in 2017" - a bare year
    if (m := re.search(rf"\b({_YEAR})\b", text)):
        year = int(m.group(1))
        return _span(_day(year, 1, 1), _day(year, 12, 31), str(year))

    if "decade" in text:
        return _span(now - timedelta(days=3653), now, "the last decade")

    # "last june" / "in june" - a bare month with no year, meaning the most recent one.
    #
    # Without this the expression fell through to `_relative`, which knows about days and
    # weeks and nothing about months, and came back with the default forward horizon: asked
    # for rainfall last June the answer covered next week, labelled "last june". Silently, and
    # for every bare month there is.
    #
    # The most recent one is the only reading worth having. A June that has not happened is
    # past the ten-day forecast, so it is not answerable in the other direction anyway.
    if (window := _bare_month(text, now)):
        return window

    return _relative(text, now)


# "may" is a month and it is also the commonest modal verb in a weather question. Requiring a
# preposition in front of it costs nothing on the real month ("in may", "last may") and keeps
# "how much rain may we get" from being answered with a month of archive.
_NEEDS_A_PREPOSITION = {"may", "march"}


def _bare_month(text: str, now: datetime) -> Window | None:
    """A month named with no year -> its most recent occurrence. None when none is named."""
    for word in re.findall(r"[a-z]+", text):
        if word not in MONTHS:
            continue
        if word in _NEEDS_A_PREPOSITION and not re.search(
                rf"\b(?:last|in|of|during|for)\s+{word}\b", text):
            continue
        month = MONTHS[word]
        # already been this year, or we mean the one before
        year = now.year if month <= now.month else now.year - 1
        start, end = _day(year, month, 1), _day(year, month, month_days(year, month))
        # asked during the month itself, "june" is June so far - the rest has not happened
        if end > now:
            end = datetime.combine(now.date(), time.max)
        return _span(start, end, f"{word} {year}")
    return None


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
    # Drop rows already in the past - the forecast feed starts a couple of days back, and
    # "tomorrow" must mean tomorrow. Unless the question looks back, in which case those rows
    # ARE the answer.
    #
    # That used to be a two-name exemption list, {"yesterday", "last week"}, and every other
    # way of naming the past fell through it: "last 7 days" arrived with twenty-one rows from
    # the archive and left with one, the single row dated today. The list was approximating a
    # property the resolver can simply be asked for, so it is asked.
    asked_window = resolve(canonical, now) if canonical else None
    looks_back = asked_window is not None and asked_window.start.date() < now.date()
    if not looks_back:
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

    # Unknown to the ladder above - but never unknown to `resolve`, which computes an absolute
    # window for every expression there is, including all the ones with no branch here.
    # Filtering by it is what a month needs: thirty rows of June came back from the archive and
    # this line used to hand back the first seven, so a question about a month was answered
    # from a week, labelled with the month and counted as "across 7 readings".
    #
    # `rows[:7]` stays as the floor. When nothing at all falls inside the window the feed and
    # the question disagree about which dates exist, and that is quality's to report - not a
    # reason to return nothing.
    window = asked_window or resolve(canonical, now)
    inside = [r for r in rows if window.start <= _stamp(r) <= window.end]
    return (inside or rows[:7]), canonical
