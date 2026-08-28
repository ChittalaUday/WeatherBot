"""
Absolute windows, and the rows an expression asks for.

    python tests/test_pipeline_units.py          # the checks for this module

`backend.nlu.duckling` resolves the wording and hands back **absolute dates**, so the calendar
that used to live here is gone - it was a second implementation of arithmetic Duckling already
does, written twice over (once in `resolve`, once again in `select_rows`).

What is left is only what Duckling cannot do, measured against this deployment's vocabulary:

    forms it does not place      "early morning" - the one part of day it has no rule for
    ranges it truncates          "11 jan 2026 and 17 jan 2026" comes back as 11 Jan alone
    "for all of 2023"            no reading at all; a bare year on its own is fine
    readings that are wrong      "in june" -> June 2027. Correct English, and past the
                                 ten-day forecast, so the most recent June is the only
                                 answerable reading
    the product's own default    no period named -> the next 7 days

Everything else arrives here already absolute - "2026-08-30", "2026-08-30 to 2026-09-01",
"07:00-11:00" - and needs a parser, not a calendar.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from src.dates import MONTHS, dates_in, month_days, one_date_in
from src.dates import YEAR as _YEAR

DEFAULT_HORIZON_DAYS = 7          # what "no time mentioned" means

# The parts of day Duckling has no rule for. It places "this morning", "tonight" and the rest;
# this is the measured remainder, so it is a gap list rather than a table.
EDGE_PARTS = {"early morning": (4, 7)}


@dataclass(frozen=True)
class Window:
    """An absolute period, and what to print above the table."""

    start: datetime
    end: datetime
    label: str
    granularity: str = "daily"        # daily | hourly - what the wording implies, not the feed
    # False when the wording was not recognised and these dates are the default horizon
    # rather than an answer. Every `time_resolution` correction in the feedback table is a
    # caller that answered from one of these without checking.
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


def _label(start: date, end: date) -> str:
    """What goes above the table. The dates are absolute now, so the label is built from them
    rather than echoing a canonical string nobody typed."""
    if start == end:
        return f"{start:%d %b %Y}"
    return f"{start:%d %b} to {end:%d %b %Y}"


def _horizon(now: datetime, label: str = "next few days", understood: bool = True) -> Window:
    end = now.date() + timedelta(days=DEFAULT_HORIZON_DAYS - 1)
    return Window(datetime.combine(now.date(), time.min), datetime.combine(end, time.max),
                  label, understood=understood)


def resolve(canonical: str | None, now: datetime | None = None) -> Window:
    """An absolute form -> a window. Anything relative was resolved by Duckling upstream."""
    now = now or datetime.now()
    text = (canonical or "").strip().lower()

    if not text:
        return _horizon(now)

    # "2026-08-14T06:00 to 2026-08-14T11:59" - an absolute span that carries its hours
    if (m := re.match(r"([\d-]{10}t[\d:]{5})\s+to\s+([\d-]{10}t[\d:]{5})$", text)):
        start, end = (datetime.fromisoformat(g.upper()) for g in m.groups())
        return Window(start, end, f"{start:%d %b %H:%M} to {end:%H:%M}", "hourly")

    # "07:00-11:00" - a clock range, from Duckling or from the tables
    if "-" in text and ":" in text:
        start_hour, end_hour = (int(part.split(":")[0]) for part in text.split("-"))
        day = now.date() if end_hour >= now.hour else now.date() + timedelta(days=1)
        return Window(datetime.combine(day, time(start_hour, 0)),
                      datetime.combine(day, time(min(end_hour, 23), 59)), text, "hourly")

    # "18:45" - a clock point. `resolve` rolls it to the next occurrence if it has passed.
    if ":" in text and not re.search(r"[a-z]", text):
        hour, minute = (int(part) for part in text.split(":"))
        moment = datetime.combine(now.date(), time(hour, minute))
        if moment < now:
            moment += timedelta(days=1)
        return Window(moment, moment + timedelta(hours=1), text, "hourly")

    # The product's own default vocabulary, not a user's words: `advice.DEFAULT_WINDOW` writes
    # "today" and "next 3 days" for an advisory turn that named no period, and those strings
    # never reach Duckling because nobody typed them. Two forms, and no more.
    if text == "today":
        return _span(datetime.combine(now.date(), time.min),
                     datetime.combine(now.date(), time.min), "today")
    if (m := re.fullmatch(r"next (\d+) days", text)):
        end = now.date() + timedelta(days=int(m.group(1)) - 1)
        return _span(datetime.combine(now.date(), time.min), datetime.combine(end, time.min),
                     text)

    if text in EDGE_PARTS:
        start_hour, end_hour = EDGE_PARTS[text]
        return Window(datetime.combine(now.date(), time(start_hour, 0)),
                      datetime.combine(now.date(), time(end_hour, 59)), text, "hourly")

    # Two dates is a range. Duckling returns only the first of "11 jan 2026 and 17 jan 2026",
    # so this branch is a gap it leaves rather than a duplicate of it.
    found = parse_dates(text)
    if len(found) >= 2:
        first, last = min(found), max(found)
        return _span(datetime.combine(first, time.min), datetime.combine(last, time.min),
                     _label(first, last))
    if len(found) == 1:
        only = datetime.combine(found[0], time.min)
        return _span(only, only, _label(found[0], found[0]))

    # "from 2010 to 2025" - Duckling reads the first year and leaves the `to` open, so the
    # range comes back running to today instead of to the end of 2025.
    if (m := re.search(rf"({_YEAR})\s*(?:to|-|and|until)\s*({_YEAR})", text)):
        a, b = sorted((int(m.group(1)), int(m.group(2))))
        return _span(datetime(a, 1, 1), datetime(b, 12, 31), f"{a}-{b}")

    # "march 2022" - a whole month. Kept ahead of the bare year below, which would otherwise
    # read "august 2019" as the whole of 2019.
    if (m := re.search(rf"([a-z]+)\s+({_YEAR})", text)) and m.group(1) in MONTHS:
        year, month = int(m.group(2)), MONTHS[m.group(1)]
        return _span(datetime(year, month, 1),
                     datetime(year, month, month_days(year, month)), m.group(0))

    # "the last decade" - Duckling has no rule for it either.
    if "decade" in text:
        return _span(now - timedelta(days=3653), now, "the last decade")

    # "for all of 2023" - Duckling reads nothing at all from this one.
    if (m := re.search(rf"\b({_YEAR})\b", text)):
        year = int(m.group(1))
        return _span(datetime(year, 1, 1), datetime(year, 12, 31), str(year))

    # "in june" - Duckling answers with next June, which is past the ten-day forecast.
    if (window := _bare_month(text, now)):
        return window

    return _horizon(now, text, understood=False)


# "may" is a month and the commonest modal in a weather question. Requiring a preposition
# costs nothing on the real month and keeps "how much rain may we get" out of the archive.
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
        start, end = datetime(year, month, 1), datetime(year, month, month_days(year, month))
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
    """The rows the expression asks for, plus the label that goes above the table.

    One filter over `resolve`'s window. This used to be a second ladder of special cases -
    part of day, weekday, named range, next-N - each one a copy of the branch in `resolve`
    that had already computed the same window, and each one a place for the two to disagree.
    """
    if not rows:
        return [], "no data"
    # Real wall-clock, not rows[0]: the feed can start days in the past, and "tomorrow" must
    # not mean "the day after whatever the API sent first".
    now = now or datetime.now()
    hourly = any(_stamp(r).hour for r in rows[:24])
    window = resolve(canonical, now) if canonical else None

    if window is None or not window.understood:
        return (rows[:12], "next few hours") if hourly else (rows[:7], "next few days")

    # Drop rows already in the past, unless the question looks back - in which case they ARE
    # the answer. The resolver knows which it is, so it is asked rather than guessed from a
    # list of names that could never be complete.
    if window.start.date() >= now.date():
        rows = [r for r in rows if _stamp(r).date() >= now.date()] or rows

    inside = [r for r in rows if window.start <= _stamp(r) <= window.end]
    # An exact date that the feed does not hold stays empty: in range but absent is not the
    # same as unasked, and quality reports NO_DATA for it. Everything else floors at rows[:7],
    # because a window with nothing in it means the feed and the question disagree about which
    # dates exist - which is quality's to report, not a reason to return nothing.
    if not inside and window.span_days == 1 and parse_date(canonical):
        return [], window.label
    return (inside or rows[:7]), window.label
