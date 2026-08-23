"""
When, not whether: the stretches of a period during which a condition actually holds.

    python -m backend.pipeline.windows        # self-check

Rain does not fall for a whole day. It falls between two and four in the afternoon, and the
question "can I dry the clothes today" is answered by the seven hours before it, not by the
day's total. Every activity rule used to reduce the period to one accumulated number and
threshold that, which produced three separate wrongs:

  - **A total says nothing about timing.** 8mm in one afternoon storm and 8mm of all-day
    drizzle are the same number and opposite answers.
  - **A total only grows.** The same weather scored worse the longer a period you asked about,
    so "should I spray today" and "should I spray this week" disagreed about today.
  - **A total cannot suggest anything.** The best it can say is no. It cannot say "not at two,
    but the morning is clear", which is the answer the person actually wanted.

So a rule states what one reading has to look like, and this finds the runs of readings that
look like it. A run long enough to do the thing in is a yes with a time attached; runs that
exist but are all too short are a caution - the rain is broken up but never long enough to
work in; no run at all is a no.

Works on hourly and daily rows alike. The spacing is measured from the timestamps rather than
assumed, so a daily feed gives 24-hour readings and the same rule reads both - it just cannot
say "from 14:00" about a day it only has one number for.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median

from backend.pipeline.quality import is_missing

DAY_HOURS = 24.0
# What a feed is assumed to cover per row when there is only one of them and no second
# timestamp to measure against. A lone daily row is a day; a lone hourly row is an hour - and
# there is no way to tell which from one row, so the caller's `hourly` flag decides.
DEFAULT_SPACING = {True: 1.0, False: DAY_HOURS}


@dataclass(frozen=True)
class Window:
    """A stretch of the period during which the condition held."""

    start: datetime
    end: datetime                # exclusive: the moment the last qualifying reading stops
    readings: int
    hourly: bool

    @property
    def hours(self) -> float:
        return (self.end - self.start).total_seconds() / 3600.0

    @property
    def days(self) -> float:
        return self.hours / DAY_HOURS

    def label(self) -> str:
        """How a person would say this stretch out loud."""
        if not self.hourly:
            if self.days <= 1:
                return f"{self.start:%-d %b}"
            # `end` is exclusive, so the last day it covers is the moment before it
            return f"{self.start:%-d %b} to {self.end - timedelta(seconds=1):%-d %b}"
        # `end` is exclusive, so a window running to the end of the day lands on 00:00 of the
        # next one. "10:00 to 00:00" reads as a mistake; say what a person would say.
        finish = "midnight" if self.end.hour == 0 and self.end.minute == 0 else f"{self.end:%H:%M}"
        if self.start.date() == (self.end - timedelta(seconds=1)).date():
            return f"{self.start:%H:%M} to {finish}"
        return f"{self.start:%-d %b %H:%M} to {self.end:%-d %b} {finish}"

    def describe(self) -> str:
        """The stretch and how long it is - "09:00 to 15:00 (6 hours)"."""
        if self.hourly:
            hours = round(self.hours)
            return f"{self.label()} ({hours} hour{'s' if hours != 1 else ''})"
        days = round(self.days)
        return f"{self.label()} ({days} day{'s' if days != 1 else ''})"


# --- conditions --------------------------------------------------------------
# A rule says what one reading has to look like, in these terms. Built here rather than
# written as lambdas at each rule so that one guarantee holds everywhere: a reading whose
# value is missing, blank or a sentinel is NOT suitable. An unknown hour in the middle of a
# spraying window is exactly the hour you would rather not bet on, and `row.get(f) or 0`
# quietly calls it zero - which for rainfall means "dry" and for wind means "dead calm".


def reading(row: dict, field: str) -> float | None:
    """One field of one row as a number, or None when it is missing in any of its guises."""
    value = row.get(field)
    return None if is_missing(value) else float(value)


def below(field: str, limit: float):
    """True when this reading has `field` and it is under `limit`."""
    def check(row):
        value = reading(row, field)
        return value is not None and value < limit
    return check


def above(field: str, limit: float):
    def check(row):
        value = reading(row, field)
        return value is not None and value > limit
    return check


def between(field: str, low: float, high: float):
    def check(row):
        value = reading(row, field)
        return value is not None and low <= value <= high
    return check


def every(*conditions):
    """True when every condition holds - the usual shape of an activity's requirement."""
    return lambda row: all(condition(row) for condition in conditions)


def _stamp(row: dict) -> datetime | None:
    try:
        return datetime.fromisoformat(str(row["Date_time"]))
    except (KeyError, ValueError, TypeError):
        return None


def spacing_hours(rows: list[dict], hourly: bool | None = None) -> float:
    """How much time one reading stands for, measured from the rows themselves.

    The median gap, not the mean: a feed that skips a few hours in the middle should not have
    every row inflated by the hole. Falls back to the caller's granularity when there is only
    one row to go on, because one timestamp cannot tell you what it covers.
    """
    stamps = [s for s in (_stamp(r) for r in rows or []) if s]
    if len(stamps) < 2:
        return DEFAULT_SPACING[bool(hourly)]
    gaps = [(b - a).total_seconds() / 3600.0 for a, b in zip(stamps, stamps[1:])
            if (b - a).total_seconds() > 0]
    return median(gaps) if gaps else DEFAULT_SPACING[bool(hourly)]


def runs(rows: list[dict], suitable, *, hourly: bool | None = None) -> list[Window]:
    """Every contiguous stretch of readings for which `suitable(row)` is true.

    A reading whose fields are missing is not suitable - it is unknown, and an unknown hour in
    the middle of a spraying window is exactly the hour you would rather not bet on. That
    breaks the run, which is the conservative reading and the intended one.
    """
    spacing = spacing_hours(rows, hourly)
    is_hourly = spacing < DAY_HOURS if hourly is None else bool(hourly)
    found, current = [], []

    def close():
        if current:
            found.append(Window(current[0], current[-1] + timedelta(hours=spacing),
                                len(current), is_hourly))
            current.clear()

    for row in rows or []:
        stamp = _stamp(row)
        keep = False
        if stamp is not None:
            try:
                keep = bool(suitable(row))
            except (TypeError, ValueError):     # a rule reading a column that is not there
                keep = False
        if keep:
            current.append(stamp)
        else:
            close()
    close()
    return found


def longest(windows: list[Window]) -> Window | None:
    return max(windows, key=lambda w: w.hours) if windows else None


def coverage(windows: list[Window], rows: list[dict], hourly: bool | None = None) -> float:
    """The share of the period the condition held for, 0.0 to 1.0."""
    spacing = spacing_hours(rows, hourly)
    total = spacing * len(rows or [])
    return round(sum(w.hours for w in windows) / total, 2) if total else 0.0


def fragmented(windows: list[Window], rows: list[dict], needed_hours: float,
               hourly: bool | None = None) -> bool:
    """True when the condition holds often but never for long enough to be usable.

    This is the case the accumulated total could not express at all. Six one-hour breaks in
    the rain are not a spraying window however much they add up to, and "3mm expected" says
    nothing about which of those two days you are looking at.
    """
    if not windows:
        return False
    best = longest(windows)
    return best.hours < needed_hours and coverage(windows, rows, hourly) >= 0.25


def demo():
    """Self-check: runs, spacing, labels, and the fragmented case the totals could not see."""
    def hours(pattern, start=6):
        """One row per hour; `pattern` is the rainfall in each."""
        return [{"Date_time": f"2026-08-18T{start + i:02d}:00:00", "Rainfall": v}
                for i, v in enumerate(pattern)]

    dry = below("Rainfall", 1.0)

    # A day that rains only in the afternoon has a usable morning. This is the whole point:
    # the total is 12mm either way, and only one of these two is a wet morning.
    afternoon = hours([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 3.0])       # 06:00, rain from 12
    found = runs(afternoon, dry, hourly=True)
    assert len(found) == 1, found
    assert found[0].hours == 6.0, found[0].hours
    assert found[0].label() == "06:00 to 12:00", found[0].label()
    # a window that runs to the end of the day says midnight, not 00:00
    to_midnight = runs(hours([0.0] * 4, start=20), dry, hourly=True)
    assert to_midnight[0].label() == "20:00 to midnight", to_midnight[0].label()
    assert found[0].describe() == "06:00 to 12:00 (6 hours)", found[0].describe()

    # ...and the same 12mm spread through the day leaves nothing usable
    scattered = hours([2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0])
    broken = runs(scattered, dry, hourly=True)
    assert len(broken) == 4 and all(w.hours == 1.0 for w in broken), broken
    assert fragmented(broken, scattered, needed_hours=4, hourly=True), "4 breaks, none usable"
    assert not fragmented(runs(afternoon, dry, hourly=True), afternoon, 4, hourly=True)

    # a period with no rain at all is one window covering all of it
    clear = hours([0.0] * 8)
    whole = runs(clear, dry, hourly=True)
    assert len(whole) == 1 and whole[0].hours == 8.0, whole
    assert coverage(whole, clear, hourly=True) == 1.0

    # ...and one that rains throughout has none
    assert runs(hours([3.0] * 6), dry, hourly=True) == []
    assert longest([]) is None
    assert not fragmented([], hours([3.0] * 6), 4, hourly=True), "no window is not fragmented"

    # A missing reading breaks the run rather than being assumed dry. This is why the
    # conditions are built here: `row.get("Rainfall") or 0` reads None as zero, which for
    # rainfall means "dry" and would have quietly bridged the gap.
    gappy = hours([0.0, 0.0, None, 0.0, 0.0])
    assert [w.hours for w in runs(gappy, dry, hourly=True)] == [2.0, 2.0], runs(gappy, dry, hourly=True)
    assert [w.hours for w in runs(hours([0.0, -999, 0.0]), dry, hourly=True)] == [1.0, 1.0]
    assert reading({"Rainfall": "3.5"}, "Rainfall") == 3.5      # strings that are numbers count
    assert reading({"Rainfall": -999}, "Rainfall") is None      # sentinels do not

    # conditions compose the way the rules need them to
    calm_and_dry = every(below("Rainfall", 0.2), between("Wind_Speed", 1.0, 4.5))
    assert calm_and_dry({"Rainfall": 0.0, "Wind_Speed": 3.0})
    assert not calm_and_dry({"Rainfall": 0.0, "Wind_Speed": 6.0})
    assert not calm_and_dry({"Rainfall": 0.0})                  # no wind reading, no window

    # daily rows: same rule, 24-hour readings, and a label that does not invent a clock time
    daily = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": v}
             for d, v in enumerate([0.0, 0.0, 0.0, 8.0, 0.0], start=18)]
    spans = runs(daily, dry)
    assert spacing_hours(daily) == 24.0, spacing_hours(daily)
    assert [w.days for w in spans] == [3.0, 1.0], [w.days for w in spans]
    assert spans[0].label() == "18 Aug to 20 Aug", spans[0].label()
    assert spans[1].label() == "22 Aug", spans[1].label()
    assert spans[0].describe() == "18 Aug to 20 Aug (3 days)"

    # one row is whatever the caller says it is - a timestamp alone cannot tell you
    assert spacing_hours([{"Date_time": "2026-08-18T06:00:00"}], hourly=True) == 1.0
    assert spacing_hours([], hourly=False) == 24.0
    assert runs([], dry) == []

    print("windows demo OK")
    for name, rows in (("rain from noon", afternoon), ("showers all day", scattered),
                       ("clear", clear), ("3 dry days then rain", daily)):
        got = runs(rows, dry, hourly=rows is not daily)
        best = longest(got)
        print(f"  {name:22s} {len(got)} window(s), best {best.describe() if best else 'none':28s} "
              f"cover {coverage(got, rows, hourly=rows is not daily):.0%}")


if __name__ == "__main__":
    demo()
