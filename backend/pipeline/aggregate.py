"""
One statistic, computed. Twenty-two of them, one small function each.

    python tests/test_pipeline_units.py          # the checks for this module

The model says *which* statistic a question asked for; nothing here decides that, and nothing
here invents a figure. Every function takes the same three things - the rows, the column, the
variable they belong to - and returns the same `Reduction`, so the dispatch table at the
bottom is the whole control flow. There is no branching ladder to read: to find out what
MEDIAN does, read `_median`.

Two rules hold across all of them:

    a statistic the variable cannot answer returns None, never a number. `schema.supports`
    is asked first, so "total humidity" comes back unanswered rather than as the sum of nine
    percentages.

    a missing reading is not a zero. `quality.values` drops sentinels, so -999 can never win
    a MIN or drag down a mean - which it did, and printed "coldest: -999°C" under a summer
    forecast.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime

from backend.pipeline.quality import is_missing
from backend.pipeline.render import label, stamp, unit
from src.v4.schema import COUNTABLE, Aggregation, Variable, column_supports, supports

# 16 compass buckets, so a wind rose has petals a person recognises.
COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")

# The only statistics that survive a circular quantity - see the guard in `compute`.
CIRCULAR_SAFE = frozenset({Aggregation.MODE, Aggregation.DISTRIBUTION, Aggregation.COUNT,
                           Aggregation.RAW})


@dataclass
class Reduction:
    """One answered statistic. `value` is the figure; the rest is what the figure is about."""

    kind: str
    text: str
    unit: str = ""
    value: float | None = None
    at: str = ""                                    # the timestamp a selector picked
    until: str = ""                                 # ...and the far end, for a stretch
    series: list = dataclass_field(default_factory=list)   # CUMULATIVE
    buckets: list = dataclass_field(default_factory=list)  # DISTRIBUTION
    variable: str = ""

    def as_dict(self) -> dict:
        out: dict = {"kind": self.kind, "text": self.text, "unit": self.unit,
                     "variable": self.variable}
        # `value` is tested against None, not for truthiness: 0.0 mm of rain and 0 rainy days
        # are answers, and dropping them left the figure missing from the payload entirely.
        if self.value is not None:
            out["value"] = self.value
        for name in ("at", "until", "series", "buckets"):
            if (got := getattr(self, name)):
                out[name] = got
        return out


# --- reading the column ------------------------------------------------------

def _pairs(rows: list[dict], field: str) -> list[tuple[dict, float]]:
    """Every usable (row, number). The one place a sentinel is dropped, so no statistic below
    has to remember to."""
    return [(r, float(r[field])) for r in rows if not is_missing(r.get(field))]


def _nums(pairs) -> list[float]:
    return [n for _, n in pairs]


def _when(row: dict) -> str:
    """A row's timestamp, with the clock only when the feed carries one."""
    at = stamp(row)
    return f"{at:%d %b %H:%M}" if at.hour else f"{at:%d %b}"


def _runs(pairs, over: float) -> list[list[tuple[dict, float]]]:
    """Consecutive stretches whose reading clears `over`.

    One pass and one list - the runs are built as they are met rather than by scanning for
    starts and then for ends, which is the version that goes wrong on the last row.
    """
    found, current = [], []
    for pair in pairs:
        if pair[1] >= over:
            current.append(pair)
        elif current:
            found.append(current)
            current = []
    return found + ([current] if current else [])


def _condition(variable: Variable) -> tuple[float, str]:
    """The threshold that counts as an occurrence, and the word for one."""
    _, over, word = COUNTABLE[variable]
    return over, word


# --- one number out of the column --------------------------------------------

def _sum(pairs, name, units, variable):
    total = sum(_nums(pairs))
    return Reduction("SUM", f"Total {name}: {total:.1f}{units} across {len(pairs)} readings",
                     units, round(total, 2))


def _avg(pairs, name, units, variable):
    mean = statistics.fmean(_nums(pairs))
    return Reduction("AVG", f"Average {name}: {mean:.1f}{units}", units, round(mean, 2))


def _median(pairs, name, units, variable):
    mid = statistics.median(_nums(pairs))
    return Reduction("MEDIAN", f"Median {name}: {mid:.1f}{units}", units, round(mid, 2))


def _max(pairs, name, units, variable):
    row, top = max(pairs, key=lambda p: p[1])
    return Reduction("MAX", f"Highest {name}: {top:.1f}{units} at {_when(row)}", units,
                     round(top, 2), at=row["Date_time"])


def _min(pairs, name, units, variable):
    row, low = min(pairs, key=lambda p: p[1])
    return Reduction("MIN", f"Lowest {name}: {low:.1f}{units} at {_when(row)}", units,
                     round(low, 2), at=row["Date_time"])


def _range(pairs, name, units, variable):
    numbers = _nums(pairs)
    spread = max(numbers) - min(numbers)
    return Reduction("RANGE",
                     f"{name.capitalize()} spanned {spread:.1f}{units} "
                     f"({min(numbers):.1f} to {max(numbers):.1f}{units})", units,
                     round(spread, 2))


def _stddev(pairs, name, units, variable):
    if len(pairs) < 2:
        return None
    spread = statistics.pstdev(_nums(pairs))
    steady = "steady" if spread < 1 else "variable"
    return Reduction("STDDEV", f"{name.capitalize()} was {steady} - it moved {spread:.1f}"
                               f"{units} either side of the average", units, round(spread, 2))


def _change(pairs, name, units, variable):
    if len(pairs) < 2:
        return None
    moved = pairs[-1][1] - pairs[0][1]
    way = "up" if moved > 0 else "down"
    return Reduction("CHANGE", f"{name.capitalize()} went {way} {abs(moved):.1f}{units} "
                               f"over the period", units, round(moved, 2))


def _count(pairs, name, units, variable):
    over, word = _condition(variable)
    hit = [p for p in pairs if p[1] >= over]
    span = "hours" if stamp(pairs[0][0]).hour or len(pairs) > 24 else "days"
    return Reduction("COUNT", f"{len(hit)} {word} {span} out of {len(pairs)}", "",
                     float(len(hit)))


def _run_count(pairs, name, units, variable):
    over, word = _condition(variable)
    spells = _runs(pairs, over)
    return Reduction("RUN_COUNT", f"{len(spells)} separate {word} "
                                  f"{'spell' if len(spells) == 1 else 'spells'}", "",
                     float(len(spells)))


def _frequency(pairs, name, units, variable):
    over, word = _condition(variable)
    share = len([p for p in pairs if p[1] >= over]) / len(pairs) * 100
    return Reduction("FREQUENCY", f"{word.capitalize()} {share:.0f}% of the time", "%",
                     round(share, 1))


def _cumulative(pairs, name, units, variable):
    running, total = [], 0.0
    for row, number in pairs:
        total += number
        running.append({"t": row["Date_time"], "v": round(total, 2)})
    return Reduction("CUMULATIVE", f"{name.capitalize()} reached {total:.1f}{units} in total",
                     units, round(total, 2), series=running)


def _intensity(pairs, name, units, variable):
    over, word = _condition(variable)
    hit = [p for p in pairs if p[1] >= over]
    if not hit:
        return Reduction("INTENSITY", f"No {word} readings, so there is no intensity to give")
    rate = sum(_nums(hit)) / len(hit)
    return Reduction("INTENSITY", f"{rate:.1f}{units} per {word} reading while it lasted",
                     units, round(rate, 2))


def _bearing(degrees: float) -> str:
    return COMPASS[int((degrees % 360) / 22.5 + 0.5) % 16]


def _mode(pairs, name, units, variable):
    bearings = [_bearing(n) for n in _nums(pairs)]
    top = statistics.mode(bearings)
    share = bearings.count(top) / len(bearings) * 100
    return Reduction("MODE", f"Wind mostly from the {top} ({share:.0f}% of readings)")


def _distribution(pairs, name, units, variable):
    """Every compass point and its share - all sixteen, including the empty ones.

    The empty petals are the shape. Emitting only the directions that occurred left a week of
    steady westerlies as two points, which a rose draws as a line: it is the sixteen-way
    contrast that says "the wind held one way all week" rather than "there were two readings".
    """
    bearings = [_bearing(n) for n in _nums(pairs)]
    counted = [{"bucket": point, "share": round(bearings.count(point) / len(bearings) * 100, 1)}
               for point in COMPASS]
    blowing = [b for b in counted if b["share"]]
    top = max(blowing, key=lambda b: b["share"])
    return Reduction("DISTRIBUTION",
                     f"Wind came from {len(blowing)} of the sixteen directions, most often the "
                     f"{top['bucket']} ({top['share']:.0f}%)", buckets=counted)


# --- a date or a stretch, not a number ---------------------------------------

def _peak_date(pairs, name, units, variable):
    row, top = max(pairs, key=lambda p: p[1])
    return Reduction("PEAK_DATE", f"{_when(row)} was the highest for {name}, at {top:.1f}"
                                  f"{units}", units, round(top, 2), at=row["Date_time"])


def _low_date(pairs, name, units, variable):
    row, low = min(pairs, key=lambda p: p[1])
    return Reduction("LOW_DATE", f"{_when(row)} was the lowest for {name}, at {low:.1f}"
                                 f"{units}", units, round(low, 2), at=row["Date_time"])


def _by_day(pairs) -> list[tuple[datetime, float]]:
    """One total per calendar day. Hourly rows make a "period" mean a day, not a reading."""
    days: dict = {}
    for row, number in pairs:
        days.setdefault(stamp(row).date(), []).append(number)
    return [(day, sum(numbers)) for day, numbers in sorted(days.items())]


def _peak_period(pairs, name, units, variable):
    days = _by_day(pairs)
    day, total = max(days, key=lambda d: d[1])
    return Reduction("PEAK_PERIOD", f"{day:%d %b} was the heaviest day for {name}, "
                                    f"{total:.1f}{units}", units, round(total, 2),
                     at=day.isoformat())


def _low_period(pairs, name, units, variable):
    days = _by_day(pairs)
    day, total = min(days, key=lambda d: d[1])
    return Reduction("LOW_PERIOD", f"{day:%d %b} was the lightest day for {name}, "
                                   f"{total:.1f}{units}", units, round(total, 2),
                     at=day.isoformat())


def _longest_run(pairs, name, units, variable):
    over, word = _condition(variable)
    spells = _runs(pairs, over)
    if not spells:
        return Reduction("LONGEST_RUN", f"No {word} stretch at all over the period")
    longest = max(spells, key=len)
    return Reduction("LONGEST_RUN",
                     f"The longest {word} stretch ran {len(longest)} readings, from "
                     f"{_when(longest[0][0])} to {_when(longest[-1][0])}", "",
                     float(len(longest)), at=longest[0][0]["Date_time"],
                     until=longest[-1][0]["Date_time"])


# The whole control flow. One entry per statistic, and no ladder to read past.
COMPUTE = {
    Aggregation.SUM: _sum, Aggregation.AVG: _avg, Aggregation.MEDIAN: _median,
    Aggregation.MAX: _max, Aggregation.MIN: _min, Aggregation.RANGE: _range,
    Aggregation.STDDEV: _stddev, Aggregation.CHANGE: _change, Aggregation.COUNT: _count,
    Aggregation.RUN_COUNT: _run_count, Aggregation.FREQUENCY: _frequency,
    Aggregation.CUMULATIVE: _cumulative, Aggregation.INTENSITY: _intensity,
    Aggregation.MODE: _mode, Aggregation.DISTRIBUTION: _distribution,
    Aggregation.PEAK_DATE: _peak_date, Aggregation.LOW_DATE: _low_date,
    Aggregation.PEAK_PERIOD: _peak_period, Aggregation.LOW_PERIOD: _low_period,
    Aggregation.LONGEST_RUN: _longest_run,
}


def compute(rows: list[dict], field: str, variable, aggregation) -> dict | None:
    """One statistic over one column, or None when it cannot honestly be answered.

    None has three causes and they are all the same answer to a caller: the statistic makes no
    sense for this variable, the column came back empty, or the rows are too few to say
    anything. TREND and RAW are not here - TREND is prose about the shape of the series and
    lives in `analysis`, RAW means "no reduction at all".
    """
    variable = variable if isinstance(variable, Variable) else Variable(variable)
    aggregation = (aggregation if isinstance(aggregation, Aggregation)
                   else Aggregation(aggregation))
    if aggregation not in COMPUTE or not supports(variable, aggregation):
        return None
    if not column_supports(field, aggregation):
        return None
    # A bearing is circular, so every linear statistic on it is wrong rather than approximate:
    # 350° and 10° are 20° apart and average to 180°, the exact opposite direction. Only the
    # bucketed statistics mean anything here.
    if field == "Wind_Direction" and aggregation not in CIRCULAR_SAFE:
        return None
    pairs = _pairs(rows, field)
    if not pairs:
        return None
    got = COMPUTE[aggregation](pairs, label(field).lower(), unit(field), variable)
    if got is None:
        return None
    got.variable = variable.value
    return got.as_dict()
