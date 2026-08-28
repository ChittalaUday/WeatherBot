"""Calendar dates out of free text. One parser, shared by the model layer and the pipeline.

    python -m src.dates          # the checks for this module

Three modules used to carry their own copy of this: `backend/pipeline/timewindow.py`,
`src/v4/schema.py`, and (before it was merged) the planner. Each had the same four regexes with
their own capture-group order, its own month-name table with its own `["sept"] = 9` line, and
its own component assembly. Three copies of a calendar is three places for a date to be read
differently, which is the bug the timewindow docstring was written about.

`dateutil` does the format work now. It is not a new dependency - pandas already requires it,
and it is pinned in requirements.txt so that stays true on purpose rather than by luck.

What is deliberately NOT here: relative wording. "tomorrow", "last 3 days", "next week" stay in
`backend.pipeline.timewindow`, resolved by tables and arithmetic. A natural-language date
library will parse those too, and that is exactly the problem - it will also parse things that
are not dates, confidently, and a wrong window returns real numbers for the wrong days under a
correct-looking label. This module only reads dates that are *written as dates*.

    dates_in("11 jan 2026 and 17 jan 2026")  ->  [date(2026, 1, 11), date(2026, 1, 17)]
    dates_in("11 jan")                       ->  []            no year - a bare month, not a date
    dates_in("32 jan 2026")                  ->  []            date-shaped, not a date
    month_days(2024, 2)                      ->  29
"""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime

from dateutil.parser import ParserError
from dateutil.parser import parse as _parse

YEAR = r"(?:19|20)\d{2}"

# Month name -> number, from the stdlib rather than a typed-out tuple of twelve strings.
MONTHS = {name.lower(): number for number, name in enumerate(calendar.month_name) if name}
MONTHS.update({name.lower(): number
               for number, name in enumerate(calendar.month_abbr) if name})
# `calendar.month_abbr` says "Sep"; people write "sept". dateutil already knows this form, so
# the entry is only here for callers matching a bare month name against MONTHS.
MONTHS["sept"] = 9

# Full names only, in calendar order - for callers building a regex alternation out of them.
MONTH_NAMES = tuple(name.lower() for name in calendar.month_name if name)

# A run of text shaped like a date AND carrying a four-digit year. One alternative per written
# form, no capture groups - dateutil reads the components, so nothing here has to know which
# position means what.
#
# The year is required on purpose. "11 jan" with no year is a bare month for the caller to read
# as the most recent one; parsed as a date it would land in whatever year the parser defaulted
# to, which is how a question about last June gets answered with next week.
#
# `\s*` around every separator, because by the time a date reaches this module `src.normalize`
# has put spaces around its punctuation: "11/06/2026" arrives as "11 / 06 / 2026" and
# "2026-06-11" as "2026 - 06 - 11". Without the whitespace neither matched any form here, both
# fell through to the bare-year rule, and a question about one day was answered with the whole
# of that year - `understood=True`, so nothing flagged it. `\s*` not `\s?`: the normalizer's
# spacing is not something to depend on being exactly one space.
_SEP = r"\s*[/.-]\s*"
# "11th" arrives as "11 th" for the same reason, so the suffix is allowed to drift too
_ORDINAL = r"(?:\s*(?:st|nd|rd|th))?"
_CANDIDATE = re.compile(
    rf"\d{{1,2}}{_SEP}\d{{1,2}}{_SEP}{YEAR}"                         # 11/06/2026, 11 . 06 . 2026
    rf"|{YEAR}{_SEP}\d{{1,2}}{_SEP}\d{{1,2}}"                        # 2026-06-11, 2026 - 06 - 11
    rf"|\d{{1,2}}{_ORDINAL}\s+[a-z]{{3,9}}\.?,?\s+{YEAR}"            # 11 jun 2026, 11th June 2026
    rf"|[a-z]{{3,9}}\.?\s+\d{{1,2}}{_ORDINAL},?\s+{YEAR}",           # june 11, 2026
    re.I)

# Starts with a four-digit year, so it reads year-month-day and nothing is ambiguous.
_YEAR_FIRST = re.compile(rf"^{YEAR}{_SEP}")

# Fixed, never "today": a candidate missing a component must not be completed from the wall
# clock, or the same question answers differently tomorrow. The regex above requires a year, so
# this should never actually be reached - it is here so that it cannot be.
_NO_DEFAULTS_FROM_NOW = datetime(2000, 1, 1)


def dates_in(text: str) -> list[date]:
    """Every calendar date in the wording, in the order written, de-duplicated.

    `dayfirst` is decided per candidate rather than globally, and that is load-bearing.
    "11/06/2026" is 11 June in every market this serves, so day-first is right - but applying
    day-first to an ISO "2026-06-11" makes dateutil read it as year-day-month and return the
    6th of November. Silently. That is the exact failure this module exists to prevent, so the
    shape of the candidate picks the reading instead of one flag doing both jobs.
    """
    seen, found = set(), []
    for match in _CANDIDATE.finditer(text or ""):
        written = match.group()
        year_first = bool(_YEAR_FIRST.match(written))
        # dateutil reads "11/06/2026" and not "11 / 06 / 2026", so the spacing the normalizer
        # added comes back out before it is handed over
        squeezed = re.sub(r"\s*([/.-])\s*", r"\1", written)
        try:
            when = _parse(squeezed, dayfirst=not year_first, yearfirst=year_first,
                          default=_NO_DEFAULTS_FROM_NOW).date()
        except (ParserError, ValueError, OverflowError):
            continue                      # "32 jan 2026" is date-shaped and is not a date
        if when not in seen:
            seen.add(when)
            found.append(when)
    return found


def one_date_in(text: str) -> date | None:
    """The single calendar date in the wording, or None when there are none or several."""
    found = dates_in(text)
    return found[0] if len(found) == 1 else None


def month_days(year: int, month: int) -> int:
    """Days in that month. `calendar.monthrange`, named - so leap years are not arithmetic.

    Replaces `(_day(year + month // 12, month % 12 + 1, 1) - timedelta(days=1)).day`, which was
    correct and appeared twice and read like a puzzle both times.
    """
    return calendar.monthrange(year, month)[1]


def demo():
    """Self-check: every written form, and the ones that only look like dates."""
    assert dates_in("2026-06-11") == [date(2026, 6, 11)]
    assert dates_in("2026/06/11") == [date(2026, 6, 11)]
    # the whole reason dayfirst is per-candidate: ISO must not be read day-first
    assert dates_in("2026-06-11 to 2026-06-20") == [date(2026, 6, 11), date(2026, 6, 20)]
    assert dates_in("11/06/2026") == [date(2026, 6, 11)], "day-first, not the American reading"
    assert dates_in("11.06.2026") == [date(2026, 6, 11)]
    assert dates_in("11 jun 2026") == [date(2026, 6, 11)]
    assert dates_in("11th june 2026") == [date(2026, 6, 11)]
    assert dates_in("june 11, 2026") == [date(2026, 6, 11)]
    assert dates_in("11 sept 2026") == [date(2026, 9, 11)]
    assert dates_in("29 feb 2024") == [date(2024, 2, 29)]
    # order written, de-duplicated
    assert dates_in("11 jan 2026 and 17 jan 2026") == [date(2026, 1, 11), date(2026, 1, 17)]
    assert dates_in("17 jan 2026 then 11 jan 2026") == [date(2026, 1, 17), date(2026, 1, 11)]
    assert dates_in("11 jan 2026, 12 jan 2026 and 11 jan 2026") == \
        [date(2026, 1, 11), date(2026, 1, 12)]
    # shaped like a date, is not one
    assert dates_in("29 feb 2023") == [], "2023 was not a leap year"
    assert dates_in("32 jan 2026") == [] and dates_in("13/13/2026") == []
    # no year - a bare month, for the caller to read as the most recent one
    assert dates_in("11 jan") == [] and dates_in("last june") == [] and dates_in("") == []
    assert dates_in("march 2022") == [], "a whole month is not a single date"
    # exactly as `src.normalize` hands them over - spaces around the punctuation
    assert dates_in("rain in guntur on 11 / 06 / 2026") == [date(2026, 6, 11)]
    assert dates_in("rain in guntur on 2026 - 06 - 11") == [date(2026, 6, 11)]
    assert dates_in("rain on 11 . 06 . 2026") == [date(2026, 6, 11)]
    assert dates_in("rain in guntur on 11 th june 2026") == [date(2026, 6, 11)]
    assert one_date_in("rain on 5 mar 2026") == date(2026, 3, 5)
    assert one_date_in("11 jan 2026 and 17 jan 2026") is None, "several is not one"
    assert one_date_in("no date here") is None
    assert month_days(2024, 2) == 29 and month_days(2023, 2) == 28
    assert month_days(2026, 12) == 31 and month_days(2026, 4) == 30
    assert MONTHS["january"] == 1 and MONTHS["jan"] == 1
    assert MONTHS["sept"] == 9 and MONTHS["sep"] == 9 and MONTHS["december"] == 12
    # 12 names + 12 abbreviations - "may", which is its own abbreviation - + "sept"
    assert len(MONTHS) == 24, f"expected 24 month spellings, got {len(MONTHS)}"
    assert MONTHS["may"] == 5, "the one name that is also its own abbreviation"
    assert MONTH_NAMES[0] == "january" and MONTH_NAMES[-1] == "december"
    assert len(MONTH_NAMES) == 12
    print(f"dates demo OK: {len(MONTHS)} month spellings, "
          f"{len(_CANDIDATE.pattern.split('|'))} written forms")


if __name__ == "__main__":
    demo()
