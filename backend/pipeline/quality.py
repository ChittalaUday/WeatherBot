"""
Data quality - what actually came back, before anything is computed from it.

    python tests/test_pipeline_units.py          # the checks for this module

Every upstream feed here can return less than it promised, and the shapes are not theoretical:
the historical endpoint's own sample response contains

    {"Date_time": "2026-08-08T00:00:00", "Rainfall": 1.18}

with all seventeen other columns simply absent. Averaging that row's temperature gives you an
average over nothing, and the difference between "22.5 degrees" and "22.5 degrees, from one
reading out of seven" is the difference between an answer and a guess.

So nothing downstream touches raw rows. It asks here first:

    OK        every field the answer needs is well covered
    PARTIAL   some fields usable, some not - answer with the ones that are, name the ones that are not
    SPARSE    rows arrived but the needed fields are mostly empty - report, never decide
    NO_DATA   nothing came back at all

The advice engine treats SPARSE and NO_DATA as "do not decide". A verdict computed from two
readings out of thirty is worse than saying the data is not there, because it looks the same
as a real one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# Values that mean "missing" while looking like data. -999 and 9999 are the usual sentinels in
# gridded weather output; a bare "" or "NA" shows up when something serialises through CSV.
SENTINELS = {-999, -9999, 999, 9999}
MISSING_TEXT = {"", "na", "n/a", "null", "none", "-", "nan"}

USABLE = 0.60      # a field below this share of real values cannot carry an answer
GOOD = 0.90        # at or above this it is treated as complete


def is_missing(value) -> bool:
    """True for None, blank, a sentinel, a NaN, or anything that will not become a float."""
    if value is None:
        return True
    if isinstance(value, str):
        if value.strip().lower() in MISSING_TEXT:
            return True
        try:
            value = float(value)
        except ValueError:
            return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return float(value) in SENTINELS
    except (TypeError, ValueError):
        return True


def values(rows: list[dict], field_name: str) -> list[float]:
    """Every real numeric value for one field. The only way anything should read a column."""
    out = []
    for row in rows or []:
        v = row.get(field_name)
        if v is not None and not is_missing(v):
            out.append(float(v))
    return out


@dataclass
class Quality:
    status: str                                  # OK | PARTIAL | SPARSE | NO_DATA
    rows: int = 0
    coverage: dict = field(default_factory=dict)   # field -> share of rows with a real value
    usable: list = field(default_factory=list)     # fields good enough to answer from
    unusable: list = field(default_factory=list)
    gaps: int = 0                                  # missing days inside the window
    message: str = ""

    @property
    def can_answer(self) -> bool:
        return self.status in {"OK", "PARTIAL"}

    @property
    def can_decide(self) -> bool:
        """Advice needs every field its rule reads. Reporting can limp; deciding cannot."""
        return self.status == "OK" or (self.status == "PARTIAL" and bool(self.usable))


def assess(rows: list[dict], fields: list[str], *, required: list[str] | None = None,
           expect_daily: int = 0) -> Quality:
    """Coverage of the fields an answer needs, and whether it can be given at all.

    `required` is the subset that must be present - the fields an advice rule reads. When it is
    given, the verdict is about those; when it is not, any usable field will do.
    """
    if not rows:
        return Quality("NO_DATA", 0, message="the source returned no rows for that window")

    coverage = {f: len(values(rows, f)) / len(rows) for f in fields}
    usable = [f for f, c in coverage.items() if c >= USABLE]
    unusable = [f for f, c in coverage.items() if c < USABLE]

    judged = [f for f in (required or fields) if f in coverage]
    worst = min((coverage[f] for f in judged), default=0.0)
    missing_required = [f for f in judged if coverage[f] < USABLE]

    gaps = 0
    if expect_daily:
        stamps = {str(r.get("Date_time", ""))[:10] for r in rows if r.get("Date_time")}
        gaps = max(expect_daily - len(stamps), 0)

    if not usable:
        return Quality("SPARSE", len(rows), coverage, usable, unusable, gaps,
                       f"{len(rows)} rows came back but every field needed is mostly empty")
    if missing_required:
        return Quality("SPARSE", len(rows), coverage, usable, unusable, gaps,
                       "missing the readings this needs: " + ", ".join(missing_required))
    if worst >= GOOD and not unusable and not gaps:
        return Quality("OK", len(rows), coverage, usable, unusable, gaps)

    parts = []
    if unusable:
        parts.append("no usable " + ", ".join(unusable))
    if worst < GOOD:
        parts.append(f"{worst:.0%} coverage on the thinnest reading")
    if gaps:
        parts.append(f"{gaps} day{'s' if gaps > 1 else ''} missing from the range")
    return Quality("PARTIAL", len(rows), coverage, usable, unusable, gaps, "; ".join(parts))


def caveat(quality: Quality) -> str:
    """One sentence to append to an answer, or "" when the data was clean."""
    if quality.status == "NO_DATA":
        return "No data came back for that place and time."
    if quality.status == "SPARSE":
        return f"Not enough data to answer that - {quality.message}."
    if quality.status == "PARTIAL":
        return f"Partial data: {quality.message}."
    return ""
