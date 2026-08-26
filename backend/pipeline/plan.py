"""
Query planner - what each source can do, and what this question would cost.

    python tests/test_pipeline_units.py          # the checks for this module

The NLU says what was asked. Nothing in it knows that Zarr serves points and Postgres serves
districts, that the forecast stops at ten days, or that ten years of daily rows is a million
observations. That is all here, because it changes when a database is tuned - not when a model
is retrained.

    window (backend.pipeline.timewindow)  ->  source  ->  resolution  ->  rows  ->  verdict

Four verdicts:

    EXECUTE   affordable as asked
    COARSEN   too many rows at the resolution implied, but a coarser one fits - do it and say so
    ASK       even the coarsest resolution is too much, or a needed slot is missing
    REJECT    no source holds this at all (a forecast past the horizon)

"Too long" is never a REJECT on its own. Rainfall for each year from 2010 to 2025 is sixteen
rows out of a GROUP BY; refusing it because the span is fifteen years would be refusing
arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from backend.config import ARCHIVE_REACHABLE, MAX_ROWS
from backend.pipeline.timewindow import resolve as resolve_window
from src.v4.schema import FORECAST_HORIZON_DAYS, Resolution, resolutions_for, rows_for

# How far ahead /hrlydata actually reaches. Measured, not assumed: it returns 25 rows spanning
# exactly 24 hours from the current hour.
#
# This is a reach, not a span, and the difference matters. Gating hourly on span length means
# "tomorrow" (span 1 day) qualifies - but asked at 13:43 the feed only holds tomorrow 00:00 to
# 13:00, so the answer silently drops the afternoon AND changes depending on what time you ask
# it. Hourly is only honest when the whole requested window fits inside the reach.
HOURLY_REACH_HOURS = 24
# ponytail: one global row budget. Per-source budgets if Postgres turns out to be far cheaper
# than Zarr for the same span, which is likely but unmeasured.


class Source(str, Enum):
    GFS_DAILY = "GFS_DAILY"              # /interpolate            point, forecast
    GFS_HOURLY = "GFS_HOURLY"            # /hrlydata               point, forecast, hourly
    GFS_HISTORICAL = "GFS_HISTORICAL"    # /interpolate/historical point, N days back
    ZARR_POINT = "ZARR_POINT"            # /weather                point, any dates
    ZARR_BULK = "ZARR_BULK"              # /bulk-get-weather/*     many points, any dates
    POSTGRES_AGG = "POSTGRES_AGG"        # weather.daily_*_weather admin levels, pre-aggregated


class Verdict(str, Enum):
    EXECUTE = "EXECUTE"
    COARSEN = "COARSEN"
    ASK = "ASK"
    REJECT = "REJECT"


# Geography a source can address. A village or a raw lat/lon is a POINT; a district or a state
# is an AREA that Postgres already has pre-aggregated and Zarr would have to average by hand.
POINT, AREA = "point", "area"
ADMIN_LEVELS = {"district", "state", "country", "block"}

# The columns each source actually returns. Not decoration: Zarr carries six measurements and
# the GFS feed carries eighteen, so "soil moisture last August" is a question the archive
# cannot answer however well it is planned. Naming that here is what lets the reply say which
# reading is unavailable instead of returning a column of nulls.
GFS_FIELDS = frozenset({"Rainfall", "Tmin", "Tmax", "Tavg", "RH", "RH_max", "RH_min", "DPT",
                        "Wind_Speed", "Wind_max", "Wind_Direction", "SunSD", "DayLength",
                        "Lowcloud", "Soilm10", "Soilm40", "Soilt10"})
# Tavg is derived from Tmax/Tmin by the adapter - the feed does not send it.
ZARR_FIELDS = frozenset({"Rainfall", "Tmax", "Tmin", "Tavg", "RH", "Wind_Speed", "DayLength"})


@dataclass(frozen=True)
class Capability:
    geography: str
    past: bool
    future: bool
    max_days_back: int | None       # None = unbounded
    max_days_ahead: int
    resolutions: tuple
    fields: frozenset = GFS_FIELDS
    multi_point: bool = False
    needs_network: str = "public"   # "internal" = only reachable inside the VPN


CAPABILITY = {
    Source.GFS_HOURLY: Capability(POINT, past=False, future=True, max_days_back=0,
                                  max_days_ahead=FORECAST_HORIZON_DAYS,
                                  resolutions=(Resolution.HOURLY,)),
    Source.GFS_DAILY: Capability(POINT, past=False, future=True, max_days_back=0,
                                 max_days_ahead=FORECAST_HORIZON_DAYS,
                                 resolutions=(Resolution.DAILY,)),
    # Seven days, measured - the `days` query parameter is accepted and ignored, so days=7 and
    # days=400 both return the same last-7-days window. Declaring 60 here meant anything from
    # 8 to 60 days back was routed to an endpoint that could not serve it.
    Source.GFS_HISTORICAL: Capability(POINT, past=True, future=False, max_days_back=7,
                                      max_days_ahead=0, resolutions=(Resolution.DAILY,)),
    Source.ZARR_POINT: Capability(POINT, past=True, future=False, max_days_back=None,
                                  max_days_ahead=0,
                                  resolutions=(Resolution.DAILY, Resolution.WEEKLY,
                                               Resolution.MONTHLY, Resolution.YEARLY),
                                  fields=ZARR_FIELDS, needs_network="internal"),
    Source.ZARR_BULK: Capability(POINT, past=True, future=False, max_days_back=None,
                                 max_days_ahead=0,
                                 resolutions=(Resolution.DAILY, Resolution.MONTHLY,
                                              Resolution.YEARLY),
                                 fields=ZARR_FIELDS, multi_point=True,
                                 needs_network="internal"),
    Source.POSTGRES_AGG: Capability(AREA, past=True, future=False, max_days_back=None,
                                    max_days_ahead=0,
                                    resolutions=(Resolution.DAILY, Resolution.WEEKLY,
                                                 Resolution.MONTHLY, Resolution.YEARLY)),
}


@dataclass
class QueryPlan:
    verdict: Verdict
    source: Source | None
    resolution: Resolution | None
    start: str = ""
    end: str = ""
    span_days: int = 0
    rows: int = 0
    label: str = ""
    unservable: list = field(default_factory=list)
    reason: str = ""
    offer: list = field(default_factory=list)      # coarser resolutions the user could take
    notes: list = field(default_factory=list)

    @property
    def hourly(self) -> bool:
        return self.resolution is Resolution.HOURLY

    def as_dict(self) -> dict:
        """The wire form. One definition, so the chat payload and the debug stage agree."""
        return {"verdict": self.verdict.value,
                "source": self.source and self.source.value,
                "resolution": self.resolution and self.resolution.value,
                "window": self.label, "start": self.start, "end": self.end,
                "span_days": self.span_days, "rows": self.rows,
                "unservable": self.unservable, "reason": self.reason, "notes": self.notes}


def _pick_source(*, tense: str, geography: str, points: int, days_back: int) -> Source:
    """First source whose capability covers the request. Order encodes preference."""
    if geography == AREA:
        return Source.POSTGRES_AGG                 # pre-aggregated, always cheapest
    if tense == "future":
        return Source.GFS_DAILY                    # hourly is chosen later, by resolution
    if points > 1:
        return Source.ZARR_BULK
    max_days = CAPABILITY[Source.GFS_HISTORICAL].max_days_back
    if max_days is not None and days_back <= max_days:
        return Source.GFS_HISTORICAL               # same columns as the forecast feed
    return Source.ZARR_POINT


def plan(*, times_normalized: list[str], places: list[dict], aggregation: str = "RAW",
         level: str = "village", fields: list[str] | None = None,
         now: datetime | None = None, archive: bool | None = None) -> QueryPlan:
    """Slots in, an executable plan out."""
    now = now or datetime.now()
    window = resolve_window(times_normalized[0] if times_normalized else None, now)
    start, end, span_days, label = window.start, window.end, window.span_days, window.label

    if not places:
        return QueryPlan(Verdict.ASK, None, None, span_days=span_days, label=label,
                         reason="no place resolved yet")

    # Tense is decided by where the window STARTS, not where it ends. "over the last 5 years"
    # ends a few minutes from now, and end-based tense sent it to the forecast feed - which
    # holds ten days and refused 1,826.
    tense = "past" if start.date() < now.date() else "future"
    geography = AREA if level in ADMIN_LEVELS else POINT
    days_ahead = max((end.date() - now.date()).days, 0)
    days_back = max((now.date() - start.date()).days, 0)

    if tense == "future" and days_ahead > FORECAST_HORIZON_DAYS:
        return QueryPlan(
            Verdict.REJECT, None, None, span_days=span_days, label=label,
            reason=f"the forecast runs about {FORECAST_HORIZON_DAYS} days ahead; that is "
                   f"{days_ahead} days out")

    source = _pick_source(tense=tense, geography=geography, points=len(places),
                          days_back=days_back)
    # An archive-only window with no archive is a refusal, not a fetch that fails later.
    reachable = ARCHIVE_REACHABLE if archive is None else archive
    iso = lambda d: d.isoformat(timespec="minutes")
    if (source in {Source.ZARR_POINT, Source.ZARR_BULK}
            and CAPABILITY[source].needs_network == "internal" and not reachable):
        return QueryPlan(
            Verdict.REJECT, source, None, iso(start), iso(end), span_days, 0, label,
            reason=f"that is {days_back} days back; I keep the last "
                   f"{CAPABILITY[Source.GFS_HISTORICAL].max_days_back} days here and the "
                   f"archive that holds older dates is on the internal network")

    allowed = [r for r in resolutions_for(span_days) if r in CAPABILITY[source].resolutions]
    if not allowed:                                 # e.g. hourly asked of a monthly source
        allowed = [CAPABILITY[source].resolutions[0]]

    # Hourly for any short forward window, not just ones that name a clock time. Requiring an
    # hourly-flavoured wording meant "rain today" came back as one daily row - and six of the
    # nine corrections in the feedback table say the same thing ("wrong api, it should call
    # hourly"). A daily total cannot say *when*, and "rain from 14:00" is the whole value of
    # asking. 24 rows instead of 1 is nothing.
    if tense == "future" and geography == POINT and end <= now + timedelta(hours=HOURLY_REACH_HOURS):
        source, allowed = Source.GFS_HOURLY, [Resolution.HOURLY]

    finest = allowed[0]
    rows = rows_for(span_days, finest) * max(len(places), 1)
    notes = []
    unservable = sorted(set(fields or []) - CAPABILITY[source].fields)
    if unservable:
        notes.append(f"{source.value} has no " + ", ".join(unservable))
    if CAPABILITY[source].needs_network == "internal":
        notes.append("internal network only - falls back to GFS_HISTORICAL if unreachable")

    if rows <= MAX_ROWS:
        return QueryPlan(Verdict.EXECUTE, source, finest, iso(start), iso(end), span_days, rows,
                         label, unservable=unservable, notes=notes)

    affordable = [r for r in allowed
                  if rows_for(span_days, r) * max(len(places), 1) <= MAX_ROWS]
    if affordable:
        chosen = affordable[0]
        got = rows_for(span_days, chosen) * max(len(places), 1)
        return QueryPlan(Verdict.COARSEN, source, chosen, iso(start), iso(end), span_days, got,
                         label,
                         reason=f"{rows:,} {finest.value.lower()} rows is too many; "
                                f"{got:,} {chosen.value.lower()} rows instead",
                         offer=[r.value for r in affordable], notes=notes)

    coarsest = CAPABILITY[source].resolutions[-1]
    return QueryPlan(Verdict.ASK, source, None, iso(start), iso(end), span_days, rows, label,
                     reason=f"{span_days:,} days is too much even at "
                            f"{coarsest.value.lower()} - narrow the range",
                     offer=[coarsest.value], notes=notes)
