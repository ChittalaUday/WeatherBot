"""
Query planner - what each source can do, and what this question would cost.

    python -m backend.pipeline.plan            # self-check

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
    if days_back <= CAPABILITY[Source.GFS_HISTORICAL].max_days_back:
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


def demo():
    """Self-check: routing, windows and the budget, on a fixed 'now'."""
    now = datetime(2026, 8, 13, 15, 30)
    here = [{"name": "Guntur", "lat": 16.3, "lon": 80.4}]
    two = here + [{"name": "Vizag", "lat": 17.7, "lon": 83.2}]
    # routing is what this checks; whether the internal archive answers today is a separate
    # question, so the archive-backed cases are asserted with it assumed up
    up = lambda **kw: plan(**{"archive": True, "now": now, **kw})

    # Inside the hourly feed's 24h reach -> hourly, so the answer can say *when*.
    for wording in ("now", "today", "tonight", "this evening"):
        assert up(times_normalized=[wording], places=here).source is Source.GFS_HOURLY, wording

    # ...and outside it -> daily. "tomorrow" asked at 15:30 ends 32h out, past what /hrlydata
    # holds; serving it hourly would drop tomorrow evening and make the answer depend on the
    # clock.
    for wording in ("tomorrow", "this week", "day after tomorrow"):
        p = up(times_normalized=[wording], places=here)
        assert p.source is Source.GFS_DAILY and p.resolution is Resolution.DAILY, (wording, p)

    # The rule is the reach, not the word: the same wording flips as the clock moves.
    assert plan(times_normalized=["tomorrow morning"], places=here, archive=True,
                now=datetime(2026, 8, 13, 6, 0)).source is Source.GFS_DAILY
    assert plan(times_normalized=["tomorrow morning"], places=here, archive=True,
                now=datetime(2026, 8, 13, 22, 0)).source is Source.GFS_HOURLY

    assert up(times_normalized=["yesterday"], places=here).source is Source.GFS_HISTORICAL
    assert up(times_normalized=["yesterday"], places=here,
              level="district").source is Source.POSTGRES_AGG          # an area, not a point
    assert up(times_normalized=["last week"], places=two).source is Source.ZARR_BULK

    p = up(times_normalized=["in august 2019"], places=here)
    assert p.source is Source.ZARR_POINT and p.start.startswith("2019-08-01"), p
    assert p.end.startswith("2019-08-31"), p.end
    assert up(times_normalized=["on 15 august 2023"], places=here).span_days == 1

    # 5,844 days. The ladder already answers this in YEARLY, so it is affordable as asked -
    # 16 rows out of a GROUP BY rather than a million observations.
    p = up(times_normalized=["from 2010 to 2025"], places=here)
    assert p.verdict is Verdict.EXECUTE and p.resolution is Resolution.YEARLY and p.rows == 16, p

    # The ladder caps rows for one place, so the budget only bites when places multiply them.
    assert up(times_normalized=["for all of 2023"], places=here).resolution is Resolution.DAILY
    three = two + [{"name": "Nellore", "lat": 14.4, "lon": 80.0}]
    p = up(times_normalized=["for all of 2023"], places=three)
    assert p.verdict is Verdict.COARSEN and p.rows <= MAX_ROWS and p.offer, p

    assert up(times_normalized=["next month"], places=here).verdict is Verdict.REJECT
    p = up(times_normalized=["tomorrow"], places=[])
    assert p.verdict is Verdict.ASK and "place" in p.reason, p

    # the archive holds six measurements; asking it for soil moisture has to be visible
    p = up(times_normalized=["in august 2019"], places=here, fields=["Rainfall", "Soilm10"])
    assert p.source is Source.ZARR_POINT and p.unservable == ["Soilm10"], p
    assert not up(times_normalized=["tomorrow"], places=here, fields=["Soilm10"]).unservable

    # ...and with the archive down, an old date is refused up front rather than timing out
    down = plan(times_normalized=["in august 2019"], places=here, now=now, archive=False)
    assert down.verdict is Verdict.REJECT and "internal network" in down.reason, down

    # a date within the 7-day lookback is served by the forecast API, archive or no archive
    recent = plan(times_normalized=["11 august 2026"], places=here, archive=False,
                  now=datetime(2026, 8, 14, 13, 0))
    assert recent.verdict is Verdict.EXECUTE and recent.source is Source.GFS_HISTORICAL, recent

    for wording in ("in 2017", "for all of 2023", "every year since 2018", "in the last decade",
                    "over the past 6 months", "on 12/06/2021", "2023-08-15", "march 2022",
                    "11 jan 2026 and 17 jan 2026"):
        got = up(times_normalized=[wording], places=here)
        assert got.verdict in (Verdict.EXECUTE, Verdict.COARSEN), (wording, got)
        assert got.rows <= MAX_ROWS, (wording, got.rows)

    print("plan demo OK")
    for wording, where in (("tomorrow", here), ("yesterday", here), ("in august 2019", here),
                           ("from 2010 to 2025", here), ("over the last 5 years", here),
                           ("next month", here), ("for all of 2023", here),
                           ("for all of 2023", three)):
        got = up(times_normalized=[wording], places=where)
        name = f"{wording} x{len(where)}" if len(where) > 1 else wording
        print(f"  {name:22s} {got.verdict.value:8s} {str(got.source and got.source.value):15s} "
              f"{str(got.resolution and got.resolution.value):8s} {got.span_days:>5}d "
              f"{got.rows:>5} rows  {got.reason}")


if __name__ == "__main__":
    demo()
