"""
The pipeline's unit checks. Run: python tests/test_pipeline_units.py

These were `demo()` functions living inside the modules they check. Same assertions, same
comments, now in one place that can be run without importing a module for its side effect - and
`backend/pipeline/*.py` is left with nothing in it but the code that runs in production.
"""

from _root import ROOT  # noqa: F401 - puts the repo root on sys.path


def check_timewindow():
    """Self-check: the calendar arithmetic and the row picker, on a fixed 'now'."""
    from backend.pipeline.timewindow import date, datetime, resolve, select_rows
    now = datetime(2026, 8, 13, 15, 30)          # a Thursday

    assert resolve("tomorrow", now).start.date() == date(2026, 8, 14)
    assert resolve("day after tomorrow", now).start.date() == date(2026, 8, 15)
    assert resolve("yesterday", now).start.date() == date(2026, 8, 12)
    assert resolve("monday", now).start.date() == date(2026, 8, 17)      # next, never today

    evening = resolve("this evening", now)
    assert evening.granularity == "hourly" and evening.start.hour == 17

    assert resolve("18:45", now).start == datetime(2026, 8, 13, 18, 45)
    assert resolve("06:00", now).start == datetime(2026, 8, 14, 6, 0)    # already gone

    span = resolve("next 3 days", now)
    assert span.start.date() == date(2026, 8, 13) and span.end.date() == date(2026, 8, 15)
    assert resolve("this weekend", now).start.date() == date(2026, 8, 15)   # Saturday

    default = resolve(None, now)
    assert default.label == "next few days" and default.granularity == "daily"

    # calendar wording, which the relative resolver alone could never do
    assert resolve("in august 2019", now).start.date() == date(2019, 8, 1)
    assert resolve("in august 2019", now).end.date() == date(2019, 8, 31)
    assert resolve("from 2010 to 2025", now).span_days == 5844
    assert resolve("on 15 august 2023", now).span_days == 1
    assert resolve("for all of 2023", now).start.date() == date(2023, 1, 1)
    # the abbreviated-month bug: this must be one day, not the whole of 2026
    assert resolve("11 jun 2026", now).span_days == 1, resolve("11 jun 2026", now)
    # a date range must not collapse to its first date
    assert resolve("11 jan 2026 and 17 jan 2026", now).span_days == 7

    # ...and the row picker reads the same calendar
    daily = [{"Date_time": f"2026-08-{d:02d}T00:00:00"} for d in range(12, 22)]
    picked, label = select_rows(daily, "tomorrow", now)
    assert len(picked) == 1 and picked[0]["Date_time"].startswith("2026-08-14"), picked
    assert select_rows(daily, "next 3 days", now)[0] == daily[1:4]       # 12 Aug is past
    assert len(select_rows(daily, "", now)[0]) == 7 and label == "tomorrow"
    assert select_rows(daily, "15 august 2026", now)[0][0]["Date_time"].startswith("2026-08-15")
    assert select_rows([], "tomorrow", now) == ([], "no data")

    hourly = [{"Date_time": f"2026-08-13T{h:02d}:00:00"} for h in range(24)]
    assert len(select_rows(hourly, "this evening", now)[0]) == 5        # 17:00-21:00

    print("timewindow demo OK")
    for wording in ("tomorrow", "this evening", "in august 2019", "from 2010 to 2025",
                    "over the last 5 years", "11 jun 2026", None):
        w = resolve(wording, now)
        print(f"  {str(wording):22s} {w.start:%Y-%m-%d %H:%M} -> {w.end:%Y-%m-%d %H:%M}  "
              f"{w.span_days:>5}d  {w.granularity:6s} {w.label}")

def check_places():
    """Self-check for the parts that need no network."""
    from backend.pipeline.places import (
        ALIAS_FILE,
        ALIASES,
        RELATIVE_LOCATIONS,
        SELF_NAMED_STATES,
        STATE_ALIASES,
        _squeeze,
        canonical_state,
        is_relative,
        normalize,
        relative_in,
    )
    # every lookup is data now: four empty sets would pass every assertion but this one
    assert ALIASES and STATE_ALIASES and SELF_NAMED_STATES and RELATIVE_LOCATIONS, \
        f"{ALIAS_FILE} did not load"
    assert normalize("KKD") == "Kakinada"
    assert normalize("bza") == "Vijayawada"
    assert normalize("Kakinada") == "Kakinada"          # unknown text passes through
    assert canonical_state("AP") == "Andhra Pradesh"
    assert canonical_state("andhrapradesh") == "Andhra Pradesh"
    assert canonical_state("Kakinada") is None
    # doubled letters are a spelling, not a different place - but a changed letter is
    assert _squeeze("Beeramguda")[:3] == _squeeze("beramguda")[:3]
    assert _squeeze("Kompally")[:3] == _squeeze("kompaly")[:3]
    assert _squeeze("Belamguda")[:3] != _squeeze("beramguda")[:3]
    assert is_relative("my field") and not is_relative("Guntur")
    # v4 tags no span for a relative place, so the sentence is where it has to be found
    assert relative_in("soil moisture in my field") == ["my field"]
    assert relative_in("will it rain near me") == ["near me"]
    assert relative_in("will it rain in Guntur") == []
    print(f"locations demo OK: {len(ALIASES)} aliases, {len(STATE_ALIASES)} state spellings, "
          f"{len(SELF_NAMED_STATES)} self-named states, {len(RELATIVE_LOCATIONS)} relative")

def check_plan():
    """Self-check: routing, windows and the budget, on a fixed 'now'."""
    from backend.pipeline.plan import MAX_ROWS, Resolution, Source, Verdict, datetime, plan
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

def check_windows():
    """Self-check: runs, spacing, labels, and the fragmented case the totals could not see."""
    from backend.pipeline.windows import (
        below,
        between,
        coverage,
        every,
        fragmented,
        longest,
        reading,
        runs,
        spacing_hours,
    )
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

def check_analysis():
    """Self-check: the guard, the reductions, and that notes carry their kind."""
    from backend.pipeline.analysis import (
        MAX_NOTES,
        apply_aggregation,
        build_chart,
        build_insights,
        confirm_aggregation,
        wants_chart,
    )
    days = lambda n, **f: [{"Date_time": f"2026-08-{d:02d}T00:00:00",
                            **{k: (v[d - 1] if isinstance(v, list) else v) for k, v in f.items()}}
                           for d in range(1, n + 1)]

    # the guard, both ways: a reduction the wording never asked for is dropped, and one the
    # wording plainly asks for is added when the model missed it
    assert confirm_aggregation("weather in KKD", "MAX") == "RAW"
    assert confirm_aggregation("hottest day in KKD", "MAX") == "MAX"
    assert confirm_aggregation("weather in KKD", "RAW") == "RAW"
    assert confirm_aggregation("total rain in KKD this week", "RAW") == "SUM"
    assert confirm_aggregation("rain in KKD altogether", "RAW") == "SUM"
    assert confirm_aggregation("average temperature in KKD", "RAW") == "AVG"
    # ...but a loose phrase does not invent one. "how much rain tomorrow" is a question about
    # tomorrow, not a request for a sum.
    assert confirm_aggregation("how much rain in KKD tomorrow", "RAW") == "RAW"
    assert confirm_aggregation("overall weather in KKD", "RAW") == "RAW"

    rows = days(5, Rainfall=[1.0, 12.0, 0.0, 0.5, 3.0])
    assert apply_aggregation(rows, "Rainfall", "RAW") is None
    s_agg = apply_aggregation(rows, "Rainfall", "SUM")
    assert s_agg is not None and s_agg["value"] == 16.5
    m_agg = apply_aggregation(rows, "Rainfall", "MAX")
    assert m_agg is not None and m_agg["value"] == 12.0
    assert m_agg["at"].startswith("2026-08-02")
    assert apply_aggregation([], "Rainfall", "SUM") is None
    # sentinels are not values - -999 must never become the minimum
    min_agg = apply_aggregation(days(3, Rainfall=[-999, 4.0, 6.0]), "Rainfall", "MIN")
    assert min_agg is not None and min_agg["value"] == 4.0

    trend = apply_aggregation(days(4, Tmax=[34.0, 36.0, 33.0, 31.0]), "Tmax", "TREND")
    assert trend is not None and trend["kind"] == "TREND" and "dropping" in trend["text"], trend
    rise = apply_aggregation(days(4, Tmax=[31.0, 30.0, 33.0, 36.0]), "Tmax", "TREND")
    assert rise is not None and "climbing" in rise["text"], rise
    assert apply_aggregation(days(2, Tmax=[30.0, 31.0]), "Tmax", "TREND") is None   # too short

    # notes carry their kind, so the prompt layer can group them
    notes = build_insights([rows], [{"name": "Guntur"}], ["Rainfall"], "RAW", hourly=False)
    kinds = {n.kind for n in notes}
    assert "RANGE" in kinds and "THRESHOLD" in kinds and "DRY_SPELL" in kinds, notes
    assert all(n.place == "" for n in notes), "one place needs no prefix"
    assert len(build_insights([rows] * 5, [{"name": f"P{i}"} for i in range(5)],
                              ["Rainfall"], "RAW", False)) <= MAX_NOTES

    # the comparison survives the cap, because it is what a comparison question asked for
    two = build_insights([rows, days(5, Rainfall=0.0)],
                         [{"name": "Guntur"}, {"name": "Vizag"}], ["Rainfall"], "RAW", False)
    assert two[0].kind == "COMPARISON", two
    assert two[0].text.startswith("Guntur leads Vizag"), two[0]

    # a decision does not come with a graph unless the question asked for one
    assert not wants_chart("should i take raincoat?")
    assert wants_chart("show me a graph of rain this week")
    assert not wants_chart("rain today")

    chart = build_chart([rows], [{"name": "Guntur"}], "Rainfall", hourly=False)
    assert chart is not None and chart["type"] == "line" and len(chart["series"][0]["points"]) == 5, chart
    assert build_chart([rows[:1]], [{"name": "Guntur"}], "Rainfall", False) is None  # one point

    print("analysis demo OK")
    for note in notes + two[:1]:
        print(f"  [{note.kind:11s}] {note.text}")

def check_quality():
    """Self-check against the shapes this API is actually known to return."""
    from backend.pipeline.quality import assess, caveat, values
    full = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": 1.0 + d, "Tavg": 25.0,
             "Wind_Speed": 3.0} for d in range(1, 8)]

    ok = assess(full, ["Rainfall", "Tavg"])
    assert ok.status == "OK" and ok.can_decide, ok

    # the real historical sample: one row carries only Date_time and Rainfall
    ragged = full[:6] + [{"Date_time": "2026-08-08T00:00:00", "Rainfall": 1.18}]
    partial = assess(ragged, ["Rainfall", "Tavg"])
    assert partial.status == "PARTIAL", partial
    assert partial.coverage["Rainfall"] == 1.0 and partial.coverage["Tavg"] < 1.0, partial.coverage

    # a rule that needs the thin field must not get a verdict
    needs_temp = assess([{"Date_time": "x", "Rainfall": 2.0}] * 5 + [{"Tavg": 25.0}],
                        ["Rainfall", "Tavg"], required=["Tavg"])
    assert needs_temp.status == "SPARSE" and not needs_temp.can_decide, needs_temp

    assert assess([], ["Rainfall"]).status == "NO_DATA"
    assert not assess([], ["Rainfall"]).can_answer

    # sentinels and junk are missing, not values
    junk = [{"Rainfall": -999}, {"Rainfall": "NA"}, {"Rainfall": None},
            {"Rainfall": float("nan")}, {"Rainfall": ""}, {"Rainfall": 4.2}]
    assert values(junk, "Rainfall") == [4.2], values(junk, "Rainfall")
    assert assess(junk, ["Rainfall"]).status == "SPARSE"

    # strings that are really numbers still count
    assert values([{"Rainfall": "3.5"}], "Rainfall") == [3.5]

    # gaps inside the window are reported even when every present row is complete
    gappy = assess(full[:3], ["Rainfall", "Tavg"], expect_daily=7)
    assert gappy.status == "PARTIAL" and gappy.gaps == 4, gappy

    assert caveat(assess([], ["Rainfall"])).startswith("No data")
    assert caveat(ok) == ""
    print("quality demo OK")
    for q in (ok, partial, needs_temp, assess([], ["Rainfall"]), gappy):
        print(f"  {q.status:8s} rows={q.rows:<3} usable={q.usable} "
              f"decide={q.can_decide}  {q.message}")

def check_advice():
    """Self-check: timing beats totals, thresholds still flip, and nothing decides blind."""
    from backend.pipeline.advice import CAUTION, NO, UNKNOWN, YES, Advice, evaluate, values
    def hourly_rows(pattern, start=6, **extra):
        """One row per hour from `start`. `pattern` is rainfall; extras are constant."""
        return [{"Date_time": f"2026-08-18T{start + i:02d}:00:00", "Rainfall": v, **extra}
                for i, v in enumerate(pattern)]

    def days(n, **fields):
        return [{"Date_time": f"2026-08-{d:02d}T00:00:00",
                 **{k: (v[d - 1] if isinstance(v, list) else v) for k, v in fields.items()}}
                for d in range(1, n + 1)]

    def ev(*args, **kwargs) -> Advice:
        res = evaluate(*args, **kwargs)
        assert res is not None
        return res

    # ---- the whole point: the same total, two different answers -------------------
    # 12mm either way. In the afternoon it leaves a clear morning; spread through the day it
    # leaves nothing. The old engine summed and said no to both.
    afternoon = hourly_rows([0.0] * 6 + [4.0, 5.0, 3.0], RH=60.0)
    scattered = hourly_rows([2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 2.0, 0.0, 4.0], RH=60.0)
    assert round(sum(values(afternoon, "Rainfall"))) == round(sum(values(scattered, "Rainfall")))

    morning = ev("DRYING", afternoon, hourly=True)
    assert morning.verdict == YES, morning
    assert "06:00 to 12:00" in morning.headline, morning.headline
    broken = ev("DRYING", scattered, hourly=True)
    assert broken.verdict == CAUTION and "breaking up" in broken.headline, broken.headline

    # ...and rain right through is still a flat no, with no window to offer
    soaked = ev("DRYING", hourly_rows([3.0] * 9, RH=95.0), hourly=True)
    assert soaked.verdict == NO and not soaked.window, soaked

    # ---- a period that is entirely clear says so, rather than naming a window -----
    clear = ev("OUTDOOR_ACTIVITY", hourly_rows([0.0] * 8, Tmax=30.0), hourly=True)
    assert clear.verdict == YES and "any time" in clear.headline, clear.headline

    # ---- spraying needs the rain to stay away afterwards, not just during --------
    # Two calm hours then rain an hour later: the window exists and is still useless.
    calm_then_rain = hourly_rows([0.0, 0.0, 0.0, 5.0, 5.0, 5.0], Wind_Speed=3.0)
    washed = ev("SPRAY", calm_then_rain, hourly=True)
    assert washed.verdict == NO and "wash off" in washed.headline, washed.headline
    # ...and with the afternoon clear too, the same morning is a yes
    long_clear = hourly_rows([0.0] * 10, Wind_Speed=3.0)
    assert ev("SPRAY", long_clear, hourly=True).verdict == YES

    # wind is a per-reading condition like rain: a gusty hour breaks the window
    gusty = [{**r, "Wind_Speed": 3.0 if i % 2 else 9.0} for i, r in enumerate(long_clear)]
    assert ev("SPRAY", gusty, hourly=True).verdict in (CAUTION, NO)

    # ---- daily rows read the same rules, and never invent a clock time -----------
    week = days(7, Rainfall=[0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 0.0], RH=70.0)
    harvest = ev("HARVEST", week, hourly=False)
    assert harvest.verdict == YES, harvest
    assert ":" not in harvest.headline, f"a daily feed cannot say a time: {harvest.headline}"
    assert "01 Aug" in harvest.headline or "1 Aug" in harvest.headline, harvest.headline
    # two dry days is not the three harvesting wants
    assert ev("HARVEST", days(4, Rainfall=[0.0, 0.0, 8.0, 0.0], RH=70.0),
              hourly=False).verdict == CAUTION

    # ---- the length of the period must not change the answer ---------------------
    # A sum only grows, so the old engine answered "can I go out today" and "...this week"
    # differently for identical weather. The window is the same window either way.
    drizzle = [0.3, 0.0, 0.4, 0.0, 0.3, 0.0, 0.4]
    short = ev("RAIN_PROTECTION", days(3, Rainfall=drizzle[:3]), hourly=False)
    long = ev("RAIN_PROTECTION", days(7, Rainfall=drizzle), hourly=False)
    assert short.verdict == long.verdict == NO, (short.verdict, long.verdict)
    # ...and one real shower is a coat however small the total
    assert ev("RAIN_PROTECTION", days(2, Rainfall=[0.0, 6.0]), hourly=False).verdict == YES

    # ---- state rules keep accumulating, because that is their actual question ----
    # straddle the 25mm leaching threshold: 24.9 carries it in, 25.1 washes it away
    assert ev("FERTILIZE", days(2, Rainfall=[12.4, 12.5])).verdict == YES     # 24.9
    assert ev("FERTILIZE", days(2, Rainfall=[12.5, 12.6])).verdict == NO      # 25.1
    assert ev("FERTILIZE", days(2, Rainfall=[1.0, 0.9])).verdict == CAUTION   # 1.9
    assert ev("FERTILIZE", days(2, Rainfall=[0.1, 0.1], Soilm10=0.10)).verdict == CAUTION

    # SOW wants the rain HARVEST is avoiding - the same days, opposite verdicts
    wet = days(4, Rainfall=6.0, Soilm10=0.22, Soilt10=24.0, RH=80.0)
    assert ev("SOW", wet).verdict == YES
    assert ev("HARVEST", wet, hourly=False).verdict == NO

    assert ev("IRRIGATE", days(3, Soilm10=0.12, Rainfall=0.0)).verdict == YES
    assert ev("IRRIGATE", days(3, Soilm10=0.35, Rainfall=0.0)).verdict == NO
    assert ev("IRRIGATE", days(3, Soilm10=0.12, Rainfall=4.0)).verdict == NO  # 12mm coming

    assert ev("CLOTHING", days(2, Tmin=12.0, Tmax=22.0)).verdict == YES
    assert ev("CLOTHING", days(2, Tmin=25.0, Tmax=35.0)).verdict == NO

    # ---- nothing decides without the readings its rule needs --------------------
    assert ev("SPRAY", []).verdict == UNKNOWN
    assert ev("SPRAY", [{"Date_time": "x", "Rainfall": 0.0}] * 5).verdict == UNKNOWN
    assert ev("IRRIGATE", days(5, Rainfall=0.0)).verdict == UNKNOWN   # no soil column
    assert evaluate("NONE", days(3, Rainfall=0.0)) is None

    # ---- granularity is inferred when the caller does not know it ---------------
    assert ev("DRYING", afternoon).verdict == YES, "hourly timestamps, inferred"

    print("advice demo OK")
    print(f"  {'12mm, all afternoon':26s} {morning.verdict:8s} {morning.headline}")
    print(f"  {'12mm, spread all day':26s} {broken.verdict:8s} {broken.headline}")
    print(f"  {'12mm, rain right through':26s} {soaked.verdict:8s} {soaked.headline}")
    for name, rows, hourly in (
        ("SPRAY rain 3h later", calm_then_rain, True),
        ("SPRAY clear all day", long_clear, True),
        ("HARVEST 3 dry days", week, False),
        ("FERTILIZE 35mm coming", days(2, Rainfall=[18.0, 17.0]), False),
        ("IRRIGATE dry soil", days(3, Soilm10=0.12, Rainfall=0.0), False),
    ):
        got = ev(name.split()[0], rows, hourly=hourly)
        print(f"  {name:26s} {got.verdict:8s} {got.headline}")

def check_render():
    """Self-check: a comparison names the winner, and says which way it won.

    "Min temp: A 21.3 · B 20.5. Highest: A." was true and unreadable - both a human skimming
    and the phrasing model downstream read "min ... A" as "A is the lower one", and the chat
    duly said the opposite of the data.
    """
    from backend.pipeline.render import build_table, format_value, summarize, summary_stat
    places = [{"name": "Hyderabad"}, {"name": "Vijawada"}]
    rows = [[{"Date_time": "2026-08-14T00:00:00", "Tmin": 21.3}],
            [{"Date_time": "2026-08-14T00:00:00", "Tmin": 20.5}]]
    said = summarize("COMPARE", rows, ["Tmin"], places, "tomorrow")
    assert said.startswith("Hyderabad has the higher"), said
    assert "21.3" in said and "Vijawada 20.5" in said, said

    # Adding happens only when a total was asked for. This is the whole rule: the same week
    # of rain is a mean under RAW and a total under SUM, and never a total by accident.
    assert summary_stat("Rainfall", [1.0, 2.0, 3.0], "SUM") == (6.0, "total")
    assert summary_stat("Rainfall", [1.0, 2.0, 3.0], "RAW") == (2.0, "average")
    assert summary_stat("Tmax", [10.0, 20.0], "SUM") == (15.0, "average")   # never additive
    assert summary_stat("Rainfall", [], "SUM") == (0.0, "")

    # ...and the sentence follows it. A week of rain under RAW describes the series; the total
    # appears only when the question said "total".
    week = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": v}
            for d, v in enumerate([0.2, 6.0, 0.0, 0.1, 3.0], start=14)]
    raw = summarize("GET", week, ["Rainfall"], places[:1], "this week")
    assert "total" not in raw and "rain on 2 of 5 readings" in raw, raw
    assert "up to 6.0mm" in raw, raw
    # the description does not change when a total was asked for - the total is said once, by
    # `analysis.apply_aggregation`, and this sentence follows it
    assert summarize("GET", week, ["Rainfall"], places[:1], "this week", "SUM") == raw
    dry = summarize("GET", [{"Date_time": "2026-08-14T00:00:00", "Rainfall": 0.1},
                            {"Date_time": "2026-08-15T00:00:00", "Rainfall": 0.0}],
                    ["Rainfall"], places[:1], "tomorrow")
    assert "little to no rain" in dry, dry

    # a comparison names which statistic it ranked on, so "12.5mm" cannot read as a reading
    compared = summarize("COMPARE", [week, week[:2]], ["Rainfall"], places, "this week")
    assert "average rainfall" in compared, compared

    # sentinels never reach a summary - `values` filters them, so this is 4.2 and nothing else
    junk = [{"Date_time": "2026-08-14T00:00:00", "Rainfall": -999},
            {"Date_time": "2026-08-15T00:00:00", "Rainfall": 4.2}]
    assert "4.2mm" in summarize("GET", junk, ["Rainfall"], places[:1], "tomorrow")

    # a column the feed never sent must not become a table column of dashes
    table = build_table([{"Date_time": "2026-08-14T16:00:00", "RH": 58.8}], ["RH"], places[:1],
                        hourly=True)
    assert [c["key"] for c in table["columns"]] == ["time", "RH"], table["columns"]
    assert table["rows"] == [{"time": "14 Aug 16:00", "RH": "58.8"}], table["rows"]

    # a comparison table is one column per place, zipped on time
    wide = build_table(rows, ["Tmin"], places, hourly=False)
    assert [c["key"] for c in wide["columns"]] == ["time", "Hyderabad", "Vijawada"], wide
    assert wide["rows"] == [{"time": "14 Aug", "Hyderabad": "21.3", "Vijawada": "20.5"}], wide

    assert format_value("Wind_Direction", 90) == "90° E", format_value("Wind_Direction", 90)
    assert format_value("Lowcloud", 0.42) == "42%"
    assert format_value("Rainfall", None) == "-"

    print("render demo OK")
    print(f"  {said}")

def check_routing():
    """The route, the profile and the parameters - and the two bugs they surfaced.

    Both bugs were the same shape: an expression the code did not recognise fell through to a
    default that looked like an answer. Neither failed loudly, and both answered a question
    about a month with a week of data.
    """
    from datetime import datetime
    from types import SimpleNamespace

    from backend.pipeline import params, profiles
    from backend.pipeline.timewindow import resolve, select_rows

    now = datetime(2026, 8, 26, 13, 0)

    # --- bug 1: a bare month had no branch in `resolve` and became the forward horizon.
    # "rainfall last june" was answered with next week, silently, labelled "last june".
    for text in ("last june", "in june", "june"):
        window = resolve(text, now)
        assert window.start.date() < now.date(), f"{text!r} is the past: {window.start}"
        assert window.start.month == 6 and window.span_days >= 28, (text, window)
    # asked before June, "last june" is the year before
    early = resolve("last june", datetime(2026, 3, 4, 9, 0))
    assert early.start.year == 2025, early
    # asked during June, it is June so far - the rest has not happened
    during = resolve("june", datetime(2026, 6, 15, 9, 0))
    assert during.end.date() == datetime(2026, 6, 15).date(), during
    # "may" is a month and the commonest modal in a weather question. Bare, it is the modal.
    assert resolve("may", now).start.date() >= now.date(), "bare 'may' is not the month"
    assert resolve("last may", now).start.month == 5, "'last may' is"

    # --- bug 2: `select_rows` fell back to rows[:7] for any expression its ladder did not
    # know. The archive returned all thirty days of June and twenty-three were dropped.
    june = [{"Date_time": f"2026-06-{day:02d}T00:00:00", "Rainfall": 1.0}
            for day in range(1, 31)]
    picked, _ = select_rows(june, "last june", now)
    assert len(picked) == 30, f"a month is thirty rows, not {len(picked)}"
    # ...and a row outside the window is still excluded
    picked, _ = select_rows(june + [{"Date_time": "2026-07-04T00:00:00"}], "last june", now)
    assert len(picked) == 30, "july is not june"

    # --- the routes, from slots alone
    said = lambda **kw: SimpleNamespace(**{"activity": "NONE", "action": "GET",
                                           "variables": ["RAIN"], "aggregation": "RAW",
                                           "times_normalized": [], "text": "",
                                           "fields": lambda: ["Rainfall"], **kw})
    win = lambda c: resolve(c, now)
    for slots, canonical, wanted in (
            (said(activity="SPRAY"), "tomorrow", "ACTIVITY"),
            (said(action="COMPARE"), "tomorrow", "COMPARE"),
            (said(), "last june", "HISTORICAL"),
            (said(), "tomorrow", "FORECAST")):
        got = profiles.pick(slots, win(canonical), now).route
        assert got == wanted, f"{canonical!r}: routed {got}, wanted {wanted}"

    # --- the parameters: a long look back is reduced, and an advice turn fetches what its
    # rule reads even though nobody said the word
    here = [{"name": "Guntur", "lat": 16.3, "lon": 80.4, "type": "village"}]
    profile = profiles.pick(said(times_normalized=["last june"]), win("last june"), now)
    reduced = params.resolve(said(times_normalized=["last june"]), profile, here, now=now)
    assert reduced.aggregation == "SUM", "a month of rain is a total, not thirty rows"
    assert reduced.assumed, "a reduction nobody asked for has to be admitted"

    sprayed = params.resolve(said(activity="SPRAY", times_normalized=["tomorrow"]),
                             profiles.pick(said(activity="SPRAY"), win("tomorrow"), now),
                             here, now=now)
    assert "Wind_Speed" in sprayed.fields, "a spraying rule cannot answer without wind"
    for key in ("window", "aggregation", "fields", "places", "source"):
        assert sprayed.why.get(key), f"no reason recorded for {key}"
    # --- bug 3: the "today onwards" prefilter exempted the past with a two-name list,
    # {"yesterday", "last week"}. Every other way of naming the past fell through it, so the
    # archive's twenty-one rows for "last 7 days" left as one - the row dated today.
    from datetime import timedelta
    days = [{"Date_time": (now - timedelta(days=d)).strftime("%Y-%m-%dT00:00:00"),
             "Rainfall": 1.0} for d in range(10, -4, -1)]
    for canonical, wanted in (("last 7 days", 7), ("last 2 days", 2), ("yesterday", 1)):
        picked, _ = select_rows(days, canonical, now)
        assert len(picked) == wanted, f"{canonical}: {len(picked)} rows, wanted {wanted}"
    # ...and a forward question still drops the days the feed sent from before today
    picked, _ = select_rows(days, "tomorrow", now)
    assert len(picked) == 1 and picked[0]["Date_time"].startswith(
        (now + timedelta(days=1)).strftime("%Y-%m-%d")), picked

    # --- the time gate: rules place what they can, and nothing else is believed
    from backend.nlu.times import known, mentions_time
    assert known("tomorrow") and known("last 7 days") and known("17:30")
    assert not known("prior days") and not known("last summer") and not known("")
    assert not known("next few days please"), "a sentence is not a canonical form"
    # the trigger for spending a model call at all
    assert mentions_time("can i know yesterday rainfall")
    assert not mentions_time("will it rain in guntur"), "no time words, no call"
    # --- what wins over what, when a profile default meets a spoken word. These lived in a
    # demo() inside params.py, which meant the same rules were asserted in two files.
    prof = lambda **kw: profiles.Profile(kw.pop("route", "ACTIVITY"), **kw)
    spoken = params.resolve(said(times_normalized=["tomorrow"]),
                            prof(window="next 2 days"), here, now=now)
    assert spoken.window == "tomorrow", "what they said beats the profile"
    assert not spoken.assumed, "what they said is not an assumption"
    quiet = params.resolve(said(), prof(window="next 2 days"), here, now=now)
    assert quiet.window == "next 2 days" and quiet.assumed, "an assumption is admitted"

    loud = params.resolve(said(text="total rainfall last week", aggregation="SUM",
                               times_normalized=["last week"]),
                          prof(route="HISTORICAL", aggregation="AVG"), here, now=now)
    assert loud.aggregation == "SUM", "a spoken reduction beats the profile's default"
    # a caller's pin beats both, so three compared columns reduce the same way
    pinned = params.resolve(said(text="total rainfall", aggregation="SUM"),
                            prof(route="COMPARE"), here, now=now, aggregation="RAW")
    assert pinned.aggregation == "RAW", "the caller's pin wins"
    assert sprayed.fields[0] == "Rainfall", "what was asked for still leads the fetch"

    print("routing OK - bare months, whole windows, past rows kept, four routes, time gate")

def main():
    """Every check in this file, in order. Any assertion failure stops it."""
    for check in (check_timewindow, check_places, check_plan, check_windows, check_analysis, check_quality, check_advice, check_render, check_routing,):
        print(f"{check.__name__}:")
        check()
    print("\n9 check(s) passed")


if __name__ == "__main__":
    main()
