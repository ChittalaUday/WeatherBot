"""
The pipeline's unit checks. Run: python tests/test_pipeline_units.py
"""

from _root import ROOT  # noqa: F401 - puts the repo root on sys.path


def check_timewindow():
    """Self-check: the forms Duckling cannot produce, and the row picker.

    Relative wording is not tested here and no longer belongs here - `backend.nlu.duckling`
    resolves it and hands these dates over already absolute. What is left is the gap list.
    """
    from backend.pipeline.timewindow import date, datetime, resolve, select_rows
    now = datetime(2026, 8, 13, 15, 30)          # a Thursday

    # absolute forms, which is what arrives now
    assert resolve("2026-08-14", now).start.date() == date(2026, 8, 14)
    span = resolve("2026-08-14 to 2026-08-16", now)
    assert span.start.date() == date(2026, 8, 14) and span.end.date() == date(2026, 8, 16)
    assert resolve("2026-08-14", now).span_days == 1

    # clock forms, from Duckling or from the tables
    assert resolve("18:45", now).start == datetime(2026, 8, 13, 18, 45)
    assert resolve("06:00", now).start == datetime(2026, 8, 14, 6, 0)    # already gone
    evening = resolve("17:00-21:00", now)
    assert evening.granularity == "hourly" and evening.start.hour == 17

    default = resolve(None, now)
    assert default.label == "next few days" and default.granularity == "daily"

    # --- the gap list: what Duckling has no rule for -------------------------
    early = resolve("early morning", now)
    assert early.granularity == "hourly" and early.start.hour == 4, early
    assert resolve("for all of 2023", now).start.date() == date(2023, 1, 1)
    assert resolve("the last decade", now).span_days > 3600
    # a date range must not collapse to its first date - Duckling returns only the first
    assert resolve("11 jan 2026 and 17 jan 2026", now).span_days == 7
    # the abbreviated-month bug: this must be one day, not the whole of 2026
    assert resolve("11 jun 2026", now).span_days == 1, resolve("11 jun 2026", now)
    assert resolve("in august 2019", now).start.date() == date(2019, 8, 1)
    assert resolve("in august 2019", now).end.date() == date(2019, 8, 31)

    # wording that is nobody's - neither Duckling's nor the gap list's - is refused, not
    # answered with the horizon and a straight face
    assert not resolve("sometime soonish", now).understood

    # ...and the row picker filters by that one window rather than a second calendar
    daily = [{"Date_time": f"2026-08-{d:02d}T00:00:00"} for d in range(12, 22)]
    picked, label = select_rows(daily, "2026-08-14", now)
    assert len(picked) == 1 and picked[0]["Date_time"].startswith("2026-08-14"), picked
    assert select_rows(daily, "2026-08-13 to 2026-08-15", now)[0] == daily[1:4]  # 12 Aug is past
    assert len(select_rows(daily, "", now)[0]) == 7 and label == "14 Aug 2026"
    assert select_rows(daily, "15 august 2026", now)[0][0]["Date_time"].startswith("2026-08-15")
    assert select_rows([], "2026-08-14", now) == ([], "no data")

    hourly = [{"Date_time": f"2026-08-13T{h:02d}:00:00"} for h in range(24)]
    assert len(select_rows(hourly, "17:00-21:00", now)[0]) == 5        # 17:00-21:00

    print("timewindow demo OK")
    for wording in ("2026-08-14", "17:00-21:00", "in august 2019", "early morning",
                    "the last decade", "11 jun 2026", None):
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
    from backend.pipeline.plan import (
        MAX_ROWS,
        MAX_SPAN_DAYS,
        Resolution,
        Source,
        Verdict,
        datetime,
        plan,
    )
    now = datetime(2026, 8, 13, 15, 30)
    here = [{"name": "Guntur", "lat": 16.3, "lon": 80.4}]
    two = here + [{"name": "Vizag", "lat": 17.7, "lon": 83.2}]
    # archive-backed cases are asserted with the archive assumed up - routing is the subject
    up = lambda **kw: plan(**{"archive": True, "now": now, **kw})

    # Inside the hourly feed's 24h reach -> hourly, so the answer can say *when*. These are
    # what Duckling hands over for "now", "today", "tonight" and "this evening".
    for wording in ("15:30", "2026-08-13", "2026-08-13T18:00 to 2026-08-13T23:59"):
        assert up(times_normalized=[wording], places=here).source is Source.GFS_HOURLY, wording

    # ...and outside it -> daily. "tomorrow" at 15:30 ends 32h out, past /hrlydata's reach;
    # hourly would drop tomorrow evening and make the answer depend on the clock.
    for wording in ("2026-08-14", "2026-08-14 to 2026-08-20", "2026-08-15"):
        p = up(times_normalized=[wording], places=here)
        assert p.source is Source.GFS_DAILY and p.resolution is Resolution.DAILY, (wording, p)

    # The rule is the reach, not the word: the same wording flips as the clock moves.
    morning = "2026-08-14T06:00 to 2026-08-14T11:59"
    assert plan(times_normalized=[morning], places=here, archive=True,
                now=datetime(2026, 8, 13, 6, 0)).source is Source.GFS_DAILY
    assert plan(times_normalized=[morning], places=here, archive=True,
                now=datetime(2026, 8, 13, 22, 0)).source is Source.GFS_HOURLY

    assert up(times_normalized=["2026-08-12"], places=here).source is Source.GFS_HISTORICAL
    assert up(times_normalized=["2026-08-12"], places=here,
              level="district").source is Source.POSTGRES_AGG          # an area, not a point
    assert up(times_normalized=["2026-08-03 to 2026-08-09"],
              places=two).source is Source.ZARR_BULK

    # An old window inside the cap still reaches the archive, and still says which dates.
    p = up(times_normalized=["2019-08-01 to 2019-08-10"], places=here)
    assert p.source is Source.ZARR_POINT and p.start.startswith("2019-08-01"), p
    assert p.end.startswith("2019-08-10"), p.end
    assert up(times_normalized=["on 15 august 2023"], places=here).span_days == 1

    # --- the span cap: a latency budget, refused before a source is chosen ---------
    # Every aggregation runs over every row that comes back, so the wide windows the archive
    # would happily serve are declined - and the refusal says what would work instead.
    for wide in ("in august 2019", "from 2010 to 2025", "for all of 2023",
                 "2026-05-01 to 2026-06-30"):
        p = up(times_normalized=[wide], places=here)
        assert p.verdict is Verdict.REJECT, (wide, p)
        assert str(MAX_SPAN_DAYS) in p.reason and p.offer, (wide, p)
    # ...and the day either side of the cap decides it, not the source or the place count
    inside = f"2026-08-{14 - MAX_SPAN_DAYS + 13:02d} to 2026-08-13"   # exactly the cap, in the past
    assert up(times_normalized=[inside], places=here).verdict is Verdict.EXECUTE
    three = two + [{"name": "Nellore", "lat": 14.4, "lon": 80.0}]
    assert up(times_normalized=[inside], places=three).verdict is not Verdict.REJECT
    p = up(times_normalized=["tomorrow"], places=[])
    assert p.verdict is Verdict.ASK and "place" in p.reason, p

    # the archive holds six measurements; asking it for soil moisture has to be visible
    p = up(times_normalized=["2019-08-01 to 2019-08-10"], places=here,
           fields=["Rainfall", "Soilm10"])
    assert p.source is Source.ZARR_POINT and p.unservable == ["Soilm10"], p
    assert not up(times_normalized=["2026-08-14"], places=here, fields=["Soilm10"]).unservable

    # ...and with the archive down, an old date is refused up front rather than timing out
    down = plan(times_normalized=["2019-08-01 to 2019-08-10"], places=here, now=now,
                archive=False)
    assert down.verdict is Verdict.REJECT and "internal network" in down.reason, down

    # a date within the 7-day lookback is served by the forecast API, archive or no archive
    recent = plan(times_normalized=["11 august 2026"], places=here, archive=False,
                  now=datetime(2026, 8, 14, 13, 0))
    assert recent.verdict is Verdict.EXECUTE and recent.source is Source.GFS_HISTORICAL, recent

    # Every wording is either served or refused for a reason that names the cap - never
    # planned into a fetch nobody asked for. A wide window is a redirection, not a dead end,
    # so a refusal has to carry an offer.
    for wording in ("in 2017", "for all of 2023", "every year since 2018", "in the last decade",
                    "over the past 6 months", "on 12/06/2021", "2023-08-15", "march 2022",
                    "11 jan 2026 and 17 jan 2026"):
        got = up(times_normalized=[wording], places=here)
        if got.verdict is Verdict.REJECT:
            assert got.span_days > MAX_SPAN_DAYS and got.offer, (wording, got)
            continue
        assert got.verdict in (Verdict.EXECUTE, Verdict.COARSEN), (wording, got)
        assert got.rows <= MAX_ROWS, (wording, got.rows)
        assert got.span_days <= MAX_SPAN_DAYS, (wording, got.span_days)

    print("plan demo OK")
    for wording, where in (("2026-08-14", here), ("2026-08-12", here),
                           ("2019-08-01 to 2019-08-10", here), ("from 2010 to 2025", here),
                           ("over the last 5 years", here), ("2026-09-01 to 2026-09-30", here),
                           ("for all of 2023", here), ("for all of 2023", three)):
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

    # 12mm either way, and only one of these two is a wet morning
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

    # A missing reading breaks the run rather than being assumed dry: `or 0` reads None as
    # zero, which for rainfall means "dry" and would quietly bridge the gap.
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

    here = [{"name": "Guntur"}]
    chart = build_chart([rows], here, ["Rainfall"], hourly=False)
    assert chart and chart["type"] == "line" and len(chart["series"][0]["points"]) == 5, chart
    # one place, one reading - no series and no comparison, so there is nothing to draw
    assert build_chart([rows[:1]], here, ["Rainfall"], False) is None

    # --- the shape follows the data, not a default ---------------------------
    from backend.pipeline.analysis import pick_chart

    days = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": r,
             "Tmin": t - 8, "Tmax": t, "Tavg": t - 4}
            for d, r, t in zip(range(1, 8), [0, 3.2, 5.1, 0, 12.4, 8.0, 2.2],
                               [31, 30, 29, 34, 28, 27, 33])]
    hours = [{"Date_time": f"2026-08-{d:02d}T{h:02d}:00:00", "Rainfall": (h % 7) * 0.4}
             for d in range(1, 4) for h in range(24)]
    winds = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Wind_Direction": v}
             for d, v in zip(range(1, 9), [10, 20, 350, 95, 100, 15, 200, 5])]

    # a bearing is circular, so it is a rose whatever statistic was asked - a line through
    # 350 and 10 descends through south to join two readings 20 degrees apart
    assert pick_chart(["Wind_Direction"], [], False, 1, 8, 8) == "rose"
    assert pick_chart(["Rainfall"], ["CUMULATIVE"], False, 1, 10, 10) == "area"
    assert pick_chart(["Tmin", "Tmax", "Tavg"], [], False, 1, 7, 7) == "band"
    assert pick_chart(["Rainfall", "Tavg"], [], False, 1, 7, 7) == "combo"
    assert pick_chart(["Rainfall"], [], True, 1, 72, 3) == "heatmap"
    assert pick_chart(["Rainfall"], [], False, 3, 5, 5) == "bar"
    assert pick_chart(["Rainfall"], [], False, 1, 20, 20) == "line"

    # ...and every shape builds a payload its renderer can actually read
    band = build_chart([days], here, ["Tmin", "Tmax", "Tavg"], False, [])
    assert band and band["type"] == "band" and band["points"][0]["lo"] < band["points"][0]["hi"]
    combo = build_chart([days], here, ["Rainfall", "Tavg"], False, [])
    assert combo and combo["bars"]["points"] and combo["line"]["points"], combo
    rose = build_chart([winds], here, ["Wind_Direction"], False, [])
    assert rose and len(rose["buckets"]) == 16, rose
    assert sum(b["share"] for b in rose["buckets"]) > 99, rose
    heat = build_chart([hours], here, ["Rainfall"], True, [])
    assert heat and len(heat["days"]) == 3 and len(heat["cells"]) == 72, heat
    area = build_chart([days], here, ["Rainfall"], False, ["CUMULATIVE"])
    assert area and area["series"][0]["points"][-1]["v"] == 30.9, area   # the running total
    # three places for one day is three bars, which is exactly what a comparison wants
    three = [{"name": n} for n in ("Guntur", "Vizag", "Nellore")]
    bars = build_chart([days[:1], days[:1], days[:1]], three, ["Rainfall"], False, [])
    assert bars and bars["type"] == "bar" and len(bars["series"]) == 3, bars

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
    # 12mm either way; only the afternoon case leaves a clear morning. A sum says no to both.
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
    # A sum only grows, so "today" and "this week" used to differ for identical weather.
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

    "Min temp: A 21.3 · B 20.5. Highest: A." was true and unreadable - it reads as "A is the
    lower one", and the chat duly said the opposite of the data.
    """
    from backend.pipeline.render import build_table, format_value, summarize, summary_stat
    places = [{"name": "Hyderabad"}, {"name": "Vijawada"}]
    rows = [[{"Date_time": "2026-08-14T00:00:00", "Tmin": 21.3}],
            [{"Date_time": "2026-08-14T00:00:00", "Tmin": 20.5}]]
    said = summarize("COMPARE", rows, ["Tmin"], places, "tomorrow")
    assert said.startswith("Hyderabad has the higher"), said
    assert "21.3" in said and "Vijawada 20.5" in said, said

    # the same week of rain is a mean under RAW and a total under SUM, never by accident
    assert summary_stat("Rainfall", [1.0, 2.0, 3.0], "SUM") == (6.0, "total")
    assert summary_stat("Rainfall", [1.0, 2.0, 3.0], "RAW") == (2.0, "average")
    assert summary_stat("Tmax", [10.0, 20.0], "SUM") == (15.0, "average")   # never additive
    assert summary_stat("Rainfall", [], "SUM") == (0.0, "")

    # ...and the sentence follows it: the total appears only when the question said "total"
    week = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": v}
            for d, v in enumerate([0.2, 6.0, 0.0, 0.1, 3.0], start=14)]
    raw = summarize("GET", week, ["Rainfall"], places[:1], "this week")
    assert "total" not in raw and "rain on 2 of 5 readings" in raw, raw
    assert "up to 6.0mm" in raw, raw
    # the total is said once, by `analysis.apply_aggregation`; this sentence follows it
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
    """The route, the profile and the parameters - and the two bugs they surfaced, both the
    same shape: an unrecognised expression falling through to a default that looked like an
    answer."""
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

    # --- bug 2: `select_rows` fell back to rows[:7] for anything its ladder did not know,
    # so all thirty days of June came back as seven.
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
            (said(activity="SPRAY"), "2026-08-27", "ACTIVITY"),
            (said(action="COMPARE"), "2026-08-27", "COMPARE"),
            (said(), "last june", "HISTORICAL"),
            (said(), "2026-08-27", "FORECAST")):
        got = profiles.pick(slots, win(canonical), now).route
        assert got == wanted, f"{canonical!r}: routed {got}, wanted {wanted}"

    # --- the parameters: a long look back is reduced, and an advice turn fetches what its
    # rule reads even though nobody said the word
    here = [{"name": "Guntur", "lat": 16.3, "lon": 80.4, "type": "village"}]
    profile = profiles.pick(said(times_normalized=["last june"]), win("last june"), now)
    reduced = params.resolve(said(times_normalized=["last june"]), profile, here, now=now)
    assert reduced.aggregation == "SUM", "a month of rain is a total, not thirty rows"
    assert reduced.assumed, "a reduction nobody asked for has to be admitted"

    sprayed = params.resolve(said(activity="SPRAY", times_normalized=["2026-08-27"]),
                             profiles.pick(said(activity="SPRAY"), win("2026-08-27"), now),
                             here, now=now)
    assert "Wind_Speed" in sprayed.fields, "a spraying rule cannot answer without wind"
    for key in ("window", "aggregation", "fields", "places", "source"):
        assert sprayed.why.get(key), f"no reason recorded for {key}"
    # --- bug 3: the "today onwards" prefilter exempted the past with a two-name list, so
    # the archive's twenty-one rows for "last 7 days" left as one - the row dated today.
    from datetime import timedelta
    days = [{"Date_time": (now - timedelta(days=d)).strftime("%Y-%m-%dT00:00:00"),
             "Rainfall": 1.0} for d in range(10, -4, -1)]
    iso = lambda d: (now - timedelta(days=d)).strftime("%Y-%m-%d")
    for canonical, wanted in ((f"{iso(7)} to {iso(1)}", 7), (f"{iso(2)} to {iso(1)}", 2),
                              (iso(1), 1)):
        picked, _ = select_rows(days, canonical, now)
        assert len(picked) == wanted, f"{canonical}: {len(picked)} rows, wanted {wanted}"
    # ...and a forward question still drops the days the feed sent from before today
    picked, _ = select_rows(days, (now + timedelta(days=1)).strftime("%Y-%m-%d"), now)
    assert len(picked) == 1 and picked[0]["Date_time"].startswith(
        (now + timedelta(days=1)).strftime("%Y-%m-%d")), picked

    # --- the time gate: rules place what they can, and nothing else is believed
    from backend.nlu.times import known, mentions_time
    # `known` gates what `resolve` recognises, and that is now absolute forms plus the gap
    # list. "tomorrow" is Duckling's to place, not this module's.
    assert known("2026-08-27") and known("17:30") and known("early morning")
    assert not known("prior days") and not known("last summer") and not known("")
    assert not known("tomorrow"), "relative wording is resolved upstream, not here"
    # the trigger for spending a model call at all
    assert mentions_time("can i know yesterday rainfall")
    assert not mentions_time("will it rain in guntur"), "no time words, no call"
    # --- what wins over what, when a profile default meets a spoken word
    prof = lambda **kw: profiles.Profile(kw.pop("route", "ACTIVITY"), **kw)
    spoken = params.resolve(said(times_normalized=["2026-08-27"]),
                            prof(window="next 2 days"), here, now=now)
    assert spoken.window == "2026-08-27", "what they said beats the profile"
    assert not spoken.assumed, "what they said is not an assumption"
    quiet = params.resolve(said(), prof(window="next 2 days"), here, now=now)
    assert quiet.window == "next 2 days" and quiet.assumed, "an assumption is admitted"

    loud = params.resolve(said(text="total rainfall last week", aggregation="SUM",
                               times_normalized=["2026-08-17 to 2026-08-23"]),
                          prof(route="HISTORICAL", aggregation="AVG"), here, now=now)
    assert loud.aggregation == "SUM", "a spoken reduction beats the profile's default"
    # a caller's pin beats both, so three compared columns reduce the same way
    pinned = params.resolve(said(text="total rainfall", aggregation="SUM"),
                            prof(route="COMPARE"), here, now=now, aggregation="RAW")
    assert pinned.aggregation == "RAW", "the caller's pin wins"
    assert sprayed.fields[0] == "Rainfall", "what was asked for still leads the fetch"

    print("routing OK - bare months, whole windows, past rows kept, four routes, time gate")

def check_presentation():
    """Self-check: what goes on screen. Offline - the model's pick is an input here, so the
    rule, the correction of an impossible pick, and the wire shape are all checkable without
    a model being up."""
    from backend.pipeline import Answer
    from backend.pipeline.render import presentation

    def answer(rows=0, columns=0, points=0, **kw):
        a = Answer(places=[{"name": "Guntur"}])
        a.table = {"columns": [{"key": "time"}] * (columns + 1),
                   "rows": [{"time": str(i)} for i in range(rows)]}
        a.chart = ({"series": [{"points": [{}] * points}]}) if points else None
        for key, value in kw.items():
            setattr(a, key, value)
        return a

    # the rule, with nothing chosen for it
    assert presentation(answer(rows=7, points=7))["detail"] == "chart", "a series is a shape"
    assert presentation(answer(rows=7, columns=4))["detail"] == "table", "a grid is a grid"
    assert presentation(answer(rows=2, columns=2))["detail"] == "none", "the sentence said it"
    assert presentation(answer(rows=7, points=7, advice=object()))["detail"] == "none", \
        "a verdict is a one-line answer, whatever is under it"

    # the model's pick, taken - and corrected when the payload cannot fill it
    picked = presentation(answer(rows=7, points=7), "table", "scan the values", "gemma")
    assert picked["detail"] == "table" and picked["decided_by"] == "gemma", picked
    impossible = presentation(answer(rows=7), "chart", "", "gemma")
    assert impossible["detail"] == "table" and impossible["decided_by"] == "rule", impossible
    assert presentation(answer(), "table", "", "gemma")["detail"] == "none", "no rows, no table"
    assert presentation(answer(rows=7, points=7), "sideways")["decided_by"] == "rule", \
        "a view that does not exist is not a choice"

    # the wire shape: what is not open is still offered, never silently dropped
    offered = presentation(answer(rows=7, points=7), "chart", "", "gemma")
    assert offered["chart"] == "open" and offered["table"] == "available", offered
    assert offered["rows"] == 7 and offered["columns"] == 0, offered
    print("  rule -> chart/table/none, model's pick honoured, impossible pick downgraded")
    print("presentation OK - one decision, and it is never allowed to be unrenderable")


def check_aggregations():
    """Self-check: every statistic computes, and none of them answers a question the data
    cannot support. The refusals are the point - a number that means nothing is worse than
    no number, because it is indistinguishable from one that does."""
    from backend.pipeline.aggregate import COMPUTE, compute
    from src.v4.schema import Aggregation, Variable, supports

    rain = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": v}
            for d, v in zip(range(1, 11), [0, 3.2, 5.1, 0, 0, 12.4, 8.0, 0, 0.1, 2.2])]

    # every statistic in the table answers for a variable that supports it
    answered = {a.value: compute(rain, "Rainfall", "RAIN", a) for a in COMPUTE}
    for name, got in answered.items():
        if supports(Variable.RAIN, name):
            assert got and got["text"], f"{name} computed nothing for RAIN"

    def figure(name: str) -> dict:
        """One answered statistic, or a failure that names which one went missing."""
        got = answered[name]
        assert got is not None, f"{name} computed nothing for RAIN"
        return got

    # the figures themselves, against a series small enough to check by hand
    assert figure("SUM")["value"] == 31.0, figure("SUM")
    assert figure("MAX")["value"] == 12.4 and figure("MAX")["at"].startswith("2026-08-06")
    assert figure("MIN")["value"] == 0.0
    assert figure("COUNT")["value"] == 5, figure("COUNT")              # readings >= 0.2mm
    assert figure("RUN_COUNT")["value"] == 3, figure("RUN_COUNT")      # 2-3, 6-7, 9-10
    assert figure("FREQUENCY")["value"] == 50.0
    assert figure("LONGEST_RUN")["value"] == 2, figure("LONGEST_RUN")
    assert len(figure("CUMULATIVE")["series"]) == 10
    assert figure("CUMULATIVE")["series"][-1]["v"] == 31.0

    # what the data cannot support is refused, not approximated
    humid = [{"Date_time": f"2026-08-0{d}T00:00:00", "RH": v}
             for d, v in zip(range(1, 6), [80, 85, 90, 70, 75])]
    assert compute(humid, "RH", "HUMIDITY", "AVG"), "an average of humidity is fine"
    for refused in ("SUM", "LONGEST_RUN", "MODE", "INTENSITY"):
        assert compute(humid, "RH", "HUMIDITY", refused) is None, f"{refused} is not humidity's"

    # a bearing is circular, so every linear statistic on it is wrong rather than approximate
    wind = [{"Date_time": f"2026-08-0{d}T00:00:00", "Wind_Direction": v}
            for d, v in zip(range(1, 9), [10, 20, 350, 95, 100, 15, 200, 5])]
    dominant = compute(wind, "Wind_Direction", "WIND", "MODE")
    spread = compute(wind, "Wind_Direction", "WIND", "DISTRIBUTION")
    assert dominant and dominant["text"].endswith("of readings)"), dominant
    # all sixteen points, empty ones included - the empty petals are what make it a rose
    assert spread and len(spread["buckets"]) == 16, spread
    assert len([b for b in spread["buckets"] if b["share"]]) == 4, spread
    for wrong in ("AVG", "MAX", "STDDEV", "MEDIAN"):
        assert compute(wind, "Wind_Direction", "WIND", wrong) is None, wrong

    # a sentinel is not a reading: -999 must never win a MIN or drag down a mean
    dirty = rain + [{"Date_time": "2026-08-11T00:00:00", "Rainfall": -999}]
    assert (compute(dirty, "Rainfall", "RAIN", "MIN") or {}).get("value") == 0.0
    assert (compute(dirty, "Rainfall", "RAIN", "SUM") or {}).get("value") == 31.0
    assert compute([], "Rainfall", "RAIN", "SUM") is None

    # --- several statistics in one turn ------------------------------------
    from backend.pipeline.analysis import pair_up, reduce_all

    # one statistic over several variables, and several over one - the two shapes people say
    assert pair_up(["TEMPERATURE", "RAIN"], ["PEAK_DATE"]) == [
        ("TEMPERATURE", "PEAK_DATE"), ("RAIN", "PEAK_DATE")]
    assert pair_up(["RAIN"], ["SUM", "AVG"]) == [("RAIN", "SUM"), ("RAIN", "AVG")]
    # a combination the variable cannot support is dropped from the pairing, not computed
    assert ("TEMPERATURE", "SUM") not in pair_up(["RAIN", "TEMPERATURE"], ["SUM", "AVG"])
    assert pair_up(["HUMIDITY"], ["SUM"]) == [] and pair_up(["RAIN"], ["RAW"]) == []

    mixed = [{"Date_time": f"2026-08-{d:02d}T00:00:00", "Rainfall": r,
              "Tmax": t, "Tmin": t - 8, "Tavg": t - 4}
             for d, r, t in zip(range(1, 11), [0, 3.2, 5.1, 0, 0, 12.4, 8.0, 0, 0.1, 2.2],
                                [31, 30, 29, 34, 36, 28, 27, 33, 35, 32])]
    have = ["Tmin", "Tmax", "Tavg", "Rainfall"]

    # "the hottest and the rainiest day" - one statistic, two variables, two answers
    both = reduce_all(mixed, ["TEMPERATURE", "RAIN"], ["PEAK_DATE"], have)
    assert len(both) == 2, both
    # and each reads the column its statistic actually wants: the high, not whichever
    # temperature column the fetch happened to put first
    assert both[0]["value"] == 36.0, both[0]
    assert both[1]["value"] == 12.4, both[1]
    assert reduce_all(mixed, ["TEMPERATURE"], ["LOW_DATE"], have)[0].get("value") == 19.0
    assert reduce_all(mixed, ["TEMPERATURE"], ["AVG"], have)[0].get("value") == 27.5   # Tavg

    # --- the column gate: a variable can accumulate and one of its columns still cannot ---
    from backend.pipeline.analysis import column_named, confirm_aggregation

    june = [{"Date_time": f"2025-06-{d:02d}T00:00:00", "DayLength": 12.0 + d * 0.01,
             "SunSD": 6.0} for d in range(1, 31)]
    # SUNSHINE accumulates - hours of sun add up over a month - but day length does not.
    # Thirty days of daylight totalling 158 hours is not a reading of anything.
    assert compute(june, "SunSD", "SUNSHINE", "SUM"), "hours of sun do add up"
    assert compute(june, "DayLength", "SUNSHINE", "SUM") is None, "day length does not"
    peak = compute(june, "DayLength", "SUNSHINE", "PEAK_DATE")
    assert peak and peak["at"].startswith("2025-06-30"), peak

    # the wording names a column its variable would not otherwise pick
    assert column_named("the largest day in june", ["SunSD", "DayLength"]) == "DayLength"
    assert column_named("sunshine in june", ["SunSD", "DayLength"]) == ""
    # ...and a superlative names a date, whatever the model said. Without this the model's
    # RAW stood, the profile turned it into a total, and "the largest day" was answered with
    # every day's length added together.
    assert confirm_aggregation("the largest day in june 2025", "RAW") == "PEAK_DATE"
    assert confirm_aggregation("total rainfall in june", "RAW") == "SUM"
    # "sum" lives inside "summarize", and a substring match made every summary a total
    assert confirm_aggregation("summarize the weather in Hyderabad", "SUM") == "RAW"
    # "which day" supported both directions, so it confirmed whichever the model guessed
    assert confirm_aggregation("which day is the hottest", "LOW_DATE") == "RAW"
    assert confirm_aggregation("which day is the hottest", "PEAK_DATE") == "PEAK_DATE"

    print(f"  {len(COMPUTE)} statistics, {len(Aggregation)} labels, refusals hold")
    print("aggregations OK - every figure computed, combined, nothing approximated")


def main():
    """Every check in this file, in order. Any assertion failure stops it."""
    for check in (check_timewindow, check_places, check_plan, check_windows, check_analysis, check_quality, check_advice, check_render, check_routing, check_presentation,
                  check_aggregations,):
        print(f"{check.__name__}:")
        check()
    print("\n11 check(s) passed")


if __name__ == "__main__":
    main()
