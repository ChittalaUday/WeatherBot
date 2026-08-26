"""
Advice engine - activity plus numbers, out comes a verdict, a reason, and a time to do it.

    python -m backend.pipeline.advice            # self-check

The model says what is being decided; the rule here reads the fields that decision depends on
and returns YES / NO / CAUTION with the numbers that produced it. Nothing is generated: every
sentence a rule emits names a measured quantity, because "do not spray" is only useful next to
"35mm expected Friday".

Two kinds of question, and they are not the same shape:

  **When can I do this?** - spraying, drying washing, travelling, harvesting, going out. Rain
  does not fall for a whole day; it falls between two and four. These are answered by
  `backend.pipeline.windows`, which finds the stretches during which the conditions actually
  hold, and the verdict is about the best of those stretches - so the answer can be "not at
  two, but you have until noon" instead of a flat no built from a day's total.

  **Is the ground ready?** - irrigating, sowing, fertilising, what to wear. These are about
  state and about how much water is coming, where an accumulated total is the right quantity
  and always was. They stay accumulated, and each one says over what period.

The rules are two-sided on purpose. FERTILIZE says no to a downpour *and* no to bone-dry soil;
SOW wants the rain that HARVEST is trying to avoid. That opposition is the whole reason these
are separate activities rather than one FARMING label with a sub-activity.

Thresholds are the tunable part and several of them are literature defaults rather than
measurements - they carry a `ponytail:` note where that is true.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from backend.pipeline.quality import assess, values
from backend.pipeline.windows import (
    Window,
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
from backend.pipeline.windows import _stamp as _when

YES, NO, CAUTION, UNKNOWN = "YES", "NO", "CAUTION", "UNKNOWN"

# ponytail: soil-moisture bands are loam defaults from the literature, not measurements from
# these fields. Calibrate against real plots before anyone acts on IRRIGATE or SOW.
SOIL_DRY, SOIL_WET = 0.15, 0.30
SPRAY_WIND_MIN, SPRAY_WIND_MAX = 1.0, 4.5      # below: inversion drift. above: spray drift.
HEAVY_RAIN_48H = 25.0                          # leaches fertiliser
USEFUL_RAIN_48H = 2.0                          # enough to carry it in

# mm in ONE reading that counts as "it is raining". Under this is not much rain: a coat is for
# rain you would shelter from, and 0.6mm in an hour is not that.
WET = 1.0
# mm in one reading that is enough to interfere with something delicate - wet spray leaf,
# washing on a line. Lower than WET, because "not enough to shelter from" and "not enough to
# ruin what you are doing" are different questions.
DAMP = 0.2

# Conditions, named once and reused. A rule composes these; nothing writes a lambda that reads
# a column directly, so "a missing reading is not a suitable reading" holds everywhere.
DRY = below("Rainfall", WET)
NOT_DAMP = below("Rainfall", DAMP)
SPRAY_WIND = between("Wind_Speed", SPRAY_WIND_MIN, SPRAY_WIND_MAX)
BEARABLE = below("Tmax", 38.0)
DRYING_AIR = below("RH", 85.0)


@dataclass
class Advice:
    verdict: str
    headline: str
    reasons: list = field(default_factory=list)
    evidence: dict = field(default_factory=dict)
    activity: str = ""
    sub_activity: str = ""
    caveats: list = field(default_factory=list)
    # the stretch this verdict points at, when the answer is "yes, then"
    window: str = ""


@dataclass(frozen=True)
class Timed:
    """An activity you do *during* a stretch of good conditions.

    `during` is what one reading has to look like. `hours` is how long a usable stretch has to
    be - twenty minutes of calm between showers is not a spraying window. `settle_hours` is how
    long it then has to stay rain-free afterwards, which is what separates spraying (four
    hours, or it washes off) from walking to the shop (none).
    """

    needs: tuple
    during: object
    hours: float
    verb: str                 # "spray", "dry the washing" - goes in the headline
    blocker: str              # why a reading fails: "rain", "wind", "rain or heat"
    settle_hours: float = 0.0


# The activities that are a question about *when*. Minimum durations are the tunable part.
# ponytail: `hours` are working-practice estimates, not measurements. A grower who says two
# hours is not enough to spray their plot is right and this is where to change it.
TIMED = {
    "SPRAY": Timed(("Wind_Speed", "Rainfall"), every(NOT_DAMP, SPRAY_WIND), hours=2,
                   verb="spray", blocker="rain or the wrong wind", settle_hours=4),
    "DRYING": Timed(("Rainfall", "RH"), every(NOT_DAMP, DRYING_AIR), hours=4,
                    verb="get things dry", blocker="rain or damp air"),
    "OUTDOOR_ACTIVITY": Timed(("Rainfall", "Tmax"), every(DRY, BEARABLE), hours=2,
                              verb="be outside", blocker="rain or heat"),
    "TRAVEL": Timed(("Rainfall", "Wind_Speed"), every(DRY, below("Wind_Speed", 14.0)), hours=1,
                    verb="travel", blocker="rain or strong wind"),
    "HARVEST": Timed(("Rainfall", "RH"), DRY, hours=72,        # three dry days in a row
                     verb="harvest", blocker="rain"),
}

# The activities that are a question about *state* - the ground, the air, what is coming.
STATE_NEEDS = {
    "RAIN_PROTECTION": ["Rainfall"],
    "SUN_PROTECTION": ["SunSD"],
    "CLOTHING": ["Tmin", "Tmax"],
    "FERTILIZE": ["Rainfall"],
    "IRRIGATE": ["Soilm10", "Rainfall"],
    "SOW": ["Soilm10", "Rainfall"],
}

# The window each activity means when the user names none. "Can I go out?" is about the next
# few hours, not the next week. Fertiliser and spray look further because their rules read
# rain that has not fallen yet.
DEFAULT_WINDOW = {
    "RAIN_PROTECTION": "today", "SUN_PROTECTION": "today", "CLOTHING": "today",
    "OUTDOOR_ACTIVITY": "today", "TRAVEL": "today", "DRYING": "today",
    "SPRAY": "next 2 days", "FERTILIZE": "next 3 days", "IRRIGATE": "next 3 days",
    "HARVEST": "next 5 days", "SOW": "next 7 days",
}

NEEDS = {**STATE_NEEDS, **{name: list(spec.needs) for name, spec in TIMED.items()}}


def _total(rows, f):
    return round(sum(values(rows, f)), 1)


def _peak(rows, f):
    vals = values(rows, f)
    return round(max(vals), 1) if vals else None


def _mean(rows, f):
    vals = values(rows, f)
    return round(sum(vals) / len(vals), 1) if vals else None


def _wet_readings(rows):
    return sum(1 for v in values(rows, "Rainfall") if v >= WET)


def _spell(hours: float, hourly: bool) -> str:
    """A duration said the way a person says it."""
    if not hourly:
        days = round(hours / 24)
        return f"{days} day{'s' if days != 1 else ''}"
    if hours < 1:
        return f"{round(hours * 60)} minutes"
    whole = round(hours)
    return f"{whole} hour{'s' if whole != 1 else ''}"


def _horizon(rows: list[dict], hourly: bool):
    """The moment the forecast stops telling us anything."""
    stamps = [s for s in (_when(r) for r in rows or []) if s]
    return max(stamps) + timedelta(hours=spacing_hours(rows, hourly)) if stamps else None


def _settles(window: Window, shelter: list[Window], doing_hours: float, settle_hours: float,
             horizon) -> tuple:
    """Can the job start inside this window and still get its dry hours afterwards?

    Spraying is what this exists for: two calm hours are no use if it rains an hour later,
    because the spray washes off the leaf before it has done anything.

    What is measured is the *job*, not the window. A ten-hour clear spell against a two-hour
    job needs six clear hours from whenever you start - not fourteen - and requiring the whole
    spell plus the settle time refused a perfectly good day for being too long.

    Returns `(ok, only_because_the_data_ends)`. The forecast running out is not rain: a clear
    spell that reaches the end of what we know is accepted, and says so, because "it will wash
    off" about hours nobody has a reading for is an invented forecast.
    """
    if settle_hours <= 0:
        return True, False
    needed = timedelta(hours=doing_hours + settle_hours)
    for cover in shelter:
        start = max(window.start, cover.start)
        if start + timedelta(hours=doing_hours) > window.end:
            continue                       # no room to do the job inside the clear spell
        if cover.end >= start + needed:
            return True, False
        if horizon is not None and cover.end >= horizon:
            return True, True              # clear as far as we can see, which is not far
    return False, False


def _timed(spec: Timed, rows: list[dict], hourly: bool, sub: str) -> Advice:
    """The verdict for a "when can I do this" activity: the best usable stretch, or why none."""
    usable = runs(rows, spec.during, hourly=hourly)
    shelter = runs(rows, NOT_DAMP, hourly=hourly) if spec.settle_hours else []
    # two separate reasons a window can fail: too short to work in, or long enough but the
    # rain arrives before the job has settled. They need different sentences.
    horizon = _horizon(rows, hourly)
    long_by_time = [w for w in usable if w.hours >= spec.hours]
    long_enough, unconfirmed = [], False
    for candidate in long_by_time:
        ok, ran_out = _settles(candidate, shelter, spec.hours, spec.settle_hours, horizon)
        if ok:
            long_enough.append(candidate)
            unconfirmed = unconfirmed or ran_out
    best = longest(long_enough) or longest(usable)
    share = coverage(usable, rows, hourly)
    wanted = _spell(spec.hours, hourly)

    ev = {"usable_windows": len(usable), "share_of_period": share,
          "longest_clear": best.describe() if best else None,
          "hours_needed": spec.hours, "peak_mm": _peak(rows, "Rainfall"),
          "wet_readings": f"{_wet_readings(rows)} of {len(rows)}"}
    if sub:
        ev["sub_activity"] = sub

    if long_enough and (pick := longest(long_enough)) is not None:
        thin = (["the forecast does not reach far enough to confirm it stays dry afterwards"]
                if unconfirmed else [])
        if share >= 0.99:
            return Advice(YES, "Yes - clear right through, so any time works.",
                          [f"nothing to stop you {spec.verb}ing across the whole period"],
                          ev, window=pick.describe(), caveats=thin)
        return Advice(YES, f"Yes, but pick your moment - {pick.describe()} is your window.",
                      [f"{spec.blocker} outside that"], ev, window=pick.describe(),
                      caveats=thin)

    if long_by_time and spec.settle_hours and (early := longest(long_by_time)) is not None:
        # long enough to do the job in, but the rain arrives before it has settled. Checked
        # before the length branches below, or a three-hour window against a two-hour job
        # comes back "tight", which is the wrong complaint entirely.
        return Advice(NO,
                      f"No - {early.label()} would work, but rain follows too soon after and "
                      f"it will wash off.",
                      [f"needs {_spell(spec.settle_hours, hourly)} dry afterwards"], ev,
                      window=early.describe())

    if fragmented(usable, rows, spec.hours, hourly):
        best_str = _spell(best.hours, hourly) if best else ""
        return Advice(CAUTION,
                      f"It keeps breaking up - {len(usable)} clear "
                      f"spell{'s' if len(usable) != 1 else ''} but the longest is only "
                      f"{best_str}, and you need {wanted}.",
                      [f"{spec.blocker} on and off through the period"], ev,
                      window=best.describe() if best else "")

    if best:
        return Advice(CAUTION,
                      f"Tight - only {_spell(best.hours, hourly)} clear ({best.label()}), "
                      f"against the {wanted} you want.",
                      [f"{spec.blocker} either side of it"], ev, window=best.describe())

    return Advice(NO, f"No - {spec.blocker} right through the period, with no clear spell at all.",
                  [f"peak {_peak(rows, 'Rainfall')}mm in a reading"] if _peak(rows, "Rainfall")
                  else [], ev)


# --- state rules: about the ground and what is coming, not about when -------------

def _rain_protection(rows, sub, hourly):
    """Decided reading by reading, and told with the timing attached.

    A sum answers "how much water fell across the period", which is the wrong question for a
    coat. Seven hours of 0.4mm drizzle add up to 2.8mm and used to come back "take it - 2.8mm
    expected", which nobody standing outside would recognise; and because the sum only grows,
    the same weather scored worse the longer a period you asked about. What decides a coat is
    whether any single stretch is wet enough to want one - and if it is, when.
    """
    rain = values(rows, "Rainfall")
    peak = round(max(rain), 1) if rain else 0.0
    wet_spells = runs(rows, lambda row: not DRY(row) and reading(row, "Rainfall") is not None,
                      hourly=hourly)
    dry_spells = runs(rows, DRY, hourly=hourly)
    ev = {"peak_mm": peak, "wet_readings": f"{len(wet_spells)} spells of rain",
          "total_mm": _total(rows, "Rainfall"),
          "when": wet_spells[0].label() if wet_spells else None}

    if not wet_spells:
        return Advice(NO, f"Leave it - {peak}mm at the wettest, which is not much rain.", [], ev)
    first = wet_spells[0]
    clear = longest(dry_spells)
    reason = [f"{len(wet_spells)} spell{'s' if len(wet_spells) != 1 else ''} of rain"]
    if clear and clear.hours >= 3 and len(wet_spells) == 1:
        return Advice(YES, f"Take it - rain around {first.label()}, up to {peak}mm. "
                           f"{clear.label()} stays clear if you can go then.",
                      reason, ev, window=first.describe())
    return Advice(YES, f"Take it - rain {first.label()}, up to {peak}mm in one reading.",
                  reason, ev, window=first.describe())


def _sun_protection(rows, sub, hourly):
    sun, daylength = _total(rows, "SunSD"), _total(rows, "DayLength")
    # The hourly feed carries no DayLength, but each of its rows is exactly one hour - so the
    # row count is the denominator. Without this, sun advice on any short window divided by
    # zero and always came back "not really".
    span = daylength or len(values(rows, "SunSD"))
    frac = round(sun / span, 2) if span else 0
    ev = {"sunshine_hrs": sun, "day_length_hrs": daylength or None,
          "hours_covered": None if daylength else span, "sun_fraction": frac}
    caveat = ["no UV index in the feed - judged from sunshine hours and cloud cover"]
    if frac >= 0.75:
        return Advice(YES, f"Yes - {frac:.0%} of the possible sunshine, the sun will be strong.",
                      [], ev, caveats=caveat)
    if frac >= 0.5:
        return Advice(YES, f"Worth it - {frac:.0%} of the possible sunshine.", [], ev,
                      caveats=caveat)
    return Advice(NO, f"Not really - only {frac:.0%} of the possible sunshine.", [], ev,
                  caveats=caveat)


def _clothing(rows, sub, hourly):
    lows = values(rows, "Tmin") or values(rows, "Tavg")
    coldest = round(min(lows), 1) if lows else None
    ev = {"min_c": coldest, "max_c": _peak(rows, "Tmax")}
    if coldest is None:
        return Advice(UNKNOWN, "I cannot tell without a temperature reading.", [], ev)
    if coldest <= 15:
        return Advice(YES, f"Take something warm - it drops to {coldest}°C.", [], ev)
    if coldest <= 20:
        return Advice(CAUTION, f"A light layer - {coldest}°C at the coldest.", [], ev)
    return Advice(NO, f"You will be fine - {coldest}°C at the coldest.", [], ev)


def _fertilize(rows, sub, hourly):
    """Wants rain soon, but not a downpour - the two-sided rule.

    An accumulated total is the right quantity here and always was: what matters is how much
    water arrives to carry the fertiliser in, or to wash it off. It is summed over the whole
    period the question asked about, which is why FERTILIZE defaults to three days rather than
    to whatever range happened to be on screen.
    """
    total, soil = _total(rows, "Rainfall"), _mean(rows, "Soilm10")
    ev = {"total_mm": total, "soil_moisture": soil, "summed_over": "the whole period"}
    if total >= HEAVY_RAIN_48H:
        return Advice(NO, f"Hold off - {total}mm is coming and it will leach or wash off.",
                      [], ev)
    if total >= USEFUL_RAIN_48H:
        dry_first = longest(runs(rows, DRY, hourly=hourly))
        headline = f"Good timing - {total}mm expected, enough to carry it in."
        if dry_first and dry_first.hours >= 2:
            headline += f" Spread it {dry_first.label()}, before the rain."
        return Advice(YES, headline, [], ev, window=dry_first.describe() if dry_first else "")
    if soil is not None and soil < SOIL_DRY:
        return Advice(CAUTION, f"Dry ground ({soil} m³/m³) and only {total}mm coming - it may "
                               f"just sit there.", [], ev)
    return Advice(CAUTION, f"Only {total}mm expected - irrigate after, or wait for rain.",
                  [], ev)


def _irrigate(rows, sub, hourly):
    """How much water is coming, accumulated - the one place a total is the actual question."""
    soil, total = _mean(rows, "Soilm10"), _total(rows, "Rainfall")
    ev = {"soil_moisture": soil, "total_mm": total, "dry_band": SOIL_DRY,
          "summed_over": "the whole period"}
    if total >= 10:
        return Advice(NO, f"Skip it - {total}mm is coming.", [], ev)
    if soil is not None and soil >= SOIL_WET:
        return Advice(NO, f"Not needed - the soil is at {soil} m³/m³.", [], ev)
    if soil is not None and soil <= SOIL_DRY:
        return Advice(YES, f"Yes - soil at {soil} m³/m³ and only {total}mm coming.", [], ev)
    return Advice(CAUTION, f"Borderline - soil {soil} m³/m³, {total}mm expected.", [], ev)


def _sow(rows, sub, hourly):
    soil, total = _mean(rows, "Soilm10"), _total(rows, "Rainfall")
    soil_t = _mean(rows, "Soilt10")
    ev = {"soil_moisture": soil, "total_mm": total, "soil_temp_c": soil_t,
          "summed_over": "the whole period"}
    if soil is not None and soil >= SOIL_DRY and total >= 10:
        return Advice(YES, f"Good - soil at {soil} m³/m³ and {total}mm expected.", [], ev)
    if total < 5 and (soil is None or soil < SOIL_DRY):
        return Advice(NO, f"Too dry - soil {soil} m³/m³ and only {total}mm coming.", [], ev)
    if soil_t is not None and soil_t < 15:
        return Advice(CAUTION, f"The ground is cold ({soil_t}°C) - germination will be slow.",
                      [], ev)
    return Advice(CAUTION, f"Borderline - soil {soil} m³/m³, {total}mm expected.", [], ev)


STATE_RULES = {
    "RAIN_PROTECTION": _rain_protection, "SUN_PROTECTION": _sun_protection,
    "CLOTHING": _clothing, "FERTILIZE": _fertilize, "IRRIGATE": _irrigate, "SOW": _sow,
}


def evaluate(activity: str, rows: list[dict], *, sub_activity: str = "",
             hourly: bool | None = None) -> Advice | None:
    """A verdict, or None when this is not an advice turn.

    Returns UNKNOWN rather than a verdict when the data cannot support one - a confident
    answer computed from two readings out of thirty is indistinguishable from a real one,
    which is exactly what makes it dangerous.

    `hourly` says what one row covers. It is inferred from the timestamps when not given, so a
    caller that does not know still gets the right answer for any period with two readings in.
    """
    if activity not in NEEDS:
        return None
    if hourly is None:
        hourly = spacing_hours(rows) < 24.0

    needed = NEEDS[activity]
    quality = assess(rows, needed, required=needed)
    if not quality.can_decide:
        return Advice(UNKNOWN, "I cannot answer that from the data I got back.",
                      [quality.message], {"status": quality.status, "rows": quality.rows},
                      activity=activity, sub_activity=sub_activity)

    if (spec := TIMED.get(activity)):
        # a car does not care about damp air, only about rain, and it dries in less time
        if activity == "DRYING" and sub_activity == "vehicle":
            spec = Timed(spec.needs, NOT_DAMP, hours=2, verb="wash the vehicle",
                         blocker="rain")
        elif activity == "OUTDOOR_ACTIVITY" and sub_activity:
            def is_daylight(r):
                val = reading(r, "SunSD")
                if val is not None:
                    return val > 0.0
                w = _when(r)
                return w is not None and 6 <= w.hour < 19
            sports_during = every(DRY, BEARABLE, is_daylight)
            spec = Timed(("Rainfall", "Tmax", "SunSD"), sports_during, hours=spec.hours,
                         verb=f"play {sub_activity}", blocker="rain, heat or darkness")
        advice = _timed(spec, rows, hourly, sub_activity)
    else:
        advice = STATE_RULES[activity](rows, sub_activity, hourly)

    advice.activity, advice.sub_activity = activity, sub_activity
    if quality.status == "PARTIAL":
        advice.caveats.append(f"partial data: {quality.message}")
    return advice


def demo():
    """Self-check: timing beats totals, thresholds still flip, and nothing decides blind."""
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


if __name__ == "__main__":
    demo()
