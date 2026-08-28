"""
What each kind of question needs, before anything has been fetched for it.

    python tests/test_pipeline_units.py          # the checks for this module

One `run()` used to answer every question with the same nine steps. A forecast, a look at last
June and "can I spray tomorrow" want different windows, different columns and different
reductions, and every one of those differences was either absent or buried in an `if` halfway
down the pipeline. They are here instead, as data.

    route      what kind of question this is - derived, never predicted
    window     what "no time named" means for it
    fields     what the answer reads, whether or not the user named it
    resolution AUTO unless this route knows better than the planner
    aggregation what to do with the range when nobody said

**A key exists only when two profiles disagree on it.** That is the same test `src/v4/schema.py`
applies to its Activity labels, and it is what keeps this from becoming config for values that
never change. `blocks`, `min_hours` and `settle_hours` are named in the plan and deliberately
absent until the step that reads them - an unread field is a lie about what the system does.

The activity profiles read `backend.pipeline.advice` rather than copying it. Its `DEFAULT_WINDOW`
and `NEEDS` are already the per-activity config and they are already correct; a second copy here
would be two tables that disagree the first time either is tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.pipeline import advice as advice_engine
from backend.pipeline.timewindow import DEFAULT_HORIZON_DAYS

FORECAST, HISTORICAL, ACTIVITY, COMPARE = "FORECAST", "HISTORICAL", "ACTIVITY", "COMPARE"

# Fields that mean something added up. `render.ADDITIVE` says the same thing and is the one
# that decides how a figure is *written*; this decides whether a long look back is summed or
# averaged, which is a different question asked of the same fact.
SUMMABLE = ("Rainfall", "SunSD")

# Below this many days, a look at the past is a short list you can read. Above it, thirty rows
# of numbers is a table rather than an answer, and the question wanted a figure.
LONG_LOOK_BACK_DAYS = 7


@dataclass(frozen=True)
class Profile:
    """The defaults for one kind of question. Every field is a default, never an override:
    anything the user actually said wins over all of it, in `backend.pipeline.params`."""

    route: str
    window: str = ""              # canonical time expression, "" = the planner's own default
    fields: tuple = ()            # columns the answer needs regardless of what was asked for
    resolution: str = "AUTO"      # AUTO | HOURLY | DAILY - AUTO leaves it to the planner
    aggregation: str = "RAW"
    note: str = ""                # why this profile, for the audit trail


def route_for(understanding, window, now) -> str:
    """Which of the four this turn is. Derived from slots v4 already predicts.

    Nothing is trained and nothing is retrained. `activity` decides an advice turn, the
    COMPARE/GET axis is already derived in the registry, and past-or-future is a property of
    the window - which is why the window is resolved before the route rather than after it.

    Tense is decided by where the window STARTS. "over the last 5 years" ends a few minutes
    from now, and an end-based test called it a forecast. `backend.pipeline.plan` learned that
    the hard way and the same rule has to hold here or the two disagree about the same turn.
    """
    if understanding.activity != "NONE":
        return ACTIVITY
    if understanding.action == "COMPARE":
        return COMPARE
    if window is not None and window.start.date() < now.date():
        return HISTORICAL
    return FORECAST


def _historical(window, variables: list) -> Profile:
    """Looking back. Long spans get reduced, because thirty rows is not an answer.

    The reduction is a real change of behaviour and the only one in this file: "rainfall in
    Guntur last June" used to come back as thirty raw daily readings. Nobody asking that wants
    thirty numbers, and the aggregation slot said RAW only because the sentence contained no
    word like "total" - which is not the same as the reader wanting every row.

    It is a default, so a spoken aggregation still wins, and `params` records it in `assumed`
    so the answer says a figure was chosen for them.
    """
    if window.span_days <= LONG_LOOK_BACK_DAYS:
        return Profile(HISTORICAL, note=f"{window.span_days} days back - short enough to read")
    summed = any(v in ("RAIN", "SUNSHINE") for v in variables)
    return Profile(HISTORICAL, aggregation="SUM" if summed else "AVG",
                   note=f"{window.span_days} days back - reduced rather than listed row by row")


def _activity(understanding) -> Profile:
    """An advice turn: the window and the columns its rule reads.

    Both come from the advice engine, which is where they were already correct. What is added
    here is that the columns the *decision* needs are carried as a default at all - the rule
    needs wind to answer a spraying question whether or not the sentence said "wind", and
    nothing used to put it in the fetch. A rule handed a column it needs and did not get
    returns UNKNOWN, which reads to the user as "I cannot answer that".
    """
    activity = understanding.activity
    return Profile(
        ACTIVITY,
        window=advice_engine.DEFAULT_WINDOW.get(activity, ""),
        fields=tuple(advice_engine.NEEDS.get(activity, ())),
        note=f"{activity} reads {', '.join(advice_engine.NEEDS.get(activity, ())) or 'nothing'}")


def pick(understanding, window, now) -> Profile:
    """The profile for this turn. `window` is the user's own window, or None if they named none."""
    route = route_for(understanding, window, now)
    if route == ACTIVITY:
        return _activity(understanding)
    if route == HISTORICAL:
        return _historical(window, understanding.variables)
    if route == COMPARE:
        return Profile(COMPARE, note="two or more places, side by side")
    return Profile(FORECAST, note=f"forward, default horizon {DEFAULT_HORIZON_DAYS} days")
