"""
The four decisions, in one object, each with the reason it was made.

    python tests/test_pipeline_units.py           # the checks for this module

    which API        source and resolution   <- the planner, which knows what each one costs
    how many places  the resolved places, capped where a route says so
    which action     the aggregation over the range
    which window     the span, from what was said or what the profile assumes

None of these are new. They were made in three different places - `run()` chose the fields and
the aggregation, `plan()` chose the source and resolution, `confirm_aggregation()` re-read the
sentence - and no two of them recorded why. That is the whole change: one object, one audit
trail, one place to look when an answer came back hourly and nobody knows why.

**What the user said always wins.** Everything a profile carries is a default for the case
where they said nothing. A default that overrode a spoken word would be the system deciding it
knows better, which is the one thing a slot-filling architecture must never do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from backend.pipeline import analysis
from backend.pipeline import plan as planner


@dataclass
class Params:
    """Everything the fetch needs, and how each part of it was decided."""

    route: str
    window: str = ""                 # the canonical expression, as handed to the planner
    fields: list = field(default_factory=list)
    aggregation: str = "RAW"
    places: list = field(default_factory=list)
    plan: planner.QueryPlan | None = None
    assumed: list = field(default_factory=list)   # defaults applied, for the answer to admit
    why: dict = field(default_factory=dict)       # decision -> one line, for the audit strip

    def as_dict(self) -> dict:
        """The wire form, for `stages["params"]`. The plan renders itself."""
        return {"route": self.route, "window": self.window, "fields": self.fields,
                "aggregation": self.aggregation, "places": len(self.places),
                "assumed": self.assumed, "why": self.why}


def resolve(understanding, profile, places: list, *, now: datetime | None = None,
            aggregation: str | None = None) -> Params:
    """Slots plus a profile, out come the parameters for one fetch.

    `aggregation` overrides everything when given - the comparison view pins it so three
    columns are reduced the same way and can honestly be put side by side.
    """
    now = now or datetime.now()
    params = Params(route=profile.route, places=places)
    said = lambda key, line: params.why.__setitem__(key, line)

    # 1. window - theirs, else the profile's, else the planner's own default horizon
    spoken = understanding.times_normalized[0] if understanding.times_normalized else ""
    if spoken:
        params.window = spoken
        said("window", f"{spoken!r} - named in the question")
    elif profile.window:
        params.window = profile.window
        said("window", f"{profile.window!r} - assumed for {profile.route}: {profile.note}")
        params.assumed.append(f"looked at {profile.window}")
    else:
        said("window", "no time named - the planner's default horizon")

    # 2. aggregation - a caller's pin, else what was spoken, else the profile's default
    if aggregation is not None:
        params.aggregation = aggregation
        said("aggregation", f"{aggregation} - pinned by the caller")
    else:
        confirmed = analysis.confirm_aggregation(understanding.text, understanding.aggregation)
        if confirmed != "RAW":
            params.aggregation = confirmed
            said("aggregation", f"{confirmed} - asked for in the wording")
        elif profile.aggregation != "RAW":
            params.aggregation = profile.aggregation
            said("aggregation", f"{profile.aggregation} - {profile.note}")
            params.assumed.append(f"reported the {profile.aggregation.lower()} "
                                  f"rather than every reading")
        else:
            said("aggregation", "RAW - no reduction asked for")

    # 3. fields - what was asked for, plus what the decision reads whether it was asked or not.
    # The union is the fix: a spraying rule needs wind to answer at all, the sentence rarely
    # says "wind", and nothing used to put it in the fetch. A rule handed a column it needs
    # and did not get returns UNKNOWN, which reads as "I cannot answer that".
    asked = understanding.fields()
    extra = [f for f in profile.fields if f not in asked]
    params.fields = asked + extra
    said("fields", f"{len(asked)} asked for" +
         (f", plus {', '.join(extra)} that {profile.route} reads" if extra else ""))

    # 4. places
    said("places", f"{len(places)} resolved")

    # 5. source and resolution - the planner's, unchanged. It is the layer that knows Zarr
    # serves points and Postgres serves districts, and it is the only one holding a row budget.
    params.plan = planner.plan(
        times_normalized=[params.window] if params.window else [], places=places,
        aggregation=params.aggregation, level=(places[0].get("type") or "village"
                                               if places else "village"),
        fields=params.fields, now=now)
    chosen = params.plan
    said("source", f"{chosen.source and chosen.source.value} at "
                   f"{chosen.resolution and chosen.resolution.value}"
                   f" - {chosen.reason or f'{chosen.rows} rows, within budget'}")
    return params
