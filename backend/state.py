"""
Context engine - what the conversation remembers, and how each turn changes it.

Deliberately model-free. "What about tomorrow?" needs no inference: it carries one new slot,
so the rest is inherited. Spending a classifier on that would be slower and less predictable
than the twenty lines below.

    turn 1  "weather in Kakinada"        SET       location=Kakinada, time=None
    turn 2  "what about tomorrow?"       MODIFY    location=Kakinada, time=tomorrow
    turn 3  "what about Rajahmundry?"    REPLACE   location=Rajahmundry, time=tomorrow
    turn 4  "and there?"                 INHERIT   location=Rajahmundry, time=tomorrow

References ("there", "same place") are matched by rules first - they are a closed set of
phrases. Only genuinely ambiguous wording is worth a model, and this domain has almost none.
"""

from __future__ import annotations

import re

from src.schema import Aggregation, ConversationState, Operation, Reference

# Closed sets: cheap to match, impossible to get wrong.
LOCATION_REFERENCES = {
    "there", "that place", "same place", "that city", "that town", "same location",
    "that village", "the same", "over there", "same spot", "that area",
}
DATE_REFERENCES = {"same day", "that day", "same date", "then", "that time"}
FOLLOW_UP = re.compile(
    r"^\s*(and|but|ok|okay|so)?\s*(what|how)?\s*(about|abt)\b|^\s*(and|what about)\b", re.I)


def detect_reference(text: str) -> Reference:
    """Which part of the previous turn a phrase points at, if any."""
    lowered = " ".join(text.lower().split()).strip(" ?.!,")
    if any(phrase in lowered for phrase in LOCATION_REFERENCES):
        return Reference.LOCATION
    if any(phrase in lowered for phrase in DATE_REFERENCES):
        return Reference.DATE
    return Reference.NONE


def is_follow_up(text: str) -> bool:
    """"what about tomorrow?" / "and in Guntur?" - a fragment that leans on the last turn."""
    return bool(FOLLOW_UP.match(text.strip()))


def classify_operation(state: ConversationState, has_location: bool, has_time: bool,
                       reference: Reference, action: str, follow_up: bool,
                       confident: bool = True) -> Operation:
    """What this turn does to the state. Order matters: the most specific case wins."""
    # COMPARE has to be believed to be acted on: "and there?" is not a comparison just
    # because a 36%-confidence classifier said so.
    if action == "COMPARE" and confident and (has_location or reference != Reference.NONE):
        return Operation.COMPARE
    if not state.turns:
        return Operation.SET
    if reference == Reference.LOCATION and not has_location:
        # "there next 3 days" - the place is inherited, the time is new
        return Operation.MODIFY if has_time else Operation.INHERIT
    if follow_up and has_location and not has_time:
        return Operation.REPLACE          # "what about Rajahmundry?"
    if follow_up and has_time and not has_location:
        return Operation.MODIFY           # "what about tomorrow?"
    if has_location and has_time:
        return Operation.SET
    if has_location:
        return Operation.REPLACE
    if has_time:
        return Operation.MODIFY
    return Operation.INHERIT


def apply(state: ConversationState, *, weather_intent, action, aggregation,
          location: list[str], time_raw: str | None, time_normalized: str | None,
          reference: Reference, follow_up: bool, confident: bool = True
          ) -> tuple[ConversationState, Operation]:
    """Fold one turn into the state and report which operation it was.

    Returns a new state; the caller decides whether to keep it (a rejected turn should not
    poison the next one).

    A low-confidence follow-up keeps the previous intent instead of adopting a guess:
    "what about tomorrow?" carries no weather word, so whatever the classifier ranks first
    is noise. The question is still the previous question, asked about a new day.
    """
    operation = classify_operation(state, bool(location), bool(time_normalized or time_raw),
                                   reference, action, follow_up, confident)

    merged = state.model_copy(deep=True)
    merged.turns += 1
    inherit_intent = (not confident and state.weather_intent is not None
                      and operation in {Operation.MODIFY, Operation.REPLACE, Operation.INHERIT})
    merged.weather_intent = state.weather_intent if inherit_intent else weather_intent
    merged.action = state.action if inherit_intent else action
    merged.aggregation = aggregation or Aggregation.RAW

    if operation in {Operation.SET, Operation.COMPARE, Operation.REPLACE} and location:
        merged.location = list(location)
        merged.resolved = []                      # the old coordinates no longer apply
    if (operation in {Operation.SET, Operation.COMPARE, Operation.MODIFY, Operation.INHERIT}
            and (time_raw or time_normalized)):
        merged.time_raw, merged.time_normalized = time_raw, time_normalized
    if operation == Operation.SET and not (time_raw or time_normalized):
        # a fresh question with no time means "the near term", not "whatever we said before"
        merged.time_raw = merged.time_normalized = None

    # An intent-only follow-up ("and the humidity?") keeps both slots, which INHERIT covers.
    return merged, operation


def missing_slots(state: ConversationState) -> list[str]:
    """Slots the query planner still needs. Time is optional - the API has a default horizon."""
    return [] if (state.location or state.coords) else ["location"]


def demo():
    """Self-check: the four-turn conversation from the module docstring."""
    state = ConversationState()
    turn = lambda **kwargs: {"weather_intent": "CURRENT_CONDITIONS", "action": "GET",
                             "aggregation": Aggregation.RAW, "reference": Reference.NONE,
                             "follow_up": False, "time_raw": None, "time_normalized": None,
                             "location": [], **kwargs}

    state, op = apply(state, **turn(location=["Kakinada"]))
    assert op == Operation.SET and state.location == ["Kakinada"]

    state, op = apply(state, **turn(time_raw="tomorrow", time_normalized="tomorrow",
                                    follow_up=True))
    assert op == Operation.MODIFY, op
    assert state.location == ["Kakinada"] and state.time_normalized == "tomorrow"

    state, op = apply(state, **turn(location=["Rajahmundry"], follow_up=True))
    assert op == Operation.REPLACE, op
    assert state.location == ["Rajahmundry"] and state.time_normalized == "tomorrow"

    state, op = apply(state, **turn(reference=Reference.LOCATION, follow_up=True))
    assert op == Operation.INHERIT, op
    assert state.location == ["Rajahmundry"] and state.time_normalized == "tomorrow"

    # "there" plus a new time modifies the time and inherits the place
    state, op = apply(state, **turn(reference=Reference.LOCATION, time_raw="next 3 days",
                                    time_normalized="next 3 days", follow_up=True))
    assert op == Operation.MODIFY, op
    assert state.location == ["Rajahmundry"] and state.time_normalized == "next 3 days"

    # a low-confidence fragment keeps the previous question rather than adopting a guess
    state, op = apply(state, **turn(weather_intent="WIND_SPEED", action="ALERT",
                                    time_raw="tomorrow", time_normalized="tomorrow",
                                    follow_up=True), confident=False)
    assert state.weather_intent == "CURRENT_CONDITIONS", state.weather_intent
    assert state.action == "GET", state.action

    assert detect_reference("what about there?") == Reference.LOCATION
    assert detect_reference("same day please") == Reference.DATE
    assert detect_reference("rain in Guntur") == Reference.NONE
    assert is_follow_up("what about tomorrow?") and is_follow_up("and in Guntur?")
    assert not is_follow_up("will it rain in Guntur?")
    assert missing_slots(ConversationState()) == ["location"]
    print("state demo OK:", state.model_dump(include={"location", "time_normalized", "turns"}))


if __name__ == "__main__":
    demo()
