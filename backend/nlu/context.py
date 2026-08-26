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

from src.build_dataset import NOUNS
from src.schema import Aggregation, ConversationState, Operation, Reference

# Every word that names a measurement, from both model vocabularies. A follow-up that
# contains none of them is not asking about a new variable, whatever the classifier says -
# "what about Vizag?" came back as ALERT at 95%, and confidence alone cannot catch that.
# NOUNS holds whole phrases ("what's the weather like", "how hot it is"), so the function
# words have to be stripped out - otherwise "what about Vizag?" counts as naming a variable
# because of the word "what".
_FUNCTION_WORDS = {
    "what", "whats", "how", "will", "going", "the", "like", "right", "now", "today", "and",
    "for", "with", "you", "your", "get", "got", "there", "here", "that", "this", "kind",
    "much", "level", "levels", "when", "where", "which", "does", "did", "are", "was", "its",
    "current", "currently", "actual", "feels", "outside", "next", "days", "day", "week",
    "ahead", "upcoming", "future", "daily", "weekly", "long", "range", "probability", "amount",
}
VARIABLE_WORDS = {word.lower()
                  for nouns in NOUNS.values() for noun in nouns for word in noun.split()
                  if len(word) > 2 and word.lower() not in _FUNCTION_WORDS}
ACTION_WORDS = {"compare", "vs", "versus", "against", "difference", "between", "alert",
                "warn", "notify", "remind", "ping", "watch", "tell"}


def mentions_variable(text: str) -> bool:
    """Does the message name a measurement at all?"""
    words = {word.strip(" ,.?!").lower() for word in text.split()}
    return bool(words & VARIABLE_WORDS)


def mentions_action(text: str) -> bool:
    """Does it explicitly ask to compare or to be alerted?"""
    words = {word.strip(" ,.?!").lower() for word in text.split()}
    return bool(words & ACTION_WORDS)

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
                       confident: bool = True, explicit_compare: bool = True) -> Operation:
    """What this turn does to the state. Order matters: the most specific case wins."""
    # A comparison needs evidence: either the message says so ("compare", "vs") or it names
    # two places. "and there next week?" is one reference and a date, whatever the
    # classifier ranked first.
    if action == "COMPARE" and confident and explicit_compare:
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
          reference: Reference, follow_up: bool, confident: bool = True,
          text: str = "", variables: list[str] | None = None
          ) -> tuple[ConversationState, Operation]:
    """Fold one turn into the state and report which operation it was.

    Returns a new state; the caller decides whether to keep it (a rejected turn should not
    poison the next one).

    A follow-up that names no measurement keeps the previous intent and variables, however
    confident the classifier is: "what about Vizag?" contains no weather word, so a 95%
    ALERT is 95% confidence in noise. The question is still the previous question, asked
    about a new place.
    """
    explicit_compare = len(location) >= 2 or (bool(text) and mentions_action(text))
    operation = classify_operation(state, bool(location), bool(time_normalized or time_raw),
                                   reference, action, follow_up, confident, explicit_compare)

    merged = state.model_copy(deep=True)
    merged.turns += 1

    fragment = operation in {Operation.MODIFY, Operation.REPLACE, Operation.INHERIT}
    silent = bool(text) and not mentions_variable(text)      # says nothing about a measurement
    inherit_intent = (state.weather_intent is not None and fragment
                      and (silent or not confident))
    merged.weather_intent = state.weather_intent if inherit_intent else weather_intent
    merged.action = (state.action if inherit_intent and not mentions_action(text) else action)
    # Coerced: callers pass the plain string their model produced and the field on
    # ConversationState is an enum - pydantic warns on every model_dump() otherwise, and the
    # state is dumped into the turn log on every answered turn. `.value` first, because on
    # 3.9 `str(Aggregation.RAW)` is "Aggregation.RAW", not "RAW" - which is why passing the
    # enum itself raised instead of being the trivially correct case.
    merged.aggregation = (Aggregation(getattr(aggregation, "value", aggregation))
                          if aggregation else Aggregation.RAW)

    # v2 carries several variables; a silent follow-up keeps all of them, so "rain and
    # temperature in Guntur" -> "what about Vizag?" still answers with both columns
    if variables is not None:
        merged.variables = list(state.variables) if inherit_intent and state.variables else list(variables)

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
