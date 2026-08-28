"""
Understanding: conversation state, the model registry, and the hosted parser's coercion.
Run: python tests/test_nlu_units.py

Moved out of `backend/nlu/*.py`. The hosted-parser check is the offline half - the parsing and
coercion that turn prose, fences and invented labels into a valid contract or into nothing. The
half that calls the API lives in tests/test_live_stack.py.
"""

from _root import ROOT  # noqa: F401 - puts the repo root on sys.path


def check_context():
    """Self-check: the four-turn conversation from the module docstring."""
    from backend.nlu.context import (
        Aggregation,
        ConversationState,
        Operation,
        Reference,
        apply,
        detect_reference,
        is_follow_up,
        missing_slots,
    )
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

    # a CONFIDENT fragment that names no measurement is inherited too - this is the v2 case
    # where "what about Vizag?" arrived as ALERT at 95% and dropped one of two variables
    multi = ConversationState(location=["Guntur"], weather_intent="FORECAST", action="GET",
                              variables=["RAIN", "TEMPERATURE"], turns=1)
    multi, op = apply(multi, **turn(weather_intent="ALERT", action="ALERT",
                                    location=["Vizag"], follow_up=True),
                      confident=True, text="what about Vizag?", variables=["TEMPERATURE"])
    assert op == Operation.REPLACE, op
    assert multi.weather_intent == "FORECAST", multi.weather_intent
    assert multi.variables == ["RAIN", "TEMPERATURE"], multi.variables
    assert multi.location == ["Vizag"], multi.location

    # a bare reference plus a date is not a comparison, however the classifier reads it
    multi, op = apply(multi, **turn(weather_intent="CURRENT_CONDITIONS", action="COMPARE",
                                    reference=Reference.LOCATION, time_raw="next week",
                                    time_normalized="next week", follow_up=True),
                      confident=True, text="and there next week?", variables=["GENERAL"])
    assert op == Operation.MODIFY, op
    assert multi.action == "GET", multi.action
    assert multi.time_normalized == "next week", multi.time_normalized

    # but naming a measurement replaces it, as it should
    multi, op = apply(multi, **turn(weather_intent="HUMIDITY", action="GET", follow_up=True),
                      confident=True, text="what about humidity there?", variables=["HUMIDITY"])
    assert multi.variables == ["HUMIDITY"], multi.variables

    assert detect_reference("what about there?") == Reference.LOCATION
    assert detect_reference("same day please") == Reference.DATE
    assert detect_reference("rain in Guntur") == Reference.NONE
    assert is_follow_up("what about tomorrow?") and is_follow_up("and in Guntur?")
    assert not is_follow_up("will it rain in Guntur?")
    assert missing_slots(ConversationState()) == ["location"]
    print("state demo OK:", state.model_dump(include={"location", "time_normalized", "turns"}))

def check_registry():
    """Self-check: both models answer, and v4 routes a greeting away from the weather API."""
    from backend.nlu.registry import MODELS, Registry, normalize_text
    cleaned = normalize_text("Whats da wthr in KKD tmrw??")
    assert cleaned.normalized == "what is the weather in KKD tomorrow??", cleaned.normalized
    assert cleaned.replacements, "every rewrite has to stay auditable"

    registry = Registry()
    for version, spec in MODELS.items():
        if not spec["path"].exists():
            print(f"skip {spec['name']}: bundle not built")
            continue
        u = registry.understand("rain and temperature in Guntur tomorrow", version)
        assert u.version == version and u.locations == ["Guntur"], u
        assert u.fields(), u
        print(f"  {spec['name']} ({version}): intent={u.intent} action={u.action} "
              f"vars={u.variables} fields={u.fields()}")

    if MODELS["v4"]["path"].exists():
        greeting = registry.understand("hey there", "v4")
        assert not greeting.needs_weather and greeting.reply, greeting
        assert greeting.family == "conversational", greeting.family
        print(f"  greeting -> {greeting.intent} ({greeting.family}) reply={greeting.reply!r}")

        spray = registry.understand("should i spray fertilizer on the cotton tomorrow", "v4")
        assert spray.activity == "SPRAY", spray.activity
        print(f"  advice   -> activity={spray.activity} sub={spray.sub_activity or '-'} "
              f"entities={spray.entities}")
    print("registry demo OK")

def check_hosted_parser_coercion():
    """The parsing and coercion, without the network - the quota is not always there.

    This is where the bugs actually are: a hosted model returns prose, fences, invented labels
    and spans that are not in the sentence, and every one of those has to become either a valid
    contract or nothing.
    """
    from backend.nlu.llm import Activity, Aggregation, Intent, _coerce, _json_from
    assert _json_from('{"intent":"ADVICE"}')["intent"] == "ADVICE"
    assert _json_from('```json\n{"intent":"ADVICE"}\n```')["intent"] == "ADVICE"
    assert _json_from('Sure! Here you go:\n{"intent":"ADVICE"}\nHope that helps')["intent"] == "ADVICE"
    assert _json_from("not json at all") == {}

    text = "should i spray fertilizer on the cotton in Guntur tomorrow"
    got = _coerce({"intent": "ADVICE", "activity": "SPRAY", "variables": ["WIND", "RAIN"],
                   "aggregation": "RAW", "locations": ["Guntur"], "times": ["tomorrow"]}, text)
    assert got["intent"] is Intent.ADVICE and got["activity"] is Activity.SPRAY, got
    assert got["locations"] == ["Guntur"] and got["times"] == ["tomorrow"], got

    # invented labels are dropped, not passed through
    junk = _coerce({"intent": "WEATHER_QUERY", "activity": "PLOUGHING",
                    "variables": ["RAIN", "MOONPHASE"], "aggregation": "AVERAGE",
                    "locations": ["Hyderabad"], "times": []}, text)
    assert junk["intent"] is Intent.INFORMATION, junk["intent"]     # unknown -> fallback
    assert junk["activity"] is Activity.NONE, junk["activity"]      # unknown, and not ADVICE
    assert [v.value for v in junk["variables"]] == ["RAIN"], junk["variables"]
    assert junk["aggregation"] is Aggregation.RAW, junk["aggregation"]
    # "Hyderabad" is not in the sentence - a span the model invented is worse than no span
    assert junk["locations"] == [], junk["locations"]

    # activity only survives on an ADVICE turn
    assert _coerce({"intent": "INFORMATION", "activity": "SPRAY"}, text)["activity"] is Activity.NONE

    # a chat turn carries no window, no variables, no spans
    chat = _coerce({"intent": "GREETING", "variables": ["RAIN"], "locations": ["Guntur"],
                    "times": ["tomorrow"]}, "hey there Guntur tomorrow")
    assert chat["variables"] == [] and chat["times"] == [] and chat["locations"] == [], chat
    # ...except CHANGE_LOCATION, which is entirely about the place it names
    moved = _coerce({"intent": "CHANGE_LOCATION", "locations": ["Guntur"]}, "set location to Guntur")
    assert moved["locations"] == ["Guntur"], moved

    assert _coerce({}, text)["intent"] is Intent.INFORMATION       # empty reply still valid
    print("llm_nlu offline check OK - coercion, fences, invented labels, chat blanking")

def check_derived():
    """weather_intent and venue: derived from slots, so they cannot disagree with them.

    Both are properties rather than fields, which is what this checks - there is no
    construction site to forget, and no second head to drift.
    """
    from backend.nlu import Registry
    from src.v4.schema import Activity, venue_for

    # the word in the sentence beats the list of indoor sports, both ways round
    cases = [
        ((Activity.OUTDOOR_ACTIVITY, "badminton", "can i play badminton at six"), "indoor"),
        ((Activity.OUTDOOR_ACTIVITY, "cricket", "can i play cricket tomorrow"), "outdoor"),
        ((Activity.OUTDOOR_ACTIVITY, "cricket", "indoor cricket tomorrow"), "indoor"),
        ((Activity.OUTDOOR_ACTIVITY, "badminton", "outdoor badminton court"), "outdoor"),
        # not an OUTDOOR_ACTIVITY: a journey and a crop are outdoors by definition
        ((Activity.TRAVEL, "car", "should i drive to guntur"), "outdoor"),
        ((Activity.SPRAY, "", "should i spray tomorrow"), "outdoor"),
        ((Activity.NONE, "", "will it rain tomorrow"), "outdoor"),
    ]
    for (activity, sub, text), want in cases:
        got = venue_for(activity, sub, text)
        assert got == want, f"{text!r}: got {got}, want {want}"

    registry = Registry()
    for text, wanted in (("will it rain in guntur tomorrow", "TOMORROW"),
                         ("rainfall in guntur last june", "HISTORICAL"),
                         ("weather in guntur right now", "CURRENT"),
                         ("weather in guntur next week", "FORECAST"),
                         ("hey there", "NONE")):
        got = registry.understand(text).weather_intent
        assert got == wanted, f"{text!r}: weather_intent {got}, wanted {wanted}"

    # a turn that needs no weather has no window at all - weather_intent_for(None) says
    # FORECAST, which is right for a question and wrong for a greeting
    assert registry.understand("hey there").weather_intent == "NONE"
    assert registry.understand("can i play badminton this evening").venue == "indoor"
    # --- the time gate: a model may propose any string, and only forms the deterministic
    # resolver recognises are believed. Asked to place "the last 7 days" a 1.7b answered
    # "last 7 weeks" - a valid form and the wrong window - which is why the rules go first
    # and the model is only asked what they could not place.
    import asyncio

    from backend.nlu import times

    times._SEEN.clear()
    replies = {"prior days": "last 7 days", "coming through tonight": "tonight",
               "at half five": "17:30",
               "the other day": "sometime recently",   # not a canonical form: must be refused
               "rainfall": ""}                          # names no time: must be refused
    kept = {span: form for span, form in replies.items() if times.known(form)}
    assert kept == {"prior days": "last 7 days", "coming through tonight": "tonight",
                    "at half five": "17:30"}, kept

    # a refusal is remembered, so a phrase nobody can place is asked about once, not per turn
    times._SEEN["the other day"] = ""
    assert asyncio.run(times.canonicalize(["the other day"])) == {}, "a refusal is remembered"
    times._SEEN.clear()
    print("time gate OK - invented forms refused, refusals remembered")

    print("derived slots OK - weather_intent from the time slot, venue from the words")

def main():
    """Every check in this file, in order. Any assertion failure stops it."""
    for check in (check_context, check_registry, check_hosted_parser_coercion,
                  check_derived):
        print(f"{check.__name__}:")
        check()
    print("\n4 check(s) passed")


if __name__ == "__main__":
    main()
