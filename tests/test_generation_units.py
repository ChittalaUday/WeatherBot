"""
Wording: the context the model is given, the prompts, and the grounding checks on what it says.
Run: python tests/test_generation_units.py

Moved out of `backend/generation/*.py`. The last check reaches for ollama if it is running and
passes either way - it says which happened.
"""

from _root import ROOT  # noqa: F401 - puts the repo root on sys.path


def check_context():
    """Self-check: the sections keep their order, and a big table is summarised not truncated."""
    from backend.generation.context import build
    from backend.pipeline.analysis import Note

    def month(days):
        return {"places": [{"name": "Guntur", "state": "Andhra Pradesh"}],
                "when": "this week", "hourly": False,
                "insights": [Note("RANGE", "average rainfall 2.0mm, range 0.0-6.0mm"),
                             Note("THRESHOLD", "heavy rain on 1 of 7 days (peak 12.0mm)")],
                "table": {"columns": [{"key": "time", "label": "When"},
                                      {"key": "Rainfall", "label": "Rainfall (mm)"}],
                          "rows": [{"time": f"{d} Aug", "Rainfall": "2"}
                                   for d in range(1, days + 1)]}}

    week = build(month(7)).render()
    assert "Places: Guntur (Andhra Pradesh)" in week, week
    assert "Range: average rainfall" in week and "Notable: heavy rain" in week, week
    assert "1 Aug | 2" in week, week
    assert week.index("Range") < week.index("Notable") < week.index("Figures"), week

    # too much table: the rows go, the summary of them stays
    year = build(month(365)).render()
    assert "1 Aug | 2" not in year and "365 rows" in year, year
    assert "average rainfall" in year, "the summary must survive when the rows do not"
    assert "|" not in build(month(24)).render(), "a day of hourly rows is a large set"

    # a comparison leads, whatever order the notes arrived in
    compared = build({"places": [{"name": "A"}, {"name": "B"}], "when": "tomorrow",
                      "insights": [Note("RANGE", "A: average 2mm", "A"),
                                   Note("COMPARISON", "A leads B by 2mm on rainfall")]}).render()
    assert compared.index("Comparison") < compared.index("Range"), compared

    # an advice turn carries the verdict and what it was read off
    from backend.pipeline.advice import Advice
    decided = build({"places": [{"name": "Guntur"}], "when": "tomorrow",
                     "advice": Advice("YES", "Yes, but pick your moment.", [],
                                      {"mean_wind_ms": 6.0, "total_mm": 0.0},
                                      window="06:00 to 12:00 (6 hours)",
                                      caveats=["partial data: 80% coverage"])}).render()
    assert "Decision:" in decided and "pick your moment" in decided, decided
    assert "Best window: 06:00 to 12:00 (6 hours)" in decided, decided
    # the evidence dict's machine keys must never reach the prompt - the model reads them out
    assert "mean wind ms" not in decided and "mean_wind_ms" not in decided, decided
    assert "Caution:" in decided, decided
    # one line renders inline, several render as a list - so a heading never sits alone
    assert "Period: tomorrow" in decided, decided

    assert not build({}), "nothing retrieved is an empty context, not an empty heading list"
    print("context demo OK\n")
    print("\n".join("  " + line for line in week.splitlines()))

def check_prompts():
    """Print every shape, and assert the blocks do not contradict or leak into each other."""
    from backend.generation.prompts import (
        ANSWERING,
        GROUNDING,
        NEAR,
        NOTHING,
        ROLE,
        VOICE,
        system,
        user,
    )
    for answering, grounded in ((True, True), (True, False), (False, True), (False, False)):
        name = ("answer" if answering else "no-data") + (" + retrieval" if grounded else "")
        prompt = system(answering=answering, grounded=grounded)
        print(f"--- {name}  ({len(prompt)} chars, {len(prompt.split())} words) ---")

    # every block must reach exactly one of the four shapes, or it is text nobody reads
    every = "\n".join(system(answering=a, grounded=g)
                      for a in (True, False) for g in (True, False))
    for block in (ROLE, VOICE, ANSWERING, GROUNDING, NOTHING, NEAR):
        assert block in every, block[:40]
    assert GROUNDING not in system(answering=True, grounded=False)
    assert NOTHING not in system(answering=True, grounded=True)

    # The permission to round and the ban on inventing must both reach an answering turn, and
    # must be separate sentences - told only "never invent a figure" a model writes "1.93mm",
    # and told only "round the way people speak" it writes whatever sounds round.
    answering_prompt = system(answering=True, grounded=True)
    assert "Round figures the way people speak" in answering_prompt
    assert "may not invent, recalculate, add together" in answering_prompt

    # ...and neither reaches a no-data turn, which has nothing to round and must not be given
    # the idea that producing a figure is ever in scope
    no_data = system(answering=False, grounded=True)
    assert "Round figures" not in no_data and "recalculate" not in no_data, no_data

    # the user block leads with the question and always closes by asking for an answer
    body = user("Guntur, tomorrow: 1.9mm.", "will it rain", "Places: Guntur")
    assert body.startswith("They asked: will it rain"), body
    assert body.rstrip().endswith("in your own words."), body
    assert "Conclusion:" in body and "Retrieved:" in body, body
    assert "Retrieved" not in user("x", "y"), "no context, no heading for it"
    assert "Note:" in user("x", "y", answering=False)

    print("\nprompts demo OK - four shapes, every block reachable, figure rules answering-only")

def check_grounding():
    """Self-check: the cleaning rules offline, then one live call if Ollama is up."""
    import asyncio

    from backend.generation.llm import (
        MAX_REPLY_CHARS,
        TROUBLE_LINES,
        _rounds_to,
        clean,
        echoed,
        explain,
        figures_in,
        first_sentence,
        grounded,
        invented,
        reverses,
        say,
        scaffolded,
        trouble_line,
        usable,
    )
    assert clean("<think>hmm</think>\nRain tomorrow.") == "Rain tomorrow."
    assert clean('```\n"Rain tomorrow."\n```') == "Rain tomorrow."
    assert clean("x" * (MAX_REPLY_CHARS + 1)) == ""          # a lecture, not a reply
    assert clean("x" * MAX_REPLY_CHARS) != ""                # ...but four sentences fit
    assert clean("") == ""
    assert first_sentence("Did you mean Vedurumudi? It is in East Godavari.") == \
        "Did you mean Vedurumudi?"

    # the no-data guard: weather or a figure the note did not carry is invented
    assert invented("It will rain tomorrow.", "I could not find that place.")
    assert invented("Try again in 5 minutes.", "I could not find that place.")
    assert not invented("I could not find that place.", "I could not find that place.")

    # The fact guard. A figure it was given, or that figure rounded the way people speak, is
    # fine; anything else is a fabricated forecast in a confident sentence.
    given = "Visakhapatnam Rural has the higher rainfall for this week: 12.5mm against Guntur 9.9mm."
    assert grounded("Vizag is wetter this week, 12.5mm to Guntur's 9.9mm.", given)
    assert grounded("Vizag is the wetter of the two this week.", given)      # no figures at all
    assert grounded("Vizag gets about 13mm this week, Guntur about 10mm.", given)   # rounded
    assert not grounded("About 3mm of rain is expected in Visakhapatnam Rural.", given)
    assert not grounded("Vizag gets 14mm this week.", given)                 # too far to be 12.5
    assert not grounded("Rain starts at 2pm.", given)                        # nowhere in scope

    # precision sets the tolerance: a whole number may stand for anything within half of one,
    # a tenth may only stand for something within a twentieth
    assert _rounds_to("2", 1.93) and _rounds_to("1.9", 1.93)
    assert not _rounds_to("2.4", 1.93)
    assert grounded("About 2mm tomorrow.", "Guntur, tomorrow: 1.93mm.")
    assert not grounded("About 2.4mm tomorrow.", "Guntur, tomorrow: 1.93mm.")
    assert grounded("In the low twenties, about 26 degrees.", "25.3-27.2°C (avg 26.3°C)")

    assert figures_in("25.3-27.2°C (avg 26.3°C)") == {"25.3", "27.2", "26.3"}

    # The reversal gate. This is the one that is not about style: asked whether clothes would
    # dry, a 1b handed "No - 2.4mm expected from 06:00" answered "making it ideal for drying
    # clothes", and somebody hangs washing out on that.
    verdict_line = "No - 2.4mm expected from 06:00. Guntur, today: rain on 1 of 23 readings."
    context = verdict_line + " Decision: No - 2.4mm expected from 06:00"
    assert reverses("...making it ideal for drying clothes.", "NO")
    assert reverses("Hold off on spraying today.", "YES")
    assert not reverses("You will want to wait - rain from six.", "NO")
    # a caution is flattened, not reversed: dropping the condition is dropping the answer
    assert reverses("You should harvest now.", "CAUTION")
    assert not reverses("You can harvest, but the dry spell is only two days.", "CAUTION")
    assert not reverses("Tight - watch the rain on Thursday.", "CAUTION")
    assert not reverses("anything at all", ""), "not an advice turn, nothing to reverse"

    # ...and the one gate every caller goes through
    assert usable("...making it ideal for drying clothes.", verdict_line, context, "NO") == ""
    assert usable(verdict_line, verdict_line, context) == "", "an echo is not a rewrite"
    assert usable("The data shows 2.4mm of rain.", verdict_line, context) == ""
    assert usable("", verdict_line, context) == ""
    kept = "You will want to wait - about 2.4mm is coming from six, so nothing dries today."
    assert usable(kept, verdict_line, context + " 2.4", "NO") == kept

    assert scaffolded("According to the forecast, rain.") and not scaffolded("Rain tomorrow.")
    # the model narrating its own brief is the same failure wearing different words
    assert scaffolded("The conclusion is correct based on the retrieved data.")
    assert scaffolded("The evidence shows 2mm.")

    assert trouble_line("locations") == TROUBLE_LINES["location"]
    assert trouble_line("fetch") == TROUBLE_LINES["data"]
    assert trouble_line("nonsense") == TROUBLE_LINES["unknown"]
    # a bare failure never reaches the model - there is nothing to augment it with
    assert asyncio.run(explain("fetch")) == TROUBLE_LINES["data"]

    assert asyncio.run(say("")) == ""                         # empty in, empty out, no call
    line = ("Safe to spray. Wind speed in Guntur for tomorrow: 1.2-3.4m/s (avg 2.1m/s). "
            "(No exact match for \"guntur\" - showing the closest, Guntur, Andhra Pradesh.)")
    said = asyncio.run(say(line))
    print(f"  in : {line}\n  out: {said}")
    assert echoed(line, line) and not echoed("Rain tomorrow.", line)
    if said != line and echoed(said, line):
        print("  WARNING: the model echoed its input instead of rewriting it - check the "
              "prompt and OLLAMA_MODEL")

    # the retrieved sections must reach the answer: the conclusion names one place, the reply
    # has to be able to name the other
    both = asyncio.run(say(
        "Hyderabad has a higher minimum temperature of 21.3C compared to Vijawada's 20.5C.",
        "compare between hyderabad and vijawada",
        context="Places: Hyderabad (Telangana), Vijawada (Andhra Pradesh)\n"
                "Period: next 7 days (daily readings)\n"
                "Comparison: Hyderabad leads Vijawada by 0.8°C on min temp\n"
                "Range:\n- Hyderabad: average min temp 21.3°C, range 20.6-21.9°C\n"
                "- Vijawada: average min temp 20.5°C, range 19.7-20.9°C"))
    print(f"  both: {both}")
    if said != line:                                          # ollama answered - a real test
        assert "vijawada" in both.lower(), both
    print("generation demo OK" + (" (ollama offline - passed through)" if said == line else ""))

def main():
    """Every check in this file, in order. Any assertion failure stops it."""
    for check in (check_context, check_prompts, check_grounding,):
        print(f"{check.__name__}:")
        check()
    print("\n3 check(s) passed")


if __name__ == "__main__":
    main()
