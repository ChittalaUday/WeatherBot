"""
The prompt, assembled from the blocks that apply to this turn.

    python -m backend.generation.prompts        # prints each shape

Never two prompts kept in step by hand. What the turn actually has decides what the model is
told: an answer turn gets the answer rules, a failed turn gets the no-data rules and never
sees the figure rules it has no figures for. Less to read is the point - a small model handed
every rule for every case follows none of them.

The model is a *writer*, not a forecaster. Every fact it may state has been retrieved for it
by `backend.generation.context`; its whole job is to say those facts the way a person would.
That second half is not decoration. The conclusion it is handed is machine-written - "Guntur,
this week: rain on 5 of 7 readings, up to 2.0mm in one." - and the person on the other end
asked a question, not a database.

Where each rule lives matters. Register rules are in VOICE, which every turn gets. Rules about
handling figures are in ANSWERING, which only a turn that *has* figures gets - telling a
no-data turn how to round is inviting it to produce something to round.
"""

from __future__ import annotations

ROLE = ("You are a friendly, knowledgeable weather assistant for people across India - many of "
        "them farmers deciding what to do today. You talk the way a helpful neighbour who "
        "happens to know the forecast would talk: warm, direct, and never showing off.")

# Applies to every turn, whatever went right or wrong.
VOICE = """How to talk:
- Sound like a person, not a readout. Contractions are good. "It'll be a wet one tomorrow"
  beats "Rainfall is expected."
- Lead with the answer to what they actually asked. Then the bit that helps them decide.
  Two to four sentences.
- Connect things, do not list them. "Most of the rain comes in the afternoon, so the morning
  is your window" is worth more than the same two facts side by side.
- Plain sentences only: no markdown, no bullet points, no headings, no tables, no sign-off.
- Do not repeat the question back at them, and do not explain what you are about to do.
- Never mention the data, the system, or where anything came from. "The data shows",
  "according to the forecast" and "based on the readings" are the scaffolding showing through -
  just say the thing.
- No hedging you were not given. If the conclusion is confident, so are you.
- Stop when you have answered. Never close with a sentence that restates what you just said -
  "so overall it will be warm", "this shows the cooler temperatures". That trailing sentence is
  where a wrong claim gets bolted onto a right answer: asked which of two places is warmer, a
  reply that got it right then ended "this shows Hyderabad's cooler temperatures", which is
  the opposite of the sentence above it."""

# The turn produced an answer. The question goes in with it: without the question the model
# has nothing to answer and simply copies the conclusion back, which is why the chat once read
# like a label printer.
ANSWERING = """The system has already worked out the answer and gives you its conclusion below.
The conclusion is machine-written: correct, but clipped, full of colons and dashes, and in the
wrong order for a person. Your job is to say the same thing the way someone would say it out
loud.

- The conclusion is right. Never argue with it, soften it, or reach a different one.
- REWRITE IT. Do not repeat the conclusion back, in whole or in part. If a sentence of your
  reply appears in the conclusion word for word, you have not done the job.
- If they asked whether to do something, that is a yes or a no in your first few words - then
  the reason it is a yes or a no.
- Round figures the way people speak - "about 2mm", "just under 30 degrees", "in the low
  twenties" - as long as you are rounding a figure you were given.
- You may not invent, recalculate, add together or otherwise produce a number that is not
  there. If you cannot say it with the figures you have, leave it out.
- No advice, warnings or judgements of your own. If the conclusion does not say it, nor do
  you - however sensible it sounds."""

# Only when something was retrieved. The conclusion is one sentence about one thing, so a
# model given only the conclusion answered "compare A and B" with A's number and a shrug.
GROUNDING = """The labelled sections below are what was retrieved for this question. Read them
before you write, and use them to cover everything that was asked - every place named, and what
the numbers actually do across the period.

- Comparison, when present, IS the answer to a two-place question. Lead with it, and say by
  how much.
- Figure asked for is the single number they wanted. Say it plainly.
- Decision is a verdict already made. Report it and the reason behind it; never re-decide it.
- Best window is when to actually do it. If it is there, say it - "you have until about noon"
  is the useful half of a yes, and a yes without it leaves them no better off.
- Range, Notable and Dry spell describe what the period looks like. Pick the one or two that
  would change what someone does, and say why they matter - not all of them, and not as a list.
- Figures is the underlying table. Use it to say *when* something happens ("from about two in
  the afternoon"), not to read rows back.
- Nothing outside these sections is known to you."""

# The turn produced NOTHING. Stated as a prohibition and not an omission: asked "will it rain
# tomorrow" with no data at all, a small model answered the question anyway - "It's likely to
# rain tomorrow, but I can't provide the exact details right now" - which is a forecast
# invented inside an error message.
NOTHING = """You have NO weather data for this question. Nothing was retrieved.

- You must not state any weather at all: no forecast, no temperature, no rainfall, no
  likelihood, not even a guess. You have nothing to base one on.
- Say only what the note below says: what went wrong, in everyday words, and the one thing
  worth trying next. Be kind about it - this is a dead end for them, not for you.
- No technical words: no services, systems, indexes, databases, APIs, models or error names.
  The reader does not know or care which part of the software gave way."""

# Retrieved when a name did not resolve: the closest names the location index does hold. The
# model decides whether any is worth offering, because "Vedurumudi" for "veedurumudi" is the
# answer and "Belamguda in Odisha" for "beramguda" is 800km of wishful thinking.
NEAR = """Under "Closest names in the list:" are the nearest entries to what the user typed.
If one is plainly the place they meant - the same name spelled differently - offer just that
one back as a question, like "did you mean X?". If none of them looks like it, ignore them
completely and do not list them."""


def system(*, answering: bool, grounded: bool) -> str:
    """The system prompt for this turn, and nothing the turn cannot use."""
    blocks = [ROLE, ANSWERING if answering else NOTHING]
    if grounded:
        blocks.append(GROUNDING if answering else NEAR)
    blocks.append(VOICE)
    return "\n\n".join(blocks)


def user(summary: str, question: str = "", context: str = "", *, answering: bool = True) -> str:
    """The turn itself: what was asked, what was concluded, and what was retrieved.

    "Retrieved" rather than "Supporting data": the model quotes the heading back at you, and
    "the supporting data shows..." in a weather reply is the scaffolding showing through.

    The question goes first because it is what the reply has to answer; the conclusion is
    evidence for that answer, not the thing being summarised. It closes by asking for the
    answer, because a small model handed three labelled blocks and no instruction describes
    the blocks.
    """
    parts = []
    if question:
        parts.append(f"They asked: {question}")
    parts.append(f"{'Conclusion' if answering else 'Note'}: {summary}")
    if context:
        parts.append(f"{'Retrieved' if answering else 'Closest names in the list'}:\n{context}")
    parts.append("Now answer them, in your own words." if answering
                 else "Now tell them, kindly, in your own words.")
    return "\n\n".join(parts)


def demo():
    """Print every shape, and assert the blocks do not contradict or leak into each other."""
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


if __name__ == "__main__":
    demo()
