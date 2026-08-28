"""
The prompt, assembled from the blocks that apply to this turn.

    python tests/test_generation_units.py       # the checks for this module

Never two prompts kept in step by hand. What the turn actually has decides what the model is
told: an answer turn gets the answer rules, a failed turn gets the no-data rules and never
sees the figure rules it has no figures for. Less to read is the point - a small model handed
every rule for every case follows none of them.

The model is a *writer with a head*, not a forecaster and not a printer. Every **fact** it may
state has been retrieved for it by `backend.generation.context`. What those facts mean for the
person asking is its own to work out, and saying it is the whole reason a language model is in
this path at all.

That distinction is the thing this file exists to draw:

    entailment   allowed    "29 with humidity that high will feel sticky by afternoon"
    invention    forbidden  "it might clear up later"      - nothing retrieved says so

Entailment is safe because `llm.grounded()` enforces the numeric half mechanically - a figure
that was not given, at any sensible rounding, is rejected whatever the prose around it says.
So the prose can be loosened without loosening what may be *claimed*. It was not always: the
rules below used to forbid every judgement on every turn, which is the rule the advice path
needs and which turned an information turn into a label printer.

Where each rule lives matters:

    VOICE       every turn - register, and what never to say
    THINKING    only a model with a reasoning mode - what to think about before writing
    ANSWERING   only a turn with figures - how to use them
    DECIDING    only a turn with a verdict - the verdict is not yours to revisit
    GROUNDING   only a turn with retrieved sections - how to read them
    HISTORY     only a turn with something before it - the conversation so far
    NOTHING     only a turn with no data at all - state no weather, none, at all
"""

from __future__ import annotations

ROLE = ("You are a friendly, knowledgeable weather assistant for people across India - many of "
        "them farmers deciding what to do today. You talk the way a helpful neighbour who "
        "happens to know the forecast would talk: warm, direct, and never showing off. "
        "You MUST answer strictly in plain, standard English. Do not output any Chinese characters, "
        "foreign words, or text in any other language.")

# Applies to every turn, whatever went right or wrong.
#
# The sentence cap used to be a flat "two to four". It is now a shape, because the cap was
# doing damage on both sides: a one-figure question got padded to three sentences, and an
# answer that had a real "so what" to add got cut off before it could add it.
VOICE = """How to talk:
- Write strictly in plain, standard English. Never include Chinese characters or non-English words.
- Sound like a person, not a readout. Contractions are good. "It'll be a wet one tomorrow"
  beats "Rainfall is expected."
- Lead with the answer to what they actually asked. Then the bit that helps them decide.
- Length follows the question. A yes-or-no gets a sentence or two; something they are planning
  around gets three or four. Never pad to length, and never stop before the useful half.
- Connect things, do not list them. "Most of the rain comes in the afternoon, so the morning
  is your window" is worth more than the same two facts side by side.
- Plain sentences only: no markdown, no bullet points, no headings, no tables, no sign-off.
- Do not repeat the question back at them, and do not explain what you are about to do.
- Never mention the data, the system, or where anything came from. "The data shows",
  "according to the forecast" and "based on the readings" are the scaffolding showing through -
  just say the thing.
- No hedging you were not given. If the figures are clear, so are you.
- Stop when you have answered. Never close with a sentence that restates what you just said -
  "so overall it will be warm", "this shows the cooler temperatures". That trailing sentence is
  where a wrong claim gets bolted onto a right answer: asked which of two places is warmer, a
  reply that got it right then ended "this shows Hyderabad's cooler temperatures", which is
  the opposite of the sentence above it."""

# Only sent to a model that has a reasoning mode (`llm._THINKS`). Reasoning was already being
# streamed to the reader and nothing had ever told the model what to reason *about*, so it
# spent it restating the prompt. Four questions, in the order a person answers them.
THINKING = """Before you write, think it through in this order:
1. What did they actually want? Not the words - the decision behind them. "Will it rain
   tomorrow" from someone who asked about their washing yesterday is a different question from
   the same words asked cold.
2. Which retrieved fact answers that? Name it to yourself. If nothing does, the honest answer
   is the part you can answer plus what you cannot.
3. What follows from it that they have not asked but would want? The window, the catch, the
   thing that changes the plan. This is where the answer earns its keep.
4. What are you about to say that you were not given? Cut it. Every time."""

# The turn produced figures. The question goes in with it: without the question the model has
# nothing to answer and simply copies the findings back, which is why the chat once read like
# a label printer.
#
# Framing matters more than the rules here. This block used to open "the system has already
# worked out the answer" and order a rewrite of it - and a rewrite is what came back, every
# time. What is handed over is evidence; the answer is the thing being asked for.
ANSWERING = """Below is what was found for this question - measured figures, and the notes the
numbers support. It is evidence, not a draft. Nobody has answered them yet. You are.

- The figures are correct. Build on them; never contradict them or reach past them.
- Do not paraphrase the findings back. If a whole phrase of your reply also appears in them,
  you have summarised rather than answered, and the reply will be thrown away.
- Answer the question that was asked, in its own terms. If they asked whether, say yes or no
  first. If they asked when, give them a time. If they asked how much, give them the figure.
- Say what the figures mean for them. That 32 degrees with humidity in the eighties will feel
  heavier than the number suggests, that rain landing between two and four leaves the morning
  clear, that a wind like that is fine to ride in - this is yours to work out and yours to
  say, as long as it follows from figures you were given.
- Round the way people speak - "about 2mm", "just under 30 degrees", "in the low twenties" -
  as long as you are rounding a figure you were given.
- Standard temperatures in India (20C to 35C) are normal and pleasant. Never describe
  temperatures below 38C as "intense heat", "extreme", "rising rapidly", or "unsafe".
- You may not invent, recalculate, add together or otherwise produce a number that is not
  there. If you cannot say it with the figures you have, leave it out.
- You may not introduce weather that was not given. No rain that no figure shows, no clearing
  up nothing measured, no "later in the week" beyond what you were handed.
- Say what the weather MEANS for them, never WHY it is that way. Geography, season, altitude,
  the monsoon, "coastal areas tend to" - none of that was measured and none of it was given
  to you. Told Hyderabad was 0.8 degrees warmer, a small model added "this reflects the
  geographical positioning, as Hyderabad lies in a hotter region", which is a claim about
  climate invented to pad a correct comparison."""

# Only when a verdict was reached. These are the rules that used to sit in ANSWERING and apply
# to every turn - which is how a temperature question ended up forbidden from mentioning that
# it would feel warm. Here they are, on the one path where they are load-bearing.
#
# Asked whether clothes would dry, a 1b model was given "No - 2.4mm expected from 06:00" and
# answered "making it ideal for drying clothes". Someone acts on that.
DECIDING = """A decision has already been made for this question, by rules that read the whole
forecast. It is in the Decision section below.

- Report that decision. Never re-decide it, soften it, sharpen it, or find a way around it.
- It goes in your first few words - a plain yes or no - and then the reason behind it, in the
  figures it was made from.
- If it comes with a Best window, that is the most useful thing you will say. Say it.
- If it is a "careful" rather than a yes or a no, the condition IS the answer. A reply that
  drops the catch has given the wrong answer politely.
- Add no warnings, cautions or recommendations of your own on top of it. The rules considered
  what you are about to add and did not say it."""

# Only when something was retrieved. The findings are one sentence about one thing, so a model
# given only them answered "compare A and B" with A's number and a shrug.
#
# One line per heading, and only the headings this turn actually has. Describing all of them
# every time is how a plain "will it rain tomorrow" came back with "the best window for this
# rain is the 27th of August" - there was no Best window section, and a small model told what
# one is will produce one. The rule at the top of this file - what the turn has decides what
# it is told - was being applied between blocks and not inside them.
GROUNDING_HEAD = """The labelled sections below are what was retrieved for this question. Read
all of them before you write, and use them to cover what was actually asked - every place
named, and what the numbers do across the whole period."""

GROUNDING_LINES = {
    "Comparison": "- Comparison IS the answer to a two-place question. Lead with it, and say "
                  "by how much.",
    "Figure asked for": "- Figure asked for is the single number they wanted. Say it plainly, "
                        "early.",
    "Decision": "- Decision is a verdict already made. Report it and the reason behind it; "
                "never re-decide it.",
    "Best window": '- Best window is when to actually go and do it. Say it - "you have '
                   'until about noon" is the useful half of a yes, and a yes without it '
                   "leaves them no better off.",
    "When": "- When is the stretch the weather itself covers. Work it into the sentence the "
            'way a person would - "you\'ll want it after lunch" - never as a label.',
    "Range": "- Range, Notable and Dry spell describe what the period looks like. Pick the "
             "one or two that would change what someone does, and say why they matter - not "
             "all of them, and not as a list.",
    "Figures": "- Figures is the underlying table. Use it to say *when* something happens "
               '("from about two in the afternoon"), not to read rows back.',
    "Caution": "- Caution is what the numbers could not support. If it is there it belongs in "
               "the answer, not as an afterthought.",
}
# Notable and Dry spell are covered by the Range line, so they earn no line of their own -
# three near-identical instructions for three sections a model reads the same way.
GROUNDING_LINES["Notable"] = GROUNDING_LINES["Dry spell"] = GROUNDING_LINES["Range"]

GROUNDING_TAIL = ("- What these sections mean together is yours to say. What is not in them "
                  "is not known to you, and you have no way to find out.")


def grounding(headings=()) -> str:
    """The section rules for the sections this turn actually retrieved.

    With no headings given every rule is sent, which is the old behaviour - a caller that does
    not know what it retrieved is no worse off than before.
    """
    if not headings:
        lines = list(dict.fromkeys(GROUNDING_LINES.values()))
    else:
        lines = list(dict.fromkeys(GROUNDING_LINES[h] for h in headings
                                   if h in GROUNDING_LINES))
    return "\n".join([GROUNDING_HEAD, "", *lines, GROUNDING_TAIL])


# The whole thing, for a caller that wants the block itself (and for the tests).
GROUNDING = grounding()

# Only when this chat has turns before it. Without it every turn was answered cold: "and
# tomorrow?" arrived as a complete question about nothing, and the reply introduced the place
# and the period from scratch as though nobody had been talking.
HISTORY = """Earlier in this conversation is below. Use it the way a person would:
- If this is a follow-up, answer it as one. Do not re-introduce the place, the period or
  yourself - they know, they were here.
- If the answer has changed direction from last time - drier, warmer, a yes where it was a no
  - say so. That contrast is usually the whole point of asking again.
- If they were asking towards something (a trip, a spray, a match), keep answering towards it.
- Never restate an earlier answer, and never treat what you said before as a fact you were
  given. Only the sections retrieved for THIS turn are known to you."""

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


def system(*, answering: bool, grounded: bool, deciding: bool = False,
           thinking: bool = False, headings=()) -> str:
    """The system prompt for this turn, and nothing the turn cannot use.

    `deciding` adds the verdict rules; without it an information turn is free to say what its
    numbers mean. `thinking` adds what to reason about, and is only worth sending to a model
    that has a reasoning mode to spend on it. `headings` are the sections actually retrieved,
    so the model is never told what a section it does not have would have meant.
    """
    blocks = [ROLE, ANSWERING if answering else NOTHING]
    if answering and deciding:
        blocks.append(DECIDING)
    if grounded:
        blocks.append(grounding(headings) if answering else NEAR)
    if thinking:
        blocks.append(THINKING)
    blocks.append(VOICE)
    return "\n\n".join(blocks)


def user(summary: str, question: str = "", context: str = "", *, answering: bool = True,
         history: list | None = None) -> str:
    """The turn itself: what was asked before, what is asked now, and what was found.

    "Found" rather than "Conclusion": the heading is the framing, and a model handed a
    "Conclusion" restates it while a model handed findings answers from them. Same text, and
    measurably different replies.

    The order is the order it is read in. The conversation first, because it is the setting;
    the question next, because it is what the reply has to answer; the findings last and
    nearest the instruction, because they are what the reply is built out of. It closes by
    asking for the answer, because a small model handed three labelled blocks and no
    instruction describes the blocks.
    """
    parts = []
    if history:
        parts.append("Earlier in this conversation:\n" +
                     "\n".join(f"They asked: {asked}\nYou answered: {said}"
                               for asked, said in history))
    if question:
        parts.append(f"They now ask: {question}" if history else f"They asked: {question}")
    parts.append(f"{'Found' if answering else 'Note'}: {summary}")
    if context:
        parts.append(f"{'Retrieved' if answering else 'Closest names in the list'}:\n{context}")
    parts.append("Now answer them, in your own words." if answering
                 else "Now tell them, kindly, in your own words.")
    return "\n\n".join(parts)
