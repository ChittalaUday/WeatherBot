"""
Time expressions the rule tables do not know, canonicalised by a model.

    python tests/test_nlu_units.py          # the checks for this module
    ollama pull qwen3:1.7b                  # OLLAMA_MODEL and OLLAMA_URL in .env

`src.tagger.normalize_time` holds 248 hand-written entries - 132 aliases, 102 span quantifiers,
14 words for "the past" - and every one of them was added after somebody's question came back
wrong. It still misses thirteen of the fourteen phrasings taken from this deployment's own
history: "prior days", "previous days", "last few days", "earlier today", "so far today",
"since morning", "the other day", "couple of days back", "last summer", "this time last year".

That table cannot be finished. English has no closed set of ways to say "the last few days",
so a lookup plus a difflib fallback will always be one bug report behind, and the bug it is
behind is silent - an expression it cannot place passes through raw, and the window resolver
answers with next week under the user's own words.

So the open half moves to a model and the closed half stays here:

    model   an arbitrary phrase  ->  one of the canonical forms      open vocabulary, semantic
    code    a canonical form     ->  absolute dates                  arithmetic, must be exact

The split is not a preference. A model that does date arithmetic is a model that will one day
put "last 7 days" three days out and be believed, because the answer will look like every
other answer. `backend.pipeline.timewindow.resolve` keeps that half and always will.

**The model proposes; this module disposes.** Whatever comes back is fed to `resolve` and kept
only if `resolve` recognises it. A model free to invent a canonical form would simply move the
silent-wrong-answer one layer along, which is the failure this exists to close.
"""

from __future__ import annotations

import json
import re

import httpx

from backend.config import OLLAMA_MODEL, OLLAMA_URL
from backend.pipeline.timewindow import resolve

# Every phrase ever asked about, and what it came back as. Novel wording costs one call for the
# life of the process and nothing after that; this deployment's whole history is 173 distinct
# questions, so the cache is small and the hit rate is high.
_SEEN: dict = {}

# A reasoning model spends the budget on reasoning FIRST, and the reasoning grows with the
# number of spans - so a budget that fits three spans returns nothing at all for ten, and
# `done_reason: "length"` with an empty content field looks exactly like a model that cannot
# do the task. Measured on qwen3:1.7b: three spans thought for ~1.1k characters, ten for
# ~4.9k, and the same ten-span call came back truncated on one run and complete on the next.
#
# So the batch is bounded instead of the budget being guessed. Small chunks keep the thinking
# short enough to be predictable, and a real turn carries one unplaceable span, not ten.
# `backend.generation.llm` gets away with a flat 700 because it phrases one answer; this one
# is handed a list.
CHUNK, NUM_PREDICT = 3, 1200
# The local model thinks before it answers and the budget above lets it. Twenty seconds is the
# wording layer's timeout, where a slow reply only costs a blunter sentence; here it costs the
# dates, so this waits longer.
TIMEOUT = 45.0

FORMS = """Allowed canonical forms, and nothing else:
  now, today, tonight, tomorrow, yesterday, day after tomorrow
  this morning, this afternoon, this evening, tomorrow morning, tomorrow afternoon,
  tomorrow evening, tomorrow night, early morning, midnight
  this week, next week, last week, this weekend, next weekend, this month, next month
  monday, tuesday, wednesday, thursday, friday, saturday, sunday
  next N hours, next N days, next N weeks        (N a whole number)
  last N days, last N weeks, last N months, last N years
  HH:MM              a wall-clock time, 24-hour
  HH:MM-HH:MM        a range of wall-clock times, 24-hour
  a month name, "<month> <year>", or "<day> <month> <year>"
  ""                 no time at all
"""

SYSTEM = """You convert time expressions into one canonical form. Reply with ONLY a JSON object
mapping each input string to its canonical form.

""" + FORMS + """
Rules:
- Past wording gets a past form. "prior days", "previous days", "the other day", "couple of
  days back", "so far today", "earlier" are all the PAST, never the next few days.
- Keep only the time words. "coming through tonight" is tonight. "for the next 5 hours or so"
  is next 5 hours.
- A vague past stretch with no number is "last 7 days". A vague future one is "next 7 days".
  But a number that IS there must survive: "past 3 days" is last 3 days, never last 7 days.
- Match the size of what was said. "so far today" is today, not a week. "this time last year"
  is last 12 months, not last week.
- Never invent a date, a month or a year the expression did not imply.

Example:
{"prior days":"last 7 days","coming through tonight":"tonight","at half five":"17:30"}"""


# Words that mean a time at all. The trigger for reading the sentence: without one of these
# there is nothing to find, and asking a model to look costs a second per turn for nothing.
_TIMEY = re.compile(
    r"\b(now|today|tonight|tomorrow|tommorow|tommorrow|tmrw|yesterday|last|next|past|prior|"
    r"previous|earlier|later|recent|history|historical|ago|since|till|until|"
    r"hour|hours|day|days|week|weeks|month|months|year|years|weekend|"
    r"morning|afternoon|evening|night|noon|midnight|summer|winter|monsoon|"
    r"mon|tue|wed|thu|fri|sat|sun|am|pm|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b", re.I)

QUESTION_SYSTEM = """You find the time period a weather question is asking about, and give it
in one canonical form. Reply with ONLY {"time": "<form>"}.

""" + FORMS + """

Rules:
- Read the WHOLE question. The period may be several words ("this time last year") or one
  ("tonight"), and it may be nowhere in the question at all.
- Past wording gets a past form. "prior days", "previous days", "the other day", "couple of
  days back", "did it rain", "history" are all the PAST, never the next few days.
- A number that is there must survive: "past 3 days" is last 3 days, never last 7 days.
- Match the size of what was said. "so far today" is today, not a week.
- If the question names no period at all, answer {"time": ""}.
- Never invent a date, a month or a year the question did not imply.

Examples:
{"time":"yesterday"}      for  can i know yesterday rainfall
{"time":"last 12 months"} for  rainfall in hyderabad last summer
{"time":"tonight"}        for  what's coming through tonight?
{"time":""}               for  will it rain in guntur"""


def mentions_time(text: str) -> bool:
    """Is there anything time-shaped in this sentence? The gate on reading it with a model."""
    return bool(_TIMEY.search(text or ""))


async def for_question(text: str, client: httpx.AsyncClient | None = None) -> str:
    """The canonical period this question asks about, read from the whole sentence.

    This exists because the span tagger is the half that fails now, not the normaliser. Handed
    "can i know yesterday rainfall" it returned no span at all, and "rainfall in hyderabad last
    summer" gave it "last" - which normalises to "last 7 days" and answers a question about a
    season with a week, confidently. Neither is fixable by a bigger alias table: a truncated
    span and a missing one both look like "no time was named" from downstream.

    So the model reads the sentence instead of the tagger's guess at part of it. Whatever it
    says still has to survive `known()`.
    """
    text = (text or "").strip()
    if not text or not mentions_time(text):
        return ""
    if text in _SEEN:
        return _SEEN[text]
    owns = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        placed = await _ask(client, text, QUESTION_SYSTEM)
    finally:
        if owns:
            await client.aclose()
    if not placed:
        return ""                       # a failed call is not an answer - do not remember it
    candidate = str(placed.get("time", "") or "").strip().lower()
    _SEEN[text] = candidate if known(candidate) else ""
    return _SEEN[text]


async def place(span: str, text: str, client: httpx.AsyncClient | None = None) -> tuple:
    """The period this turn is about: `(canonical, how)`. The one entry point.

    Rules first, always. They place "tomorrow", "next 3 days" and "last week" in microseconds
    and they are right about them; the model is for the tail they cannot reach, and only for
    that. Asked to place "the last 7 days" a 1.7b answered "last 7 weeks" - a form the
    resolver accepts and a window nobody asked for - so the cheapest protection against a
    small model is not asking it questions that are already answered.

        rules       the tables placed the span                      no call
        model       they could not, so the sentence was read        one call, cached
        none        no period is named, and none is needed          no call
        unplaceable something was named and nothing could place it  the turn stops

    "unplaceable" is the outcome that did not exist before. Every `time_resolution` correction
    in the feedback table is a turn that should have returned it and answered with next week
    instead, printing the user's own words over the wrong dates.
    """
    if span and known(span):
        return span, "rules"
    if not span and not mentions_time(text):
        return "", "none"
    # a span that exists but cannot be placed is still the best clue there is - try it alone
    # before the sentence, because it is smaller and the model is likelier to get it right
    if span:
        placed = (await canonicalize([span], client)).get(span, "")
        if placed:
            return placed, "model"
    placed = await for_question(text, client)
    return (placed, "model") if placed else ("", "unplaceable")


def known(canonical: str) -> bool:
    """Is this a form the deterministic resolver actually recognises?

    The one gate. `resolve` returning `understood=False` means it filled the window in with the
    default horizon, which is exactly the silent guess this module exists to stop.
    """
    return bool(canonical) and resolve(canonical).understood


async def canonicalize(spans: list, client: httpx.AsyncClient | None = None) -> dict:
    """{span: canonical} for the spans a model could place. Missing keys were not placed.

    Never raises and never blocks a turn: no key, no repair, and the caller says it could not
    work the dates out - which is the honest answer and the one the reader can act on.
    """
    wanted = [s for s in {(s or "").strip() for s in spans} if s]
    out = {s: _SEEN[s] for s in wanted if s in _SEEN}
    asking = [s for s in wanted if s not in _SEEN]
    if not asking:
        return {k: v for k, v in out.items() if v}

    owns = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT)
    try:
        for start in range(0, len(asking), CHUNK):
            chunk = asking[start:start + CHUNK]
            placed = await _ask(client, chunk)
            # An empty result is a call that failed, not ten phrases nobody can place.
            # Caching "" for each of them would refuse those words for the life of the
            # process - one truncated reply, and the repair is silently off from then on.
            if not placed:
                continue
            for span in chunk:
                candidate = str(placed.get(span, "") or "").strip().lower()
                # the model proposes, the resolver disposes - anything it does not recognise
                # is remembered as "could not place" rather than believed
                _SEEN[span] = candidate if known(candidate) else ""
                if _SEEN[span]:
                    out[span] = _SEEN[span]
    finally:
        if owns:
            await client.aclose()
    return {k: v for k, v in out.items() if v}


async def _ask(client: httpx.AsyncClient, payload, system: str = "") -> dict:
    """One call. {} when it failed - never raises, so a turn is never blocked on this."""
    try:
        # The local model, the one already running for the wording layer. No quota to run out
        # of and no per-turn cost, which matters because this is on the path of every question
        # carrying a phrase the tables do not hold.
        #
        # `format: "json"` is Ollama constraining the decode to valid JSON, so the reply is
        # parsed rather than fished out of prose. `_json_from` stays anyway: a reasoning model
        # can still wrap it in a <think> block.
        response = await client.post(
            f"{OLLAMA_URL}/api/chat", timeout=TIMEOUT,
            json={"model": OLLAMA_MODEL, "stream": False, "format": "json",
                  "options": {"temperature": 0, "num_predict": NUM_PREDICT},
                  "messages": [{"role": "system", "content": system or SYSTEM},
                               {"role": "user", "content": payload if isinstance(payload, str)
                                else json.dumps(payload)}]})
        response.raise_for_status()
        return _json_from(response.json()["message"]["content"])
    except (httpx.HTTPError, KeyError, ValueError, IndexError):
        # not running, too slow, or nonsense back. No repair, and the caller says it could not
        # work the dates out - which is the honest answer, not a guessed forecast.
        return {}


def _json_from(reply: str) -> dict:
    """The first JSON object in the reply. Models wrap it in prose or fences often enough."""
    reply = re.sub(r"^```(?:json)?|```$", "", (reply or "").strip(), flags=re.M).strip()
    try:
        parsed = json.loads(reply)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", reply, re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return parsed if isinstance(parsed, dict) else {}
