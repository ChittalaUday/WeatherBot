"""
Time expressions the rule tables do not know, canonicalised by a model.

    python tests/test_nlu_units.py          # the checks for this module
    ollama pull qwen3:1.7b                  # OLLAMA_MODEL and OLLAMA_URL in .env

`src.tagger.normalize_time`'s 248 hand-written entries still miss thirteen of the fourteen
phrasings taken from this deployment's own history ("prior days", "last summer", "so far
today"), and the miss is silent: an expression it cannot place passes through raw and the
resolver answers with next week under the user's own words. That table cannot be finished.

So the open half moves to a model and the closed half stays here:

    model   an arbitrary phrase  ->  one of the canonical forms      open vocabulary, semantic
    code    a canonical form     ->  absolute dates                  arithmetic, must be exact

A model that does date arithmetic will one day put "last 7 days" three days out and be
believed. **The model proposes; this module disposes** - whatever comes back is fed to
`resolve` and kept only if `resolve` recognises it.
"""

from __future__ import annotations

import json
import re

import httpx

from backend.config import OLLAMA_MODEL, OLLAMA_THINK, OLLAMA_URL
from backend.nlu import duckling
from backend.pipeline.timewindow import parse_dates, resolve
from src.tagger import normalize_time

# Every phrase ever asked about. Novel wording costs one call for the life of the process.
_SEEN: dict = {}

# A reasoning model spends its budget on reasoning first, and the reasoning grows with the
# span count: measured on qwen3:1.7b, three spans thought for ~1.1k characters and ten for
# ~4.9k, truncating on one run and not the next. So the batch is bounded rather than the budget
# guessed. The bound stays with OLLAMA_THINK off - it is the net for switching it back on.
CHUNK, NUM_PREDICT = 3, 1200
# Longer than the wording layer's 20s: there a slow reply costs a blunter sentence, here it
# costs the dates. With reasoning off a call lands in ~0.15s.
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


# The trigger for reading the sentence: without one of these there is nothing to find.
_TIMEY = re.compile(
    r"\b(now|today|tonight|tomorrow|tommorow|tommorrow|tmrw|yesterday|last|next|past|prior|"
    r"previous|earlier|later|recent|history|historical|ago|since|till|until|"
    r"hour|hours|day|days|week|weeks|month|months|year|years|weekend|"
    r"morning|afternoon|evening|night|noon|midnight|summer|winter|monsoon|"
    r"mon|tue|wed|thu|fri|sat|sun|am|pm|"
    # Full month names and a bare year, both of which the abbreviations miss: `\bjun\b` does
    # not fire inside "june", so "rain in june" and "rain for all of 2023" were gated out
    # before anything could read them and answered with the default horizon.
    r"january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|jun|jul|aug|sep|oct|nov|dec)\b"
    r"|\b(?:19|20)\d{2}\b"
    # "6pm" is one token, so `\bpm\b` never fires inside it and "between 6pm and 9pm" was
    # gated out before anything could read it.
    r"|\d{1,2}\s*(?:am|pm)\b|\d{1,2}:\d{2}"
    # "at 6" - a bare hour, anchored to the preposition so "rain over 5 acres" is not a clock
    # time. The lookahead is not optional: "soil moisture at 5 cm" is a depth, and Duckling
    # reads "at 5" out of it as five in the afternoon. `%` sits outside the `\b` group because
    # it is not a word character, so "humidity at 80 %" slipped straight through.
    r"|\bat\s+\d{1,2}\b(?!\s*(?:%|(?:cm|mm|km/?h|kmph|kph|m/s|mps|knots?|kg|ha|deg|"
    r"degrees?|acres?|ft|feet|inch|inches|litres?|hectares?|m)\b))", re.I)

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

    The span tagger is the half that fails: "can i know yesterday rainfall" returned no span,
    and "last summer" gave it "last", which normalises to "last 7 days". Neither is fixable by
    a bigger alias table. Whatever the model says still has to survive `known()`.
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
    _SEEN[text] = await placeable(candidate, client=client)
    return _SEEN[text]


async def place(span: str, text: str, client: httpx.AsyncClient | None = None,
                now=None, hint: str = "") -> tuple:
    """The period this turn is about: `(canonical, how)`. The one entry point.

    Duckling first, on the sentence. It reads the whole thing, answers in single-digit
    milliseconds and hands back absolute dates, so nothing downstream has to own a calendar.
    The classifier's own time slot is not consulted while Duckling has an answer - it is a
    span tagger, and a truncated span ("last summer" -> "last") normalises to a confident
    wrong window.

        duckling    a grammar read the sentence                     one call, ~5ms
        tables      duckling read nothing; the span normalises, and
                    duckling places the canonical form it becomes   one more call
        model       neither could, so a model read it               one call, cached
        none        no period is named, and none is needed          no call
        unplaceable something was named and nothing could place it  the turn stops

    Every `time_resolution` correction in the feedback table is a turn that should have
    returned "unplaceable" and answered with next week instead.
    """
    if not span and not mentions_time(text):
        return "", "none"

    # Two written dates are read here, not by Duckling: it returns the first of "11 jan 2026
    # and 17 jan 2026" and drops the six days in between, and it says so confidently.
    if len(written := parse_dates(text)) >= 2:
        return f"{min(written)} to {max(written)}", "dates"

    # `hint` is the same call already made beside the classifier, so its round trip costs
    # nothing on the wall clock. `known()` still gates it - Duckling proposes, `resolve`
    # disposes.
    placed = hint or await duckling.canonical(text, now, client=None)
    if placed and known(placed):
        return placed, "duckling"

    # Duckling read nothing. The tables are the spelling half it has no rules for - "whole
    # day", "aaj", "2moro", "past records" - and they normalise to a vocabulary Duckling does
    # place (29 of its 30 forms, measured), so the arithmetic stays in one place.
    if span and (placed := await placeable(normalize_time(span), now, client)):
        return placed, "tables"

    # The gap list, read straight off the sentence: "for all of 2023", "from 2010 to 2025",
    # "the last decade", "early morning" - the forms `resolve` keeps precisely because
    # Duckling has no rule for them. Cheaper than a model call, and it cannot invent a window.
    if known(text):
        return text, "edge"

    # a span that exists but cannot be placed is still the best clue there is - try it alone
    # before the sentence, because it is smaller and the model is likelier to get it right
    if span:
        placed = (await canonicalize([span], client)).get(span, "")
        if placed:
            return placed, "model"
    placed = await for_question(text, client)
    return (placed, "model") if placed else ("", "unplaceable")


async def placeable(form: str, now=None, client: httpx.AsyncClient | None = None) -> str:
    """A proposed wording -> the absolute form it means, or "" if nothing can place it.

    The one gate every proposer goes through - the tables, the model, and Duckling's own
    answer. Anything already absolute (or on `resolve`'s gap list) passes straight through;
    anything still in words is handed to Duckling, which owns the arithmetic.
    """
    form = (form or "").strip().lower()
    if not form:
        return ""
    if known(form):
        return form
    placed = await duckling.canonical(form, now, client)
    return placed if placed and known(placed) else ""


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
            # An empty result is a failed call, not ten unplaceable phrases. Caching "" for
            # each would refuse those words for the life of the process.
            if not placed:
                continue
            for span in chunk:
                candidate = str(placed.get(span, "") or "").strip().lower()
                # the model proposes, Duckling and the resolver dispose - anything neither can
                # place is remembered as "could not place" rather than believed
                _SEEN[span] = await placeable(candidate, client=client)
                if _SEEN[span]:
                    out[span] = _SEEN[span]
    finally:
        if owns:
            await client.aclose()
    return {k: v for k, v in out.items() if v}


async def _ask(client: httpx.AsyncClient, payload, system: str = "",
               think: bool | None = None) -> dict:
    """One call. {} when it failed - never raises, so a turn is never blocked on this."""
    think = OLLAMA_THINK if think is None else think
    try:
        # The local model already running for the wording layer - no quota, no per-turn cost.
        # `format: "json"` constrains the decode; `_json_from` stays because a reasoning model
        # can still wrap it in a <think> block. `think` is sent explicitly and never omitted:
        # measured on qwen3:1.7b it cost 1.95s against 0.15s for the same answer every time.
        response = await client.post(
            f"{OLLAMA_URL}/api/chat", timeout=TIMEOUT,
            json={"model": OLLAMA_MODEL, "stream": False, "format": "json",
                  "think": think,
                  "options": {"temperature": 0, "num_predict": NUM_PREDICT},
                  "messages": [{"role": "system", "content": system or SYSTEM},
                               {"role": "user", "content": payload if isinstance(payload, str)
                                else json.dumps(payload)}]})
        response.raise_for_status()
        return _json_from(response.json()["message"]["content"])
    except httpx.HTTPStatusError as exc:
        # A model with no reasoning mode 400s on `think: true`. Swallowed, that turns every
        # time expression unplaceable, so it downgrades once. `think: false` never 400s, so
        # the retry cannot loop.
        if think and exc.response.status_code == 400:
            return await _ask(client, payload, system, think=False)
        return {}
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
