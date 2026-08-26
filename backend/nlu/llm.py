"""
Model 3 - the same NLU contract, answered by a hosted LLM instead of a trained head.

    python -m backend.nlu.llm "should i spray fertilizer on my cotton field in Guntur tomorrow"

The point is comparison, not replacement. Models 1 and 2 are 20MB of TF-IDF and linear heads
that answer in single-digit milliseconds offline; this is a network round trip to a model with
no training on this label set at all, working purely from the schema in its prompt. Putting
the three side by side on the same sentence is the only honest way to know what the trained
models are worth - and where a general model is simply better, which is worth knowing too.

It returns the same `Understanding` the other two do, so everything downstream - the query
planner, the advice engine, the response builder - cannot tell which one answered.

The contract is enforced here, not hoped for: whatever the LLM emits is coerced onto the enums
and anything it invents is dropped. A model that returns `intent: "WEATHER_QUERY"` gets the
same treatment as one that returns nothing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time

import httpx

from backend.config import AI_API_KEY, AI_BASE_URL, AI_MODEL, AI_TIMEOUT
from src.tagger import normalize_time
from src.v4.entities import extract as extract_entities
from src.v4.schema import (
    CONTROL,
    DECLINED,
    NO_DATA_NEEDED,
    REPLIES,
    Activity,
    Aggregation,
    Intent,
    Variable,
    sub_activity_for,
    weather_intent_for,
)

NAME = "Model 3"
VERSION = "llm"

# The whole schema, because the model has never seen this label set. Kept terse: every token
# here is paid for on every turn.
SYSTEM = f"""You extract structured slots from weather questions. Reply with ONLY a JSON object.

intent (pick one):
  INFORMATION - asks for readings   ADVICE - asks whether to do something   COMPARISON - two places
  GREETING THANKS GOODBYE SMALL_TALK CAPABILITY - chat, no weather needed
  CHANGE_LOCATION RESET AFFIRM DENY EXPLAIN - acts on the conversation
  UNSUPPORTED_METRIC - weather-shaped but we only have {', '.join(v.value for v in Variable)}
  OUT_OF_SCOPE - not weather at all    UNCLEAR - too vague to act on

variables (list, only if intent is INFORMATION/ADVICE/COMPARISON): {', '.join(v.value for v in Variable)}
  UV has no sensor; use it only when the user asks about sun strength.

activity (only if intent is ADVICE, else NONE): {', '.join(a.value for a in Activity)}
  SPRAY vs FERTILIZE is decided by the VERB, not the material: "spray fertilizer" is SPRAY.

aggregation: {', '.join(a.value for a in Aggregation)} - RAW unless a reduction is spoken aloud.

locations: place names EXACTLY as written in the text, verbatim substrings. [] if none named.
  Never include words like "whole", "entire", "my field", "here" - those are not places.
times: time expressions EXACTLY as written, verbatim substrings. [] if none.

Example:
{{"intent":"ADVICE","variables":["WIND","RAIN"],"activity":"SPRAY","aggregation":"RAW",
"locations":["Guntur"],"times":["tomorrow"]}}"""


def _coerce(payload: dict, text: str) -> dict:
    """Force whatever came back onto the contract. Anything unrecognised is dropped."""
    def enum_of(cls, value, fallback):
        try:
            return cls(str(value).strip().upper())
        except (ValueError, AttributeError):
            return fallback

    intent = enum_of(Intent, payload.get("intent"), Intent.INFORMATION)
    activity = enum_of(Activity, payload.get("activity"), Activity.NONE)
    if intent is not Intent.ADVICE:
        activity = Activity.NONE

    variables = []
    for name in payload.get("variables") or []:
        try:
            variables.append(Variable(str(name).strip().upper()))
        except ValueError:
            continue

    # Rule 4.1: a span the model invented is worse than no span. Only verbatim survives.
    lowered = text.lower()
    spans = lambda key: [str(s) for s in (payload.get(key) or []) if str(s).lower() in lowered]
    locations, times = spans("locations"), spans("times")

    if intent in NO_DATA_NEEDED:
        variables, times = [], []
        if intent is not Intent.CHANGE_LOCATION:
            locations = []

    return {
        "intent": intent, "activity": activity, "variables": variables,
        "aggregation": enum_of(Aggregation, payload.get("aggregation"), Aggregation.RAW),
        "locations": locations, "times": times,
    }


def _json_from(reply: str) -> dict:
    """The first JSON object in the reply. Models wrap it in prose or fences often enough."""
    reply = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.M).strip()
    try:
        return json.loads(reply)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", reply, re.S)
        if not match:
            return {}
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}


async def understand(text: str, client: httpx.AsyncClient | None = None) -> dict:
    """One turn, in the shape registry.Understanding is built from.

    Never raises: a failed call returns `ok=False` with the reason, because this sits beside
    two models that always answer and a comparison with an empty column is still a comparison.
    """
    if not AI_API_KEY:
        return {"ok": False, "error": "API_KEY is not set", "latency_ms": 0}

    started = time.perf_counter()
    owns = client is None
    client = client or httpx.AsyncClient(timeout=AI_TIMEOUT)
    try:
        # the hosted endpoint rate-limits, and a 429 beside two models that always answer
        # would read as "the LLM got it wrong" rather than "the LLM was not asked"
        for attempt in range(3):
            response = await client.post(
                f"{AI_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"},
                json={"model": AI_MODEL, "temperature": 0, "max_tokens": 300,
                      "messages": [{"role": "system", "content": SYSTEM},
                                   {"role": "user", "content": text}]},
            )
            if response.status_code != 429:
                break
            await asyncio.sleep(1.5 * (attempt + 1))
        if response.status_code == 429:
            return {"ok": False, "error": "rate limited by the provider - try again in a moment",
                    "latency_ms": int((time.perf_counter() - started) * 1000)}
        response.raise_for_status()
        body = response.json()
        reply = body["choices"][0]["message"]["content"]
        parsed = _coerce(_json_from(reply), text)
        latency = int((time.perf_counter() - started) * 1000)

        times_normalized = [n for n in (normalize_time(s) for s in parsed["times"]) if n]
        entities = extract_entities(text)
        return {
            "ok": True, "latency_ms": latency, "usage": body.get("usage", {}),
            "raw": reply[:400],
            "intent": parsed["intent"].value,
            # a turn that needs no weather has no window - weather_intent_for(None) would say
            # FORECAST, which is the right default for a question and wrong for "hey there"
            "weather_intent": ("NONE" if parsed["intent"] in NO_DATA_NEEDED else
                               weather_intent_for(times_normalized[0] if times_normalized
                                                  else None).value),
            "activity": parsed["activity"].value,
            "sub_activity": sub_activity_for(parsed["activity"], entities, text),
            "variables": [v.value for v in parsed["variables"]],
            "aggregation": parsed["aggregation"].value,
            "locations": parsed["locations"],
            "times": parsed["times"],
            "times_normalized": times_normalized,
            "entities": entities,
            "family": ("control" if parsed["intent"] in CONTROL else
                       "declined" if parsed["intent"] in DECLINED else
                       "conversational" if parsed["intent"] in NO_DATA_NEEDED else "data"),
            "reply": (REPLIES.get(parsed["intent"]) or [""])[0]
                     if parsed["intent"] in NO_DATA_NEEDED else "",
        }
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}",
                "latency_ms": int((time.perf_counter() - started) * 1000)}
    finally:
        if owns:
            await client.aclose()


def available() -> bool:
    return bool(AI_API_KEY)


def model_name() -> str:
    return AI_MODEL
