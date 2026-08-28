"""
The prompted classifiers - the same NLU contract, read by an LLM instead of a trained head.

    python -m backend.nlu.llm "should i spray fertilizer on my cotton field in Guntur tomorrow"

Two of them, local and hosted, differing only in where the request goes. The trained
classifier is 46MB of TF-IDF and linear heads that answers in single-digit milliseconds
offline; these are a round trip to a model with no training on this label set at all, working
purely from the schema in its prompt. Putting them side by side on the same sentence is the
only honest way to know what the trained one is worth - and where a general model is simply
better, which is worth knowing too.

They return the same `Understanding` the trained classifier does, so everything downstream -
the query planner, the advice engine, the response builder - cannot tell which one answered.

The contract is enforced here, not hoped for: whatever the LLM emits is coerced onto the enums
and anything it invents is dropped. A model that returns `intent: "WEATHER_QUERY"` gets the
same treatment as one that returns nothing.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import NamedTuple

import httpx

from backend.config import (
    AI_API_KEY,
    AI_BASE_URL,
    AI_MODEL,
    AI_TIMEOUT,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    OLLAMA_URL,
)
from backend.nlu.registry import Understanding
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

NAME = "Prompted classifier (hosted)"
VERSION = "llm"


class Spec(NamedTuple):
    """One OpenAI-shaped chat endpoint. The prompt and the coercion below are the same for
    every one of them - only where the request goes differs."""

    version: str
    name: str
    base_url: str
    model: str
    api_key: str
    timeout: float
    description: str
    extra: dict = {}                 # request fields only this endpoint understands


HOSTED = Spec("llm", NAME, AI_BASE_URL, AI_MODEL, AI_API_KEY, AI_TIMEOUT,
              f"{AI_MODEL} - hosted general model, no training on this label set, prompted "
              f"with the schema")
# Ollama speaks the OpenAI shape at /v1 and ignores the key, so the local model reaches the
# same client with nothing but a different address. It already words every reply; this is the
# same model asked to read the sentence instead, on demand, when the switch selects it.
#
# reasoning_effort=none is not a preference. A thinking model spends the whole max_tokens
# budget reasoning and gets cut off mid-JSON, so every turn fell back to the empty reading -
# and it is 8x slower doing it (2.8s vs 0.36s on gemma4:e2b). Ollama's OpenAI shim takes this
# field; the hosted endpoint is left alone because it does not.
LOCAL = Spec("ollama", "Prompted classifier (local)", f"{OLLAMA_URL}/v1", OLLAMA_MODEL,
             "ollama", OLLAMA_TIMEOUT,
             # the model id belongs here, not in the name: the name says what it does and is
             # stable, the id says which weights are behind it today and changes with .env
             f"{OLLAMA_MODEL} - the local model that words replies, doing the intent read too "
             f"- slower than the trained head, and it never saw this label set",
             {"reasoning_effort": "none", "response_format": {"type": "json_object"}})
SPECS = {spec.version: spec for spec in (HOSTED, LOCAL)}

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


async def chat_json(system: str, text: str, spec: Spec = HOSTED,
                    client: httpx.AsyncClient | None = None, max_tokens: int = 300) -> dict:
    """One JSON answer from an OpenAI-shaped endpoint. The transport, and nothing else.

    Never raises: a failed call comes back `ok=False` with the reason. Every caller here sits
    beside a deterministic path that always answers, so an exception would only ever be caught
    and turned back into this.
    """
    if not spec.api_key:
        return {"ok": False, "version": spec.version, "error": "API_KEY is not set",
                "latency_ms": 0}

    started = time.perf_counter()
    ms = lambda: int((time.perf_counter() - started) * 1000)
    owns = client is None
    client = client or httpx.AsyncClient(timeout=spec.timeout)
    try:
        # the hosted endpoint rate-limits, and a 429 beside a model that always answers would
        # read as "the LLM got it wrong" rather than "the LLM was not asked"
        for attempt in range(3):
            response = await client.post(
                f"{spec.base_url}/chat/completions",
                timeout=spec.timeout,
                headers={"Authorization": f"Bearer {spec.api_key}",
                         "Content-Type": "application/json"},
                json={"model": spec.model, "temperature": 0, "max_tokens": max_tokens,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user", "content": text}],
                      **spec.extra},
            )
            if response.status_code != 429:
                break
            await asyncio.sleep(1.5 * (attempt + 1))
        if response.status_code == 429:
            return {"ok": False, "version": spec.version, "latency_ms": ms(),
                    "error": "rate limited by the provider - try again in a moment"}
        response.raise_for_status()
        body = response.json()
        reply = body["choices"][0]["message"]["content"]
        return {"ok": True, "version": spec.version, "latency_ms": ms(),
                "usage": body.get("usage", {}), "raw": reply[:400], "json": _json_from(reply)}
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        return {"ok": False, "version": spec.version, "latency_ms": ms(),
                "error": f"{type(exc).__name__}: {str(exc)[:160]}"}
    finally:
        if owns:
            await client.aclose()


async def understand(text: str, client: httpx.AsyncClient | None = None,
                     spec: Spec = HOSTED) -> dict:
    """One turn, in the shape registry.Understanding is built from."""
    got = await chat_json(SYSTEM, text, spec, client)
    if not got.get("ok"):
        return got
    parsed = _coerce(got["json"], text)
    times_normalized = [n for n in (normalize_time(s) for s in parsed["times"]) if n]
    entities = extract_entities(text)
    return {
        **got,
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


def to_understanding(column: dict, text: str) -> Understanding | None:
    """A column from `understand` as the Understanding the pipeline consumes.

    Here rather than in a caller because both callers - the chat turn and the comparison -
    need exactly this, and downstream genuinely cannot tell which model it is running for.
    """
    if not column.get("ok"):
        return None
    return Understanding(
        text=text, version=column.get("version", VERSION), intent=column["intent"],
        action="COMPARE" if column["intent"] == "COMPARISON" else "GET",
        aggregation=column["aggregation"], variables=column["variables"],
        locations=column["locations"], times=column["times"],
        times_normalized=column["times_normalized"], confidence=1.0,
        activity=column["activity"], sub_activity=column.get("sub_activity", ""),
        entities=column.get("entities", {}), family=column["family"],
        reply=column.get("reply", ""), detail="NORMAL")


def entry(spec: Spec, present: bool) -> dict:
    """The catalogue row for an LLM, in the same shape `Registry.available()` emits so a
    client cannot tell a trained bundle from a prompted one."""
    return {"version": spec.version, "name": spec.name, "loaded": True, "present": present,
            "size_mb": None, "description": spec.description, "default": False}


def available() -> bool:
    return bool(AI_API_KEY)


def model_name() -> str:
    return AI_MODEL
