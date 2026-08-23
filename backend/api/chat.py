"""
The chat endpoint: one POST, the turn streamed back as server-sent events.

    python -m backend.api.chat        # self-check: one turn, end to end, no server

    POST /api/chat    {"text": "will it rain in Guntur tomorrow", "chat_id": "chat-1"}
                      {"text": "<the pending question>", "lat": 17.3, "lon": 78.4}

    data: {"type":"status",  "stage":"understanding"|"locating"|"fetching"|"writing"}
    data: {"type":"nlu",     ...}                     what the model read, for the debug strip
    data: {"type":"thinking","text":"..."}            the local model reasoning, as it goes
    data: {"type":"delta",   "text":"..."}            the answer being written, piece by piece
    then exactly one of:
    data: {"type":"result",  ...}                     the forecast
    data: {"type":"chat",    ...}                     a greeting, a control turn, a refusal
    data: {"type":"clarify", ...}                     the plan cannot serve that question
    data: {"type":"need_location", ...}               ask the browser, then resend with lat/lon
    data: {"type":"error",   "message":"..."}

SSE over POST, not a WebSocket. One question has one answer, so there is nothing to hold open
between turns: no reconnect loop, no ping, no half-open socket that silently stops answering,
and it survives any proxy that speaks HTTP. The streaming that mattered - the phrasing, which
is the slowest step by a wide margin - is exactly what SSE is for.

`turn()` is an async generator, so the transport is somebody else's problem: the endpoint below
streams it and the self-check just collects it into a list.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend import generation, store
from backend.api.deps import CHATS, conversation_state, db, registry
from backend.config import CONFIDENT, MIN_CONFIDENCE
from backend.nlu import context, normalize_text
from backend.pipeline import places as place_index
from backend.pipeline import resolve_places, run, sources
from src.schema import Operation

router = APIRouter()


class Ask(BaseModel):
    # typing.Optional, not `str | None`: pydantic resolves these at runtime and this venv is 3.9
    text: str
    chat_id: Optional[str] = None
    model: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class Compare(BaseModel):
    text: str
    chat_id: Optional[str] = None


async def turn(text: str, *, chat_id: str, model: str | None = None,
               coords: dict | None = None):
    """One turn, as a stream of events.

    Everything conversation-shaped lives here - the remembered place, the operation, the
    browser's coordinates, the turn log. Everything answer-shaped lives in `backend.pipeline`,
    which this calls exactly once.
    """
    started = time.perf_counter()
    ms = lambda since: int((time.perf_counter() - since) * 1000)
    timing = {"nlu_ms": 0, "solr_ms": 0, "api_ms": 0, "llm_ms": 0, "db_ms": 0}

    yield {"type": "status", "stage": "understanding"}
    cleaned = normalize_text(text)              # shorthand and typos folded, audit kept
    began = time.perf_counter()
    understanding = registry.understand(cleaned.normalized, model)
    timing["nlu_ms"] = ms(began)

    began = time.perf_counter()
    state = conversation_state(chat_id)
    timing["db_ms"] += ms(began)

    def log(outcome: str, detail: str, **extra) -> int:
        began = time.perf_counter()
        turn_id = store.record_turn(
            db, chat_id, f"[{understanding.version}] {text}", chat_id=chat_id,
            # state.turns counts turns taken, so the index of *this* one is one less
            turn=max(state.turns - 1, 0),
            intent=understanding.intent, action=understanding.action,
            confidence=understanding.confidence, location=understanding.locations,
            time_raw=understanding.times, time_norm=understanding.times_normalized,
            outcome=outcome, detail=detail, normalized=cleaned.normalized,
            scores=understanding.scores, latency_ms=ms(started), **extra)
        timing["db_ms"] += ms(began)
        return turn_id

    # 1. A turn that needs no weather is answered here and goes no further. A "hi" that
    # reaches location resolution comes back asking which city you meant, which is the single
    # worst thing this bot can do.
    if not understanding.needs_weather:
        turn_id = log(understanding.family, understanding.reply)
        yield {"type": "chat", "turn_id": turn_id, "chat_id": chat_id,
               "model": understanding.version, "intent": understanding.intent,
               "family": understanding.family, "message": understanding.reply,
               "confidence": round(understanding.confidence, 3),
               # CHANGE_LOCATION carries the place it wants switched to
               "locations": understanding.locations,
               "metrics": {**timing, "total_ms": ms(started)}}
        return

    # 2. cheap rules before anything else: a follow-up fragment leans on the previous turn
    reference = context.detect_reference(cleaned.normalized)
    follow_up = context.is_follow_up(cleaned.normalized)
    yield {
        "type": "nlu", "model": understanding.version, "intent": understanding.intent,
        "action": understanding.action, "aggregation": understanding.aggregation,
        "variables": understanding.variables,
        "entities": {"location": understanding.locations, "time": understanding.times,
                     "time_normalized": understanding.times_normalized},
        "confidence": round(understanding.confidence, 4),
        "normalized": cleaned.normalized if cleaned.replacements else None,
        "replacements": cleaned.replacements,
        "reference": reference.value, "follow_up": follow_up,
    }

    named = [n for n in understanding.locations
             if not place_index.is_relative(n) and not place_index.is_probably_not_a_place(n)]
    relative = [n for n in understanding.locations if place_index.is_relative(n)]

    # 3. fold this turn into the conversation: SET / REPLACE / MODIFY / INHERIT / COMPARE
    state, operation = context.apply(
        state, weather_intent=understanding.intent, action=understanding.action,
        aggregation=understanding.aggregation, location=named or relative,
        time_raw=understanding.times[0] if understanding.times else None,
        time_normalized=(understanding.times_normalized[0]
                         if understanding.times_normalized else None),
        reference=reference, follow_up=follow_up,
        # a fragment the model is unsure of keeps the previous question rather than adopting
        # a guess: "there?" carries no signal on its own and the state already holds it
        confident=understanding.confidence >= MIN_CONFIDENCE,
        text=cleaned.normalized, variables=understanding.variables)
    if coords:
        state.coords = coords
    # inherited turns reuse the places already resolved, so nothing is looked up twice
    inherited = operation == Operation.INHERIT and bool(state.resolved)

    async with sources.client() as http:
        yield {"type": "status", "stage": "locating"}
        places, unresolved = [], []
        if named and not inherited:
            began = time.perf_counter()
            places, unresolved = await resolve_places(http, named)
            timing["solr_ms"] += ms(began)

            # A name the index does not hold is a dead end as an error: it says what failed
            # and nothing about what would work. Retrieve before generating - the nearest
            # names the index *does* hold are what turn "I could not find veedurumudi" into
            # "did you mean Vedurumudi, in East Godavari?".
            if unresolved and not places and not (coords or state.coords):
                solr = lambda q, rows=8: sources.solr_query(http, q, rows)
                near = await place_index.suggest(solr, unresolved[0])
                log("need_location", f"unresolved: {', '.join(unresolved)}",
                    unresolved=unresolved)
                yield {"type": "need_location", "reason": "unresolved", "text": text,
                       "message": await generation.explain("location", near=near)}
                return

        if not places and state.resolved:
            places = state.resolved                       # inherited from the previous turn
        if not places and (coords or state.coords):
            began = time.perf_counter()
            point = coords or state.coords
            places = [await sources.reverse_geocode(http, point["lat"], point["lon"])]
            timing["solr_ms"] += ms(began)

        # 4. No usable place and no coordinates yet -> ask the browser. Rule 4.1 keeps "near
        # me" as raw text; resolving it is this layer's job.
        if not places:
            log("need_location", relative[0] if relative else "no place named")
            yield {"type": "need_location",
                   "reason": "relative" if relative else "missing", "text": text,
                   "message": (f"I need your location for \"{relative[0]}\"." if relative else
                               "Which place should I check? Share your location or name one.")}
            return

        # One name, several real places ("Angara" is in Jharkhand and in Andhra Pradesh). Both
        # models commit to a reading rather than interrupting (Rule 1.1), so the ranked best
        # wins and the answer says which one it took.
        for place in places:
            if place.get("ambiguous") and len(named) == 1:
                understanding.assumed.append(
                    f"{place['raw']} = {place['normalized']}, {place['state']}")
        state.resolved = places

        yield {"type": "status", "stage": "fetching", "places": places}
        # The merged state is the source of truth from here on, so the pipeline is handed what
        # the conversation means rather than what this sentence said on its own. Skipping this
        # is how "rain in Guntur tomorrow" -> "and there?" fetched the default seven-day
        # horizon while the label above the table still read "tomorrow": the window came from
        # the state and the rows came from the bare fragment.
        understanding.variables = state.variables or understanding.variables
        if state.time_normalized:
            understanding.times_normalized = [state.time_normalized]
            understanding.times = [state.time_raw] if state.time_raw else understanding.times
        answer = await run(http, understanding, places=places)

    timing["api_ms"] = (answer.stages.get("fetch") or {}).get("ms", 0)
    unresolved = unresolved or answer.unresolved

    if answer.stopped_by:
        log("clarified", answer.plan.reason, places=places)
        message = answer.plan.reason.capitalize() + "."
        if answer.plan.offer:
            message += (f" I can give you {answer.plan.offer[0].lower()} figures for that "
                        f"period instead.")
        yield {"type": "clarify", "text": text, "message": message}
        return
    if not answer.ok:
        # the real reason goes to the log, where it can be diagnosed; the reader gets the
        # version they can act on
        log("error", answer.error, places=places)
        yield {"type": "error", "message": await generation.explain(answer.failed_at)}
        return

    # 5. Everything this turn knows that the pipeline did not goes on the sentence *before* it
    # is phrased. Appended afterwards these read as a footnote contradicting the paragraph
    # above them - and the phrasing model, asked about one place and handed figures for
    # another with nothing joining the two, answered "not covered in the provided data" under
    # a complete forecast.
    for place in places:
        if place.get("fuzzy"):
            answer.summary += (f" (No exact match for \"{place['raw']}\" - showing the "
                               f"closest, {place['normalized']}, {place['state']}.)")
    if unresolved and (coords or state.coords):
        # First, not last: this changes what the whole answer is *about*, and a model told to
        # lead with the conclusion leads with its opening words.
        answer.summary = (f"I do not have {', '.join(unresolved)} in the location index, so "
                          f"these readings are for {places[0]['name']}, the place those "
                          f"coordinates fall in. ") + answer.summary

    # 6. The phrasing, streamed. It is the slowest step by a wide margin (a local model), and
    # a chat that sits on "fetching" for all of it looks hung. The `result` that follows
    # carries the finished text anyway, so a client that missed a piece is still correct.
    yield {"type": "status", "stage": "writing"}
    began = time.perf_counter()
    retrieved = generation.build(answer.as_context()).render()
    said = ""
    async for kind, piece in generation.stream(answer.summary, text, context=retrieved):
        if kind == "answer":
            said += piece
        yield {"type": "thinking" if kind == "thinking" else "delta", "text": piece}
    timing["llm_ms"] = ms(began)
    # The wording is kept only if it invented no figure, reversed no verdict, said something
    # the conclusion did not already say, and did not narrate the machinery. Checked after the
    # stream rather than during it, because a reply is only wrong once it is whole - and the
    # client renders `result.summary`, not what it watched being typed, so a late rejection is
    # invisible rather than a sentence rewriting itself on screen.
    kept = generation.usable(said, answer.summary, answer.summary + " " + retrieved,
                             answer.advice.verdict if answer.advice else "")
    if said and not kept:
        print(f"generation: dropped a reply for {text!r}: {said[:140]!r}", flush=True)
    answer.summary = kept or answer.summary        # empty or rejected -> the rule-built line

    CHATS[chat_id] = state                         # only a turn that answered updates state
    uncertain = understanding.confidence < CONFIDENT
    result = {
        **answer.payload(understanding,
                         variables=state.variables or understanding.variables),
        "operation": operation.value,
        "window": {"start": answer.plan.start, "end": answer.plan.end},
        # answered from the middle confidence band: shown, but flagged so a wrong one gets
        # corrected rather than quietly believed
        "uncertain": uncertain,
        "unresolved": unresolved,
        "metrics": {**timing, "total_ms": ms(started)},
    }
    # Log first to mint the turn_id, complete the answer with it, then store the finished
    # payload: a replayed answer needs its own turn_id to be rateable.
    turn_id = log("uncertain" if uncertain else "answered", answer.summary, places=places,
                  unresolved=unresolved, operation=operation.value, state=state.model_dump())
    result.update({"type": "result", "turn_id": turn_id, "chat_id": chat_id,
                   "turn": max(state.turns - 1, 0)})
    store.attach_payload(db, turn_id, result)
    yield result


def _sse(events):
    """One `data:` line per event, in the order the generator yields them."""
    async def body():
        async for event in events:
            yield f"data: {json.dumps(event, default=str)}\n\n"
    return StreamingResponse(body(), media_type="text/event-stream",
                             # nginx buffers event-streams into uselessness without this
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


async def _guarded(events, text: str):
    """Never let one bad turn end as a dead stream: every path yields a terminal event.

    `str(exc)` is a stack trace's last line - it once put "name 're' is not defined" in a chat
    bubble. It goes to the log for whoever is on call; the reader gets something human.
    """
    try:
        async for event in events:
            yield event
    except Exception as exc:                                        # noqa: BLE001
        print(f"turn failed: {type(exc).__name__}: {exc}", flush=True)
        try:
            said = await generation.explain("unknown", text)
        except Exception:                                           # noqa: BLE001
            said = generation.TROUBLE_LINES["unknown"]   # never fail while failing
        yield {"type": "error", "message": said}


@router.post("/api/chat")
async def chat(body: Ask):
    """One turn, streamed. The client owns the chat id, so a reload resumes the conversation."""
    chat_id = body.chat_id or store.new_chat_id()
    coords = {"lat": body.lat, "lon": body.lon} if body.lat is not None else None
    return _sse(_guarded(turn(body.text, chat_id=chat_id, model=body.model, coords=coords),
                         body.text))


@router.post("/api/chat/reset")
def reset(body: Compare):
    """"New chat" - forget this chat's slots and hand back a fresh id."""
    if body.chat_id:
        CHATS.pop(body.chat_id, None)
    return {"chat_id": store.new_chat_id(), "message": "New chat."}


@router.post("/api/compare")
async def compare(body: Compare):
    """The same sentence through every model, streamed as each one finishes."""
    from backend.api.compare import columns

    return _sse(_guarded(columns(body.text), body.text))


async def demo():
    """Self-check: one turn's events, in order, with every stage timed."""
    sent = [event async for event in turn("will it rain in Guntur tomorrow",
                                          chat_id="demo-chat", model="v4")]
    kinds = [e["type"] for e in sent]
    streamed = {"delta", "thinking"}
    print("  " + " -> ".join(f"{e['type']}:{e.get('stage', '')}".rstrip(":") for e in sent
                             if e["type"] not in streamed)
          + f"  ({kinds.count('thinking')} thinking + {kinds.count('delta')} answer pieces)")

    assert kinds[-1] in {"result", "chat", "clarify", "need_location", "error"}, kinds[-1]
    result = next((e for e in sent if e["type"] == "result"), None)
    if not result:
        print(f"  no answer ({sent[-1].get('message', kinds[-1])}) - network or model down")
        return
    assert [e["stage"] for e in sent if e["type"] == "status"] == \
        ["understanding", "locating", "fetching", "writing"], kinds
    if "delta" in kinds:
        # words must arrive while it is writing, and add up to the answer that follows
        assert kinds.index("status") < kinds.index("delta") < kinds.index("result"), kinds
        said = "".join(e["text"] for e in sent if e["type"] == "delta")
        assert result["summary"].startswith(said), (said, result["summary"])
        if "thinking" in kinds:
            # the reasoning is its own channel, and it comes before the answer it reasons about
            assert kinds.index("thinking") < kinds.index("delta"), kinds
            thought = "".join(e["text"] for e in sent if e["type"] == "thinking")
            assert thought not in said, "reasoning leaked into the answer"
            print(f"  thought: {thought[:70]}...")
    else:
        print("  no deltas - the local model is offline, answered with the rule-built sentence")

    assert set(result["metrics"]) == {"nlu_ms", "solr_ms", "api_ms", "llm_ms", "db_ms",
                                      "total_ms"}, result["metrics"]
    assert result["metrics"]["total_ms"] >= result["metrics"]["llm_ms"]
    print("  " + "  ".join(f"{k}={v}" for k, v in result["metrics"].items()))

    # a greeting never reaches the location resolver
    greeting = [e async for e in turn("hey there", chat_id="demo-chat", model="v4")]
    assert [e["type"] for e in greeting] == ["status", "chat"], greeting
    print(f"  greeting -> {greeting[-1]['message']!r}")
    print("chat stream check OK")


if __name__ == "__main__":
    asyncio.run(demo())
