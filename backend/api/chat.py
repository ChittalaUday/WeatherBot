"""
The chat endpoint: one POST, the turn streamed back as server-sent events.

    python tests/test_live_stack.py          # the checks for this module

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

SSE over POST, not a WebSocket: one question has one answer, so there is nothing to hold open
between turns, and it survives any proxy that speaks HTTP. `turn()` is an async generator, so
the transport is somebody else's problem.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from backend import generation, store
from backend.api.deps import CHATS, conversation_state, db, registry
from backend.api.schemas import (
    AskRequest,
    CompareRequest,
    ResetChatRequest,
    ResetChatResponse,
)
from backend.config import CONFIDENT, MIN_CONFIDENCE
from backend.nlu import context, duckling, normalize_text
from backend.nlu import understand as read
from backend.pipeline import analysis, render, resolve_places, run, sources
from backend.pipeline import places as place_index
from src.schema import Operation

router = APIRouter()

# Three is two follow-ups deep, which is where "and there?" -> "what about next week?" stops
# being resolvable from the last turn alone. More is not free: a small model has a fixed
# budget of attention and spends it on whatever is in front of it.
HISTORY_TURNS = 3

# How long a turn will wait for the model's pick of table-or-chart once the phrasing is done.
# It runs beside the phrasing, so this is only ever reached when the phrasing did not happen.
VIEW_TIMEOUT = 1.0


async def turn(text: str, *, chat_id: str, model: str | None = None,
               coords: dict | None = None):
    """One turn, as a stream of events.

    Everything conversation-shaped lives here - the remembered place, the operation, the
    browser's coordinates, the turn log. Everything answer-shaped lives in `backend.pipeline`,
    which this calls exactly once.
    """
    started = time.perf_counter()
    # One wall-clock reading for the whole turn, so the time Duckling resolves against and the
    # time the pipeline resolves against cannot straddle a midnight.
    started_at = datetime.now()
    ms = lambda since: int((time.perf_counter() - since) * 1000)
    timing = {"nlu_ms": 0, "solr_ms": 0, "api_ms": 0, "llm_ms": 0, "db_ms": 0}

    yield {"type": "status", "stage": "understanding"}
    cleaned = normalize_text(text)              # shorthand and typos folded, audit kept
    began = time.perf_counter()
    understanding, hint = await asyncio.gather(
        read(registry, cleaned.normalized, model),
        duckling.canonical(cleaned.normalized, started_at),
    )
    understanding.time_hint = hint
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
    # reaches location resolution comes back asking which city you meant.
    if not understanding.needs_weather:
        began_llm = time.perf_counter()
        history = store.recent_exchanges(db, chat_id, HISTORY_TURNS)
        llm_reply = await generation.say_conversational(
            text, intent=understanding.intent, family=understanding.family, history=history, fallback=understanding.reply
        )
        timing["llm_ms"] = ms(began_llm)
        message = llm_reply or understanding.reply
        turn_id = log(understanding.family, message)
        yield {"type": "chat", "turn_id": turn_id, "chat_id": chat_id,
               "model": understanding.version, "intent": understanding.intent,
               "family": understanding.family, "message": message,
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

    named = [n for n in understanding.locations if not place_index.is_relative(n)]
    over_the_cap: list[str] = []
    relative = ([n for n in understanding.locations if place_index.is_relative(n)]
                or place_index.relative_in(cleaned.normalized))

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
            places, unresolved, over_the_cap = await resolve_places(http, named)
            timing["solr_ms"] += ms(began)

            # Retrieve before generating: the nearest names the index *does* hold are what
            # turn "I could not find veedurumudi" into "did you mean Vedurumudi?".
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
            if point is not None:
                places = [await sources.reverse_geocode(http, point["lat"], point["lon"])]
            timing["solr_ms"] += ms(began)

        # The tagger found nothing, and the index might. It is a model over 623,000 names, so
        # it will always have gaps - "Guntur weather" is two tokens with almost no context and
        # came back with no location at all. One lookup per candidate word before giving up,
        # and only on the path that was about to be a dead end anyway.
        if not places and not relative and not (coords or state.coords):
            began = time.perf_counter()
            solr = lambda q, rows=8: sources.solr_query(http, q, rows)
            places = await place_index.find_in(solr, cleaned.normalized)
            timing["solr_ms"] += ms(began)
            if places:
                understanding.locations = [p["raw"] for p in places]

        # 4. No usable place and no coordinates yet -> ask the browser. Rule 4.1 keeps "near
        # me" as raw text; resolving it is this layer's job.
        if not places:
            log("need_location", relative[0] if relative else "no place named")
            yield {"type": "need_location",
                   "reason": "relative" if relative else "missing", "text": text,
                   "message": (f"I need your location for \"{relative[0]}\"." if relative else
                               "Which place should I check? Share your location or name one.")}
            return

        # One name, several real places ("Angara" is in Jharkhand and in Andhra Pradesh).
        # Committing to the ranked best and admitting it in `assumed` was Rule 1.1 applied
        # where it does not belong: a reading the model picked can be corrected afterwards,
        # but a forecast for the wrong district is acted on before anyone reads the footnote.
        # Only a genuine tie asks - `places._is_ambiguous` already refuses to raise one
        # between a district seat and a same-named hamlet.
        undecided = next((p for p in places if p.get("ambiguous")), None)
        if undecided is not None and len(named) == 1:
            options = [{"name": m.get("normalized") or m.get("name", ""),
                        "district": m.get("district", ""), "state": m.get("state", ""),
                        "type": m.get("type", "")}
                       for m in (undecided.get("matches") or [])][:5]
            if len(options) > 1:
                log("need_location", f"ambiguous: {undecided['raw']}", places=places)
                yield {"type": "confirm_location", "text": text,
                       "raw": undecided["raw"], "options": options,
                       "message": f"There is more than one {undecided['raw']}. Which one?"}
                return
        state.resolved = places

        yield {"type": "status", "stage": "fetching", "places": places}
        # The merged state is the source of truth from here on. Skipping this is how "and
        # there?" fetched a seven-day horizon under a label still reading "tomorrow".
        understanding.variables = state.variables or understanding.variables
        if state.time_normalized:
            understanding.times_normalized = [state.time_normalized]
            understanding.times = [state.time_raw] if state.time_raw else understanding.times
        # the same `now` Duckling resolved against, so the hint and the window agree
        answer = await run(http, understanding, places=places, now=started_at)

    timing["api_ms"] = (answer.stages.get("fetch") or {}).get("ms", 0)
    unresolved = unresolved or answer.unresolved

    if answer.stopped_by and answer.plan is not None:
        reason = answer.plan.reason or ""
        log("clarified", reason, places=places)
        message = reason.capitalize() + "."
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

    # 5. What this turn knows and the pipeline did not goes on the sentence *before* it is
    # phrased. Appended after, it reads as a footnote contradicting the paragraph above it.
    for place in places:
        if place.get("fuzzy"):
            answer.summary += (f" (No exact match for \"{place['raw']}\" - showing the "
                               f"closest, {place['normalized']}, {place['state']}.)")
    answer.over_the_cap = over_the_cap
    if unresolved and (coords or state.coords):
        # First, not last: this changes what the whole answer is *about*, and a model told to
        # lead with the conclusion leads with its opening words.
        answer.summary = (f"I do not have {', '.join(unresolved)} in the location index, so "
                          f"these readings are for {places[0]['name']}, the place those "
                          f"coordinates fall in. ") + answer.summary

    # 6. The phrasing, streamed - the slowest step by a wide margin. The `result` that follows
    # carries the finished text anyway, so a client that missed a piece is still correct.
    yield {"type": "status", "stage": "writing"}
    began = time.perf_counter()
    # Which of the table and the chart to open. Started here and collected after the stream:
    # phrasing is seconds and this is a couple of hundred milliseconds, so it costs nothing as
    # long as it runs beside it. `answer.presentation` already holds the rule's answer.
    view = asyncio.create_task(render.choose(answer, text))
    # not `context` - that name is the NLU module this endpoint already imports
    sections = generation.build(answer.as_context())
    retrieved = sections.render()
    # What came before, so a follow-up is answered as one. Without it "and tomorrow?" read as
    # a complete question about nothing.
    history = store.recent_exchanges(db, chat_id, HISTORY_TURNS)
    # An advice turn is the one place the wording must not think for itself. Everywhere else
    # it may say what the figures mean - the difference between an answer and a label printer.
    deciding = answer.advice is not None
    # Read off what the analysis found, not off the question: six findings want a laid-out
    # answer whatever was asked, and one finding wants a sentence.
    structured = analysis.wants_structure(
        answer.reductions, answer.insights, answer.places,
        # minus the time column, which is not a measurement anyone came for
        max(len(answer.table.get("columns") or []) - 1, 0))
    said = ""
    async for kind, piece in generation.stream(answer.summary, text, context=retrieved,
                                               deciding=deciding, history=history,
                                               headings=sections.headings(),
                                               structured=structured):
        if kind == "answer":
            said += piece
        yield {"type": "thinking" if kind == "thinking" else "delta", "text": piece}
    timing["llm_ms"] = ms(began)
    # Checked after the stream, not during: a reply is only wrong once it is whole, and the
    # client renders `result.summary` rather than what it watched being typed, so a late
    # rejection is invisible instead of a sentence rewriting itself on screen.
    kept = generation.usable(said, answer.summary, answer.summary + " " + retrieved,
                             answer.advice.verdict if answer.advice else "")
    if said and not kept:
        print(f"generation: dropped a reply for {text!r}: {said[:140]!r}", flush=True)
    # `kept`, not `said`. This line took the reply whether or not it passed, so every guard in
    # `usable` - invented figures, reversed verdicts, scaffolding - was computed, logged as
    # dropped, and then shown anyway. A rejected reply falls back to the deterministic
    # sentence, which is what "dropped" was always supposed to mean.
    answer.summary = kept or answer.summary
    # Read off the reply, not off the request: the layout was asked for, and a model that
    # found one sentence to say correctly said one. The deterministic fallback is never
    # markdown either, so a client is never told to parse a stray "-" as a bullet.
    answer.summary_format = "markdown" if generation.is_markdown(kept) else "text"

    # Normally already done. The cap is for the turn where phrasing failed instantly; past it
    # the rule's answer, already in place, stands.
    try:
        answer.presentation = await asyncio.wait_for(view, VIEW_TIMEOUT)
    except asyncio.TimeoutError:
        pass

    CHATS[chat_id] = state                         # only a turn that answered updates state
    uncertain = understanding.confidence < CONFIDENT
    result = {
        **answer.payload(understanding,
                         variables=state.variables or understanding.variables),
        "operation": operation.value,
        "window": {"start": answer.plan.start if answer.plan else "", "end": answer.plan.end if answer.plan else ""},
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

    `str(exc)` once put "name 're' is not defined" in a chat bubble, so it goes to the log and
    the reader gets something human.
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
async def chat(body: AskRequest):
    """One turn, streamed. The client owns the chat id, so a reload resumes the conversation."""
    chat_id = body.chat_id or store.new_chat_id()
    coords = {"lat": body.lat, "lon": body.lon} if body.lat is not None else None
    return _sse(_guarded(turn(body.text, chat_id=chat_id, model=body.model, coords=coords),
                         body.text))


@router.post("/api/chat/reset", response_model=ResetChatResponse)
def reset(body: ResetChatRequest):
    """"New chat" - forget this chat's slots and hand back a fresh id."""
    if body.chat_id:
        CHATS.pop(body.chat_id, None)
    return {"chat_id": store.new_chat_id(), "message": "New chat."}


@router.post("/api/compare")
async def compare(body: CompareRequest):
    """The same sentence through every model, streamed as each one finishes."""
    from backend.api.compare import columns

    return _sse(_guarded(columns(body.text), body.text))
