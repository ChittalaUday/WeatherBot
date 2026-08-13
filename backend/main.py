"""
WeatherBot chat backend: one WebSocket, the NLU model, and the WeatherSnap APIs.

    uvicorn backend.main:app --reload --port 8000

Every exchange streams over the socket so the UI can show progress rather than a spinner:

    client -> {"type": "query",    "text": "will it rain in Nokha tomorrow?"}
    client -> {"type": "location", "lat": 17.38, "lon": 78.48, "text": "<pending query>"}
    server -> {"type": "status",   "stage": "understanding" | "locating" | "fetching"}
    server -> {"type": "nlu",      "intent", "action", "entities", "confidence"}
    server -> {"type": "need_location"}          browser geolocation, then resend as "location"
    server -> {"type": "result",   "summary", "table", "places", "when", "series"}
    server -> {"type": "error",    "message"}

No login: the collection's /user/login is deliberately unused.
"""

from __future__ import annotations

import asyncio
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import insights, locations, planner, registry as models, respond, state as context, store, weather
from src.normalize import normalize
from src.schema import ConversationState, Operation, Verdict

app = FastAPI(title="WeatherBot", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

registry = models.Registry()         # Model 1, the only model
db = store.connect()                 # every turn is logged for the retraining loop
# Slot state is keyed by chat, not by socket: a reload reconnects with the same chat_id and
# the conversation continues where it left off.
CHATS: dict[str, ConversationState] = {}


@app.on_event("startup")
def load_model():
    registry.get()                            # warm Model 1 so the first turn is not slow


@app.get("/api/health")
def health():
    return {"status": "ok", "model": True, "models": registry.available()}


@app.get("/api/chats")
def chats(limit: int = 40):
    """Recent conversations, newest first, for the history panel."""
    return {"chats": store.list_chats(db, limit)}


@app.get("/api/chats/{chat_id}")
def chat_history(chat_id: str):
    """Every turn of one conversation, with the answers as they were rendered."""
    return {"chat_id": chat_id, "turns": store.conversation(db, chat_id)}


@app.get("/api/models")
def list_models():
    """The model this deployment answers with, and how it scored at export time."""
    import json as _json

    report = {}
    metrics_path = ROOT / "models/metrics_v3.json"
    if metrics_path.exists():
        report[models.NAME] = _json.loads(metrics_path.read_text())
    return {"available": registry.available(), "default": models.DEFAULT_VERSION,
            "metrics": report}


@app.get("/api/suggest")
async def suggest(q: str):
    """Location autocomplete for the frontend picker (TanStack Query polls this)."""
    async with weather.client() as http:
        return {"suggestions": await weather.suggest_locations(http, q)}


# Confidence routing, from `python src/nlu.py --calibrate` on the hand-written eval set:
#   >= 0.95   98.9% accurate over 83% of turns   -> answer
#   0.45-0.95 ~75% accurate                      -> answer, but mark it and queue for review
#   < 0.45    0-67% accurate                     -> ask instead of guessing
# Re-run the calibration after every retrain rather than trusting these numbers forever.
MIN_CONFIDENCE = 0.45
CONFIDENT = 0.95


async def handle_query(socket: WebSocket, text: str, coords: dict | None, session: str,
                       version: str | None = None, chat_id: str | None = None):
    """One turn through the pipeline:

        normalize -> model -> context -> location resolver -> time resolver -> validator
                  -> weather API -> aggregation -> insights -> template

    Everything except the model is deterministic, and every stage is logged.
    """
    started = time.perf_counter()
    await socket.send_json({"type": "status", "stage": "understanding"})

    cleaned = normalize(text)                      # shorthand and typos folded, audit kept
    understanding = registry.understand(cleaned.normalized, version)
    intent, action = understanding.intent, understanding.action
    # the model picks the reduction (Rule 2.3); the guard drops it when the prompt contains
    # no word that could have meant it - "weather in KKD" is not a request for a maximum
    aggregation = insights.confirm_aggregation(cleaned.normalized, understanding.aggregation)

    # cheap rules before anything else: a follow-up fragment leans on the previous turn
    reference = context.detect_reference(cleaned.normalized)
    follow_up = context.is_follow_up(cleaned.normalized)
    # Model 1 commits to a reading rather than stopping to ask (registry.NEVER_ASKS)
    asks = understanding.version not in models.NEVER_ASKS
    chat_id = chat_id or session
    state = CHATS.get(chat_id)
    if state is None:
        # the chat may predate this process; rebuild its slots from the stored turns so a
        # resumed conversation still knows where "there" is
        stored = store.last_state(db, chat_id)
        state = ConversationState(**stored) if stored else ConversationState()

    await socket.send_json({
        "type": "nlu",
        "intent": intent,
        "action": action,
        "aggregation": aggregation,
        "model": understanding.version,
        "variables": understanding.variables,
        "entities": {
            "location": understanding.locations,
            "time": understanding.times,
            "time_normalized": understanding.times_normalized,
        },
        "confidence": round(understanding.confidence, 4),
        "normalized": cleaned.normalized if cleaned.replacements else None,
        "replacements": cleaned.replacements,
        "reference": reference.value,
        "follow_up": follow_up,
    })

    # "angara vs hyderbad" names no metric at all - answering it with whatever the
    # classifier ranked first is how you get RAIN one turn and TEMPERATURE the next.
    confidence = understanding.confidence
    log = lambda outcome, detail, **extra: store.record_turn(
        # state.turns counts turns taken, so the index of *this* one is one less
        db, session, f"[{understanding.version}] {text}", chat_id=chat_id,
        turn=max(state.turns - 1, 0),
        intent=intent, action=action,
        confidence=confidence, location=understanding.locations, time_raw=understanding.times,
        time_norm=understanding.times_normalized,
        outcome=outcome, detail=detail, normalized=cleaned.normalized,
        scores=understanding.scores,
        latency_ms=int((time.perf_counter() - started) * 1000), **extra)

    # An inherited follow-up is allowed to be low confidence: "there?" carries no signal on
    # its own, and the state already holds what it means.
    if asks and confidence < MIN_CONFIDENCE and not (follow_up or reference != context.Reference.NONE):
        turn_id = log("clarified", "low confidence")
        await socket.send_json({
            "type": "clarify",
            "turn_id": turn_id,
            "message": "I am not sure which reading you want. Pick one:",
            "options": [
                {"intent": intent_name, "confidence": round(probability, 3),
                 "label": intent_name.replace("_", " ").lower()}
                for intent_name, probability in sorted(understanding.scores.items(),
                                                       key=lambda kv: -kv[1])[:3]
            ],
            "text": text,
        })
        return

    named = [name for name in understanding.locations
             if not locations.is_relative(name) and not locations.is_probably_not_a_place(name)]
    relative = [name for name in understanding.locations if locations.is_relative(name)]

    # fold this turn into the conversation: SET / REPLACE / MODIFY / INHERIT / COMPARE
    state, operation = context.apply(
        state,
        weather_intent=understanding.intent, action=understanding.action,
        aggregation=understanding.aggregation,
        location=named or relative, time_raw=understanding.times[0] if understanding.times else None,
        time_normalized=(understanding.times_normalized[0]
                         if understanding.times_normalized else None),
        reference=reference, follow_up=follow_up,
        confident=confidence >= MIN_CONFIDENCE,
        text=cleaned.normalized, variables=understanding.variables,
    )
    # the state is the source of truth from here on: it holds the inherited intent
    intent, action = state.weather_intent, state.action
    if coords:
        state.coords = coords
    # inherited turns reuse the places already resolved, so nothing is looked up twice
    if operation == Operation.INHERIT and state.resolved:
        named, relative = [], []

    # A comparison with one place is not a comparison; ask instead of quietly answering
    # about whichever place happened to be extracted.
    if asks and action == "COMPARE" and len(named) + len(relative) < 2 and not state.resolved:
        turn_id = log("clarified", "comparison with one place")
        await socket.send_json({
            "type": "clarify",
            "turn_id": turn_id,
            "message": (f"Compare {named[0]} with which other place?" if named
                        else "Which two places should I compare?"),
            "options": [],
            "text": text,
        })
        return

    # No usable place in the text and no coordinates yet -> ask the browser (Rule 4.1 keeps
    # "near me" as raw text; resolving it is this layer's job).
    if not named and not state.resolved and not (coords or state.coords):
        log("need_location", relative[0] if relative else "no place named")
        await socket.send_json({
            "type": "need_location",
            "reason": "relative" if relative else "missing",
            "text": text,
            "message": ("I need your location for "
                        f"\"{relative[0]}\"." if relative else
                        "Which place should I check? Share your location or name one."),
        })
        return

    async with weather.client() as http:
        await socket.send_json({"type": "status", "stage": "locating"})
        places, unknown = [], []
        if named:
            solr = lambda query, rows=8: weather.solr_query(http, query, rows)
            # a comma span can be one address or two places - the resolver decides
            named = [part for name in named for part in locations.split_span(name)]
            resolved = await asyncio.gather(*(locations.resolve(solr, n) for n in named))
            places = [p for p in resolved if p]
            unknown = [n for n, p in zip(named, resolved) if not p]
            if unknown and not places:
                log("error", f"unresolved: {', '.join(unknown)}", unresolved=unknown)
                await socket.send_json({
                    "type": "error",
                    "message": f"I could not find {', '.join(unknown)} in the location index.",
                })
                return
        if not places and state.resolved:
            places = state.resolved                     # inherited from the previous turn
        if not places and (coords or state.coords):
            point = coords or state.coords
            places = [await weather.reverse_geocode(http, point["lat"], point["lon"])]
        state.resolved = places

        # One name, several real places ("Angara" is in Jharkhand and in Andhra Pradesh):
        # the resolver hands back every match and the user picks, rather than us guessing.
        ambiguous = next((p for p in places if p.get("ambiguous") and len(named) == 1), None)
        if ambiguous and not asks:
            # commit to the ranked best and say which one, instead of interrupting
            understanding.assumed.append(
                f"{ambiguous['raw']} = {ambiguous['normalized']}, {ambiguous['state']}")
            ambiguous = None
        if ambiguous:
            turn_id = log("clarified", f"ambiguous location: {ambiguous['raw']}", places=places)
            await socket.send_json({
                "type": "clarify",
                "turn_id": turn_id,
                "message": f"There are several places called {ambiguous['normalized']}. Which one?",
                "options": [
                    {"intent": intent, "confidence": 1.0,
                     "label": ", ".join(part for part in (m["normalized"], m["district"], m["state"])
                                        if part),
                     "query": text.replace(ambiguous["raw"], f"{m['normalized']}, {m['state']}")}
                    for m in ambiguous["matches"][:4]
                ],
                "text": text,
            })
            return

        await socket.send_json({"type": "status", "stage": "fetching", "places": places})

        # time resolver + validator: absolute window, then answer-or-ask
        query = planner.plan(state, places=places, operation=operation, aggregation=aggregation)
        if query.verdict is Verdict.CLARIFY and not asks:
            # answer for what we have and say so, rather than asking
            understanding.assumed.append(
                "compared against the one place named" if "second_location" in query.missing
                else "used the place already in context")
            query.verdict = Verdict.READY
        if query.verdict is Verdict.CLARIFY:
            turn_id = log("clarified", f"missing {', '.join(query.missing)}", places=places)
            await socket.send_json({
                "type": "clarify", "turn_id": turn_id, "options": [], "text": text,
                "message": ("Which other place should I compare with?"
                            if "second_location" in query.missing
                            else "Which place should I check?"),
            })
            return

        normalized = state.time_normalized or ""
        hourly = query.granularity == "hourly"
        fetch = weather.hourly_forecast if hourly else weather.daily_forecast
        try:
            feeds = await asyncio.gather(*(fetch(http, p["lat"], p["lon"]) for p in places))
        except Exception as exc:                                   # noqa: BLE001 - report upstream failures verbatim
            log("error", f"WeatherSnap API failed: {exc}", places=places)
            await socket.send_json({"type": "error", "message": f"WeatherSnap API failed: {exc}"})
            return

    # Model 1's detail head decides how wide the table is. The variables come from the
    # state, so a follow-up keeps them.
    understanding.variables = state.variables or understanding.variables
    fields = understanding.fields()
    selected = [respond.select_rows(feed, normalized)[0] for feed in feeds]
    when = respond.select_rows(feeds[0], normalized)[1]

    compare = action == "COMPARE" and len(places) > 1
    table = respond.build_table(selected if compare else selected[0], fields, places, hourly)
    summary = respond.summarize(intent, action, selected if compare else selected[0],
                                fields, places, when)

    reduced = insights.apply_aggregation(selected[0], fields[0], aggregation)
    if reduced:                                     # lead with the number that was asked for
        summary = f"{reduced['text']}. {summary}"
    chart = insights.build_chart(selected, places, fields[0], hourly,
                                 kind=understanding.chart or None, fields=fields)
    notes = insights.build_insights(selected, places, fields, aggregation, hourly,
                                    wanted=understanding.insights or None)
    if unknown:
        summary += f" (Could not find {', '.join(unknown)} - showing the rest.)"
    for place in places:
        if place.get("fuzzy"):
            summary += (f" (No exact match for \"{place['raw']}\" - showing the closest, "
                        f"{place['normalized']}, {place['state']}.)")

    CHATS[chat_id] = state                          # only a turn that answered updates state
    uncertain = confidence < CONFIDENT
    # type / turn_id / chat_id / turn are added after logging, which is what mints turn_id
    answer = {
        "model": understanding.version,
        # what the answer was actually built from: on a follow-up these are inherited, and
        # reporting the raw prediction here would contradict the columns in the table
        "variables": state.variables or understanding.variables,
        "intent": intent,
        "action": action,
        "when": query.time_label or when,
        "operation": operation.value,
        "window": {"start": query.start, "end": query.end},
        "places": places,
        "granularity": "hourly" if hourly else "daily",
        "summary": summary,
        # answered from the middle band: shown, but flagged so a wrong one gets corrected
        "uncertain": uncertain,
        "confidence": round(confidence, 3),
        "aggregation": aggregation,
        "reduced": reduced,
        "chart": chart,
        "insights": notes,
        # never drop a place silently: a comparison missing one side is a wrong answer
        "unresolved": unknown,
        # what the model chose, and what it committed to instead of asking
        "presentation": ({"detail": understanding.detail, "chart": understanding.chart,
                          "insights": understanding.insights}
                         if understanding.detail else None),
        "assumed": understanding.assumed,
        "table": table,
        # tidy series for the chart: one line per place
        "series": [
            {"place": place["name"],
             "points": [{"t": row["Date_time"], "v": row.get(fields[0])} for row in rows]}
            for place, rows in zip(places, selected)
        ],
    }

    # Log first to mint the turn_id, complete the answer with it, then store the finished
    # payload: a replayed answer needs its own turn_id to be rateable.
    turn_id = log("uncertain" if uncertain else "answered", summary, places=places,
                  unresolved=unknown, operation=operation.value, state=state.model_dump())
    answer.update({"type": "result", "turn_id": turn_id, "chat_id": chat_id,
                   "turn": max(state.turns - 1, 0)})
    store.attach_payload(db, turn_id, answer)
    await socket.send_json(answer)


class Feedback(BaseModel):
    # typing.Optional, not `str | None`: pydantic resolves these at runtime and this venv is 3.9
    turn_id: int
    kind: str                       # up | down | correction | choice
    intent: Optional[str] = None
    action: Optional[str] = None
    variables: Optional[List[str]] = None
    location: Optional[List[str]] = None
    time: Optional[List[str]] = None
    model: Optional[str] = None
    error_type: Optional[str] = None
    note: Optional[str] = None


@app.post("/api/feedback")
def feedback(body: Feedback):
    """Thumbs, corrections, and the intent a user picked from a clarify prompt.

    A `choice` is the cheapest gold label there is: the model was unsure, a human answered.
    """
    store.record_feedback(db, body.turn_id, body.kind, intent=body.intent, action=body.action,
                          variables=body.variables, location=body.location,
                          time_raw=body.time, model=body.model, error_type=body.error_type,
                          note=body.note)
    current = store.feedback_for(db, body.turn_id)
    return {"ok": True, "labelled": len(store.training_rows(db)), "feedback": current}


@app.get("/api/feedback/{turn_id}")
def feedback_for_turn(turn_id: int):
    """What the user already said about this turn, so a reopened chat shows its ratings."""
    return {"feedback": store.feedback_for(db, turn_id)}


@app.get("/api/labels")
def labels():
    """The label sets a correction form has to offer, straight from Model 1's enums."""
    from src.v2.schema import Intent, Variable
    from src.v3.schema import ChartKind, Detail, Insight

    return {
        "intents": [i.value for i in Intent],
        "variables": [v.value for v in Variable],
        "detail": [d.value for d in Detail],
        "chart": [c.value for c in ChartKind],
        "insights": [i.value for i in Insight],
    }


@app.get("/api/review")
def review(limit: int = 50):
    """Turns waiting for a human label - flagged wrong, or answered uncertainly and ignored."""
    return {"queue": store.review_queue(db, limit)}


@app.get("/api/stats")
def usage_stats():
    return store.stats(db)


@app.websocket("/ws")
async def chat(socket: WebSocket):
    await socket.accept()
    session = uuid.uuid4().hex[:12]
    await socket.send_json({"type": "ready", "session": session,
                            "message": "Ask me about the weather."})
    chat_id: str | None = None
    try:
        while True:
            message = await socket.receive_json()
            kind = message.get("type")
            try:
                # the client owns the chat id, so a page reload resumes the same conversation
                chat_id = message.get("chat_id") or chat_id or f"chat-{uuid.uuid4().hex[:10]}"
                if kind == "query":
                    await handle_query(socket, message["text"], message.get("coords"), session,
                                       message.get("model"), chat_id)
                elif kind == "location":
                    # browser answered the geolocation prompt: rerun the pending query
                    await handle_query(socket, message["text"],
                                       {"lat": message["lat"], "lon": message["lon"]}, session,
                                       message.get("model"), chat_id)
                elif kind == "reset":
                    CHATS.pop(chat_id, None)          # "new chat" - forget the slots
                    chat_id = message.get("chat_id") or f"chat-{uuid.uuid4().hex[:10]}"
                    await socket.send_json({"type": "ready", "session": session,
                                            "chat_id": chat_id, "message": "New chat."})
                elif kind == "ping":
                    await socket.send_json({"type": "pong"})
            except Exception as exc:                               # noqa: BLE001 - one bad turn must not kill the socket
                await socket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        return
