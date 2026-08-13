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

registry = models.Registry()         # v1 and v2, chosen per request
db = store.connect()                 # every turn is logged for the retraining loop
SESSIONS: dict[str, ConversationState] = {}   # slot state per socket, deterministic


@app.on_event("startup")
def load_model():
    registry.get(models.DEFAULT_VERSION)      # warm the default; v2 loads on first use


@app.get("/api/health")
def health():
    return {"status": "ok", "model": True, "models": registry.available()}


@app.get("/api/models")
def list_models():
    """Which NLU versions this deployment can answer with, and how they differ."""
    import json as _json

    report = {}
    for name, path in (("v1", ROOT / "models/metrics.json"), ("v2", ROOT / "models/metrics_v2.json")):
        if path.exists():
            report[name] = _json.loads(path.read_text())
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
                       version: str | None = None):
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
    state = SESSIONS.get(session, ConversationState())

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
        db, session, f"[{understanding.version}] {text}", intent=intent, action=action,
        confidence=confidence, location=understanding.locations, time_raw=understanding.times,
        time_norm=understanding.times_normalized,
        outcome=outcome, detail=detail, normalized=cleaned.normalized,
        scores=understanding.scores,
        latency_ms=int((time.perf_counter() - started) * 1000), **extra)

    # An inherited follow-up is allowed to be low confidence: "there?" carries no signal on
    # its own, and the state already holds what it means.
    if confidence < MIN_CONFIDENCE and not (follow_up or reference != context.Reference.NONE):
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
    if action == "COMPARE" and len(named) + len(relative) < 2 and not state.resolved:
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

    # v2 can ask for several variables at once; the table gets a column group per variable
    # the state owns the variables: a follow-up that named none keeps the previous ones
    keys = [models.VARIABLE_TO_FIELDS_KEY.get(v, v) for v in state.variables] or [intent]
    fields, seen = [], set()
    for key in keys:
        for name in respond.INTENT_FIELDS.get(key, []):
            if name not in seen:
                seen.add(name)
                fields.append(name)
    fields = fields[:6] or respond.INTENT_FIELDS.get(intent, ["Tavg"])
    selected = [respond.select_rows(feed, normalized)[0] for feed in feeds]
    when = respond.select_rows(feeds[0], normalized)[1]

    compare = action == "COMPARE" and len(places) > 1
    table = respond.build_table(selected if compare else selected[0], fields, places, hourly)
    summary = respond.summarize(intent, action, selected if compare else selected[0],
                                fields, places, when)

    reduced = insights.apply_aggregation(selected[0], fields[0], aggregation)
    if reduced:                                     # lead with the number that was asked for
        summary = f"{reduced['text']}. {summary}"
    chart = insights.build_chart(selected, places, fields[0], hourly)
    notes = insights.build_insights(selected, places, fields, aggregation, hourly)
    if unknown:
        summary += f" (Could not find {', '.join(unknown)} - showing the rest.)"
    for place in places:
        if place.get("fuzzy"):
            summary += (f" (No exact match for \"{place['raw']}\" - showing the closest, "
                        f"{place['normalized']}, {place['state']}.)")

    SESSIONS[session] = state                       # only a turn that answered updates state
    uncertain = confidence < CONFIDENT
    turn_id = log("uncertain" if uncertain else "answered", summary, places=places,
                  unresolved=unknown, operation=operation.value)
    await socket.send_json({
        "type": "result",
        "turn_id": turn_id,
        "model": understanding.version,
        "variables": understanding.variables,
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
        "table": table,
        # tidy series for the chart: one line per place
        "series": [
            {"place": place["name"],
             "points": [{"t": row["Date_time"], "v": row.get(fields[0])} for row in rows]}
            for place, rows in zip(places, selected)
        ],
    })


class Feedback(BaseModel):
    # typing.Optional, not `str | None`: pydantic resolves these at runtime and this venv is 3.9
    turn_id: int
    kind: str                       # up | down | correction | choice
    intent: Optional[str] = None
    action: Optional[str] = None
    location: Optional[List[str]] = None
    time: Optional[List[str]] = None
    note: Optional[str] = None


@app.post("/api/feedback")
def feedback(body: Feedback):
    """Thumbs, corrections, and the intent a user picked from a clarify prompt.

    A `choice` is the cheapest gold label there is: the model was unsure, a human answered.
    """
    store.record_feedback(db, body.turn_id, body.kind, intent=body.intent, action=body.action,
                          location=body.location, time_raw=body.time, note=body.note)
    return {"ok": True, "labelled": len(store.training_rows(db))}


@app.get("/api/stats")
def usage_stats():
    return store.stats(db)


@app.websocket("/ws")
async def chat(socket: WebSocket):
    await socket.accept()
    session = uuid.uuid4().hex[:12]
    await socket.send_json({"type": "ready", "session": session,
                            "message": "Ask me about the weather."})
    try:
        while True:
            message = await socket.receive_json()
            kind = message.get("type")
            try:
                if kind == "query":
                    await handle_query(socket, message["text"], message.get("coords"), session,
                                       message.get("model"))
                elif kind == "location":
                    # browser answered the geolocation prompt: rerun the pending query
                    await handle_query(socket, message["text"],
                                       {"lat": message["lat"], "lon": message["lon"]}, session,
                                       message.get("model"))
                elif kind == "ping":
                    await socket.send_json({"type": "pong"})
            except Exception as exc:                               # noqa: BLE001 - one bad turn must not kill the socket
                await socket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        return
