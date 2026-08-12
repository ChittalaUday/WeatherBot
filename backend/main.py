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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import insights, locations, respond, store, weather
from src.nlu import NLUModel

app = FastAPI(title="WeatherBot", version="1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

model: NLUModel | None = None
db = store.connect()                 # every turn is logged for the retraining loop


@app.on_event("startup")
def load_model():
    global model
    model = NLUModel.load()          # ~14 MB, loaded once for the process


@app.get("/api/health")
def health():
    return {"status": "ok", "model": model is not None}


@app.get("/api/suggest")
async def suggest(q: str):
    """Location autocomplete for the frontend picker (TanStack Query polls this)."""
    async with weather.client() as http:
        return {"suggestions": await weather.suggest_locations(http, q)}


# Below this the model is guessing, and a confident-looking table would be a lie.
MIN_CONFIDENCE = 0.45


async def handle_query(socket: WebSocket, text: str, coords: dict | None, session: str):
    """One user turn: understand -> locate -> fetch -> table, all of it logged."""
    started = time.perf_counter()
    await socket.send_json({"type": "status", "stage": "understanding"})
    parsed = model.predict(text)
    intent, action = parsed.weather_intent.value, parsed.action.value
    # the model picks the reduction (Rule 2.3); the guard drops it when the prompt contains
    # no word that could have meant it - "weather in KKD" is not a request for a maximum
    aggregation = insights.confirm_aggregation(text, parsed.aggregation.value)
    spans = parsed.entities

    await socket.send_json({
        "type": "nlu",
        "intent": intent,
        "action": action,
        "aggregation": aggregation,
        "entities": {
            "location": spans.location,
            "time": spans.time,
            "time_normalized": spans.time_normalized,
        },
        "confidence": round(model.confidence(text), 4),
    })

    # "angara vs hyderbad" names no metric at all - answering it with whatever the
    # classifier ranked first is how you get RAIN one turn and TEMPERATURE the next.
    confidence = model.confidence(text)
    log = lambda outcome, detail, **extra: store.record_turn(
        db, session, text, intent=intent, action=action, confidence=confidence,
        location=spans.location, time_raw=spans.time, time_norm=spans.time_normalized,
        outcome=outcome, detail=detail,
        latency_ms=int((time.perf_counter() - started) * 1000), **extra)

    if confidence < MIN_CONFIDENCE:
        turn_id = log("clarified", "low confidence")
        await socket.send_json({
            "type": "clarify",
            "turn_id": turn_id,
            "message": "I am not sure which reading you want. Pick one:",
            "options": [
                {"intent": intent_name, "confidence": round(probability, 3),
                 "label": intent_name.replace("_", " ").lower()}
                for intent_name, probability in model.top_intents(text)
            ],
            "text": text,
        })
        return

    named = [name for name in spans.location
             if not locations.is_relative(name) and not locations.is_probably_not_a_place(name)]
    relative = [name for name in spans.location if locations.is_relative(name)]

    # A comparison with one place is not a comparison; ask instead of quietly answering
    # about whichever place happened to be extracted.
    if action == "COMPARE" and len(named) + len(relative) < 2:
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
    if not named and not coords:
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
        if not places and coords:
            places = [await weather.reverse_geocode(http, coords["lat"], coords["lon"])]

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

        normalized = spans.time_normalized[0] if spans.time_normalized else ""
        hourly = respond.needs_hourly(normalized, aggregation)
        fetch = weather.hourly_forecast if hourly else weather.daily_forecast
        try:
            feeds = await asyncio.gather(*(fetch(http, p["lat"], p["lon"]) for p in places))
        except Exception as exc:                                   # noqa: BLE001 - report upstream failures verbatim
            log("error", f"WeatherSnap API failed: {exc}", places=places)
            await socket.send_json({"type": "error", "message": f"WeatherSnap API failed: {exc}"})
            return

    fields = respond.INTENT_FIELDS[intent]
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

    turn_id = log("answered", summary, places=places, unresolved=unknown)
    await socket.send_json({
        "type": "result",
        "turn_id": turn_id,
        "intent": intent,
        "action": action,
        "when": when,
        "places": places,
        "granularity": "hourly" if hourly else "daily",
        "summary": summary,
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
                    await handle_query(socket, message["text"], message.get("coords"), session)
                elif kind == "location":
                    # browser answered the geolocation prompt: rerun the pending query
                    await handle_query(socket, message["text"],
                                       {"lat": message["lat"], "lon": message["lon"]}, session)
                elif kind == "ping":
                    await socket.send_json({"type": "pong"})
            except Exception as exc:                               # noqa: BLE001 - one bad turn must not kill the socket
                await socket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        return
