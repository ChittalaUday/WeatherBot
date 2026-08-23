"""
The HTTP surface. Plain requests only - no WebSocket, no connection held between turns.

    uvicorn backend.api:app --reload --port 8787

    POST /api/chat              one turn, streamed back as server-sent events
    POST /api/chat/reset        forget this chat's slots, hand back a fresh id
    POST /api/compare           the same sentence through every model, streamed
    GET  /api/health            is it up, and what is configured
    GET  /api/models            every served model and its exported metrics
    GET  /api/labels            the label sets the correction form offers
    GET  /api/suggest           location autocomplete
    GET  /api/chats             recent conversations
    GET  /api/chats/{id}        one conversation, replayed
    POST /api/feedback          thumbs, corrections, clarify choices
    GET  /api/feedback/{id}     what was already said about a turn
    GET  /api/review            turns waiting for a human label
    GET  /api/stats             usage

Every route lives in its own module and the app only wires them together, so adding one does
not mean reading four hundred lines of turn handling to find where the decorators end.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend import generation, store
from backend.api import chat, feedback, history, meta
from backend.api.deps import registry
from backend.config import CORS_ORIGINS

GENERATION: dict = {}                  # what probe() found at startup; served by /api/health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the default bundle, and say once what is and is not working.

    Both checks are startup-only and both are about failures that are invisible from inside a
    turn: a store that will be quarantined on first write, and a generation model that is not
    there - which silently costs every answer its wording and nothing complains.
    """
    if not store.healthy():
        print("store: conversations.db fails its integrity check - it will be moved aside on "
              "first use and a fresh one started. The old file is kept.")
    GENERATION.update(await generation.probe())
    if not GENERATION.get("ok"):
        print(f"generation: {GENERATION['note']}")
    registry.get()                     # so the first real turn is not the one that pays
    yield


app = FastAPI(title="WeatherBot", version="2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                   allow_headers=["*"])

for module in (chat, meta, history, feedback):
    app.include_router(module.router)

__all__ = ["app"]
