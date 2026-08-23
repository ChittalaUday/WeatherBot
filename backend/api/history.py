"""
Past conversations, replayed from what was stored rather than re-asked.

    GET /api/chats            recent conversations, newest first, for the history panel
    GET /api/chats/{chat_id}  every turn of one, with the answers as they were rendered

Replayed, not re-queried: an old chat shows the forecast it actually gave. A re-query would
show today's weather under yesterday's question, which is a different answer wearing the same
timestamp.
"""

from __future__ import annotations

from fastapi import APIRouter

from backend import store
from backend.api.deps import db

router = APIRouter()


@router.get("/api/chats")
def chats(limit: int = 40):
    return {"chats": store.list_chats(db, limit)}


@router.get("/api/chats/{chat_id}")
def conversation(chat_id: str):
    return {"chat_id": chat_id, "turns": store.conversation(db, chat_id)}
