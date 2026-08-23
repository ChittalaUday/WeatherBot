"""
The retraining loop: what a user said about an answer, and what is still waiting for a label.

    POST /api/feedback              thumbs, a correction, or the intent picked from a clarify
    GET  /api/feedback/{turn_id}    what was already said, so a reopened chat shows its ratings
    GET  /api/review                turns waiting for a human label
    GET  /api/stats                 how the deployment is doing

A `choice` is the cheapest gold label there is: the model was unsure, a human answered. Human
labels outrank the model's (Rule 8.5), which is what `store.training_rows` acts on.
"""

from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend import store
from backend.api.deps import db

router = APIRouter()


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


@router.post("/api/feedback")
def record(body: Feedback):
    store.record_feedback(db, body.turn_id, body.kind, intent=body.intent, action=body.action,
                          variables=body.variables, location=body.location,
                          time_raw=body.time, model=body.model, error_type=body.error_type,
                          note=body.note)
    return {"ok": True, "labelled": len(store.training_rows(db)),
            "feedback": store.feedback_for(db, body.turn_id)}


@router.get("/api/feedback/{turn_id}")
def for_turn(turn_id: int):
    return {"feedback": store.feedback_for(db, turn_id)}


@router.get("/api/review")
def review(limit: int = 50):
    """Turns flagged wrong, or answered uncertainly and never rated."""
    return {"queue": store.review_queue(db, limit)}


@router.get("/api/stats")
def stats():
    return store.stats(db)
