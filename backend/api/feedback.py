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

from fastapi import APIRouter

from backend import store
from backend.api.deps import db
from backend.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    ReviewQueueResponse,
    StatsResponse,
    TurnFeedbackResponse,
)

router = APIRouter()


@router.post("/api/feedback", response_model=FeedbackResponse)
def record(body: FeedbackRequest):
    store.record_feedback(db, body.turn_id, body.kind, intent=body.intent, action=body.action,
                          variables=body.variables, location=body.location,
                          time_raw=body.time, model=body.model, error_type=body.error_type,
                          note=body.note)
    return {"ok": True, "labelled": len(store.training_rows(db)),
            "feedback": store.feedback_for(db, body.turn_id)}


@router.get("/api/feedback/{turn_id}", response_model=TurnFeedbackResponse)
def for_turn(turn_id: int):
    return {"feedback": store.feedback_for(db, turn_id)}


@router.get("/api/review", response_model=ReviewQueueResponse)
def review(limit: int = 50):
    """Turns flagged wrong, or answered uncertainly and never rated."""
    return {"queue": store.review_queue(db, limit)}


@router.get("/api/stats", response_model=StatsResponse)
def stats():
    return store.stats(db)
