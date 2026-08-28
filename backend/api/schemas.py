"""
Pydantic schemas for WeatherBot API request payloads, response payloads, and event streams.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Request Payloads
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    text: str
    chat_id: Optional[str] = None
    model: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None


class ResetChatRequest(BaseModel):
    chat_id: Optional[str] = None


class CompareRequest(BaseModel):
    text: str
    chat_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    turn_id: int
    kind: str  # up | down | correction | choice
    intent: Optional[str] = None
    action: Optional[str] = None
    variables: Optional[List[str]] = None
    location: Optional[List[str]] = None
    time: Optional[List[str]] = None
    model: Optional[str] = None
    error_type: Optional[str] = None
    note: Optional[str] = None


# ---------------------------------------------------------------------------
# Response Payloads
# ---------------------------------------------------------------------------


class ResetChatResponse(BaseModel):
    chat_id: str
    message: str


class HealthResponse(BaseModel):
    status: str
    models: List[str]
    generation: Dict[str, Any]
    db_path: str
    archive_url: str
    solr_url: str
    ollama_url: str
    ai_model: str
    cors_origins: List[str]


class ModelListResponse(BaseModel):
    available: List[str]
    default: str
    metrics: Dict[str, Any]


class V4LabelsResponse(BaseModel):
    model: Literal["v4"] = "v4"
    name: str
    intents: List[str]
    weather_intents: List[str]
    variables: List[str]
    activities: List[str]
    aggregations: List[str]


class SuggestResponse(BaseModel):
    suggestions: List[str]


class ChatItem(BaseModel):
    chat_id: str
    title: str
    turns: int
    updated_at: str


class ChatListResponse(BaseModel):
    chats: List[Dict[str, Any]]


class ConversationResponse(BaseModel):
    chat_id: str
    turns: List[Dict[str, Any]]


class FeedbackResponse(BaseModel):
    ok: bool
    labelled: int
    feedback: Optional[Dict[str, Any]] = None


class TurnFeedbackResponse(BaseModel):
    feedback: Optional[Dict[str, Any]] = None


class ReviewQueueResponse(BaseModel):
    queue: List[Dict[str, Any]]


class StatsResponse(BaseModel):
    turns: int
    chats: int
    rated: int
    thumbs_up: int
    thumbs_down: int
    corrections: int
    choices: int


# ---------------------------------------------------------------------------
# Stream Event Models (SSE)
# ---------------------------------------------------------------------------


class StatusEvent(BaseModel):
    type: Literal["status"] = "status"
    stage: str


class NLUEvent(BaseModel):
    type: Literal["nlu"] = "nlu"
    intent: str
    action: str
    confidence: float
    locations: List[str]
    times: List[str]


class ThinkingEvent(BaseModel):
    type: Literal["thinking"] = "thinking"
    text: str


class DeltaEvent(BaseModel):
    type: Literal["delta"] = "delta"
    text: str


class ErrorEvent(BaseModel):
    type: Literal["error"] = "error"
    message: str
