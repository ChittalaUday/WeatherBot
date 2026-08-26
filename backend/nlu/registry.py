"""
Model registry - the trained models, side by side.

    Model 1  v3  coarse intent + multi-label variables, and it picks detail / chart / insights
    Model 2  v4  four heads and a tagger; everything derivable is derived, not predicted

Both answer the same endpoint and the client picks per turn, so the two can be compared on the
same question without a redeploy. `Understanding` is the common denominator the rest of the
backend consumes, so the pipeline, the generation layer and the context engine never touch a
model directly - and the v4-only fields default to empty rather than being absent, so v3 turns
keep flowing through code that reads them. `backend.nlu.llm` (Model 3, hosted) produces the
same shape without being in this registry: it loads no bundle.

The bundles keep their on-disk names (models/nlu_v3.joblib, models/nlu_v4.joblib) and their
version ids ("v3", "v4"): those name the architecture, and the conversation store has turns
tagged that way. "Model 1" and "Model 2" are what a human sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.config import DEFAULT_MODEL
from src.normalize import normalize as normalize_text
from src.v3.model import BUNDLE_PATH as V3_BUNDLE
from src.v3.model import V3Model
from src.v3.schema import Detail
from src.v3.schema import fields_for as v3_fields_for
from src.v4.model import BUNDLE_PATH as V4_BUNDLE
from src.v4.model import V4Model
from src.v4.schema import (
    CONTROL,
    DECLINED,
    NO_DATA_NEEDED,
    REPLIES,
    detail_from_text,
    sub_activity_for,
)
from src.v4.schema import Intent as V4Intent
from src.v4.schema import fields_for as v4_fields_for

MODELS = {
    "v3": {"name": "Model 1", "path": V3_BUNDLE,
           "description": "coarse intent + multi-label variables; picks detail, chart and "
                          "insights - decides, never asks"},
    "v4": {"name": "Model 2", "path": V4_BUNDLE,
           "description": "16 intents incl. chat and control, 12 activities, entity gazetteer; "
                          "weather window, action group and sub-activity are derived, not predicted"},
}
DEFAULT_VERSION = DEFAULT_MODEL if DEFAULT_MODEL in MODELS else "v4"
# Both trained models commit to a reading rather than stopping to ask (MODEL_RULES Rule 1.1),
# so the clarify branches downstream are dead for them - kept only because a future head that
# does ask would need them back.
NEVER_ASKS = frozenset(MODELS)

__all__ = ["DEFAULT_VERSION", "MODELS", "NEVER_ASKS", "Registry", "Understanding", "normalize_text"]


@dataclass
class Understanding:
    """One turn, in the shape the rest of the backend expects.

    Everything below `assumed` is v4-only and defaults to empty, so a v3 turn passes through
    the same code path untouched.
    """

    text: str
    version: str
    intent: str                      # coarse intent, for display
    action: str                      # GET | COMPARE | ALERT
    aggregation: str
    variables: list[str]
    locations: list[str]
    times: list[str]
    times_normalized: list[str]
    confidence: float
    scores: dict = field(default_factory=dict)
    # the presentation decisions v3 makes instead of Python
    detail: str = ""
    chart: str = ""
    insights: list = field(default_factory=list)
    assumed: list = field(default_factory=list)
    # --- v4 only ---
    activity: str = "NONE"           # what to decide, for the advice engine
    sub_activity: str = ""           # descriptive, tunes a threshold at most
    entities: dict = field(default_factory=dict)
    family: str = "data"             # data | conversational | control | declined
    reply: str = ""                  # canned answer for a turn that needs no weather

    @property
    def needs_weather(self) -> bool:
        """False for a greeting, a reset, or a question we decline - skip the whole pipeline."""
        return self.family == "data"

    def fields(self) -> list:
        """Columns to fetch. v3's detail head picks the width; v4 maps its own variables.

        Model 3 speaks the v4 schema too - it is prompted with those enums - so it maps the
        same way. Sending it down the v3 path raised `'WIND' is not a valid Variable`, because
        v3 predates the WIND_SPEED/WIND_DIRECTION merge.
        """
        if self.version in {"v4", "llm"}:
            return v4_fields_for(self.variables, self.detail or "NORMAL")
        return v3_fields_for(self.variables, Detail(self.detail or "NORMAL"))


def _understand_v3(model, text: str) -> Understanding:
    parsed = model.predict(text)
    action = {"COMPARE": "COMPARE", "ALERT": "ALERT"}.get(parsed.intent.value, "GET")
    return Understanding(
        text=text, version="v3",
        intent=parsed.intent.value, action=action,
        aggregation=parsed.aggregation.value,
        variables=[v.value for v in parsed.slots.variables],
        locations=list(parsed.slots.locations),
        times=list(parsed.slots.times),
        times_normalized=list(parsed.slots.times_normalized),
        confidence=parsed.confidence.get("intent", 0.0),
        scores=parsed.scores,
        detail=parsed.presentation.detail.value,
        chart=parsed.presentation.chart.value,
        insights=[i.value for i in parsed.presentation.insights],
    )


def _family_of(intent: V4Intent) -> str:
    if intent in CONTROL:
        return "control"
    if intent in DECLINED:
        return "declined"
    if intent in NO_DATA_NEEDED:
        return "conversational"
    return "data"


def _understand_v4(model, text: str) -> Understanding:
    parsed = model.predict(text)
    family = _family_of(parsed.intent)
    return Understanding(
        text=text, version="v4",
        intent=parsed.intent.value,
        # the backend's COMPARE/GET split is a different axis from v4's intent, so it is
        # derived here rather than being a second thing the model has to agree with itself on
        action="COMPARE" if parsed.intent is V4Intent.COMPARISON else "GET",
        aggregation=parsed.aggregation.value,
        variables=[v.value for v in parsed.slots.variables],
        locations=list(parsed.slots.locations),
        times=list(parsed.slots.times),
        times_normalized=list(parsed.slots.times_normalized),
        confidence=parsed.confidence.get("intent", 0.0),
        scores=parsed.scores,
        activity=parsed.activity.value,
        sub_activity=sub_activity_for(parsed.activity, parsed.slots.entities, text),
        entities=dict(parsed.slots.entities),
        family=family,
        reply=(REPLIES.get(parsed.intent) or [""])[0] if family != "data" else "",
        # width is a lexical fact, not a prediction - "full report" vs "just the temperature"
        detail=detail_from_text(text),
    )


class Registry:
    """Lazy-loads each bundle once, on first use of that version."""

    def __init__(self):
        self._loaded: dict = {}

    def available(self) -> list[dict]:
        return [{
            "version": version,
            "name": spec["name"],
            "loaded": version in self._loaded,
            "present": spec["path"].exists(),
            "size_mb": round(spec["path"].stat().st_size / 1e6, 1) if spec["path"].exists() else None,
            "description": spec["description"],
            "default": version == DEFAULT_VERSION,
        } for version, spec in MODELS.items()]

    def get(self, version: str | None = None):
        version = version if version in MODELS else DEFAULT_VERSION
        if version not in self._loaded:
            spec = MODELS[version]
            if not spec["path"].exists():
                raise FileNotFoundError(
                    f"{spec['name']} bundle missing at {spec['path']} - rebuild with "
                    f"`python -m src.{version}.model --export`")
            self._loaded[version] = (V3Model if version == "v3" else V4Model).load(spec["path"])
        return self._loaded[version]

    def understand(self, text: str, version: str | None = None) -> Understanding:
        version = version if version in MODELS else DEFAULT_VERSION
        model = self.get(version)
        return (_understand_v3 if version == "v3" else _understand_v4)(model, text)
