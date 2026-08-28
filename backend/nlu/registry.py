"""
Model registry - the trained model, and the shape everything downstream reads.

    Model 2  v4  four heads and a tagger; everything derivable is derived, not predicted

`Understanding` is the common denominator the rest of the backend consumes, so the pipeline,
the generation layer and the context engine never touch a model directly. `backend.nlu.llm`
(Model 3, hosted) produces the same shape without being in this registry: it loads no bundle,
and it is only reachable through /api/compare.

The bundle keeps its on-disk name (models/nlu_v4.joblib) and its version id ("v4"): that names
the architecture, and the conversation store has turns tagged that way. "Model 2" is what a
human sees. A registry with one entry is kept because it is what makes adding v5 a data
change - `MODELS` and a loader branch - rather than a rewrite of every caller.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.config import DEFAULT_MODEL
from src.normalize import normalize as normalize_text
from src.v4.model import BUNDLE_PATH as V4_BUNDLE
from src.v4.model import V4Model
from src.v4.schema import (
    CONTROL,
    DECLINED,
    NO_DATA_NEEDED,
    REPLIES,
    detail_from_text,
    sub_activity_for,
    venue_for,
    weather_intent_for,
)
from src.v4.schema import Intent as V4Intent
from src.v4.schema import fields_for as v4_fields_for

MODELS = {
    "v4": {"name": "Model 2", "path": V4_BUNDLE,
           "description": "16 intents incl. chat and control, 12 activities, entity gazetteer; "
                          "weather window, action group and sub-activity are derived, not predicted"},
}
DEFAULT_VERSION = DEFAULT_MODEL if DEFAULT_MODEL in MODELS else "v4"
# The trained model commits to a reading rather than stopping to ask (MODEL_RULES Rule 1.1),
# so the clarify branches downstream are dead for it - kept only because a future head that
# does ask would need them back.
NEVER_ASKS = frozenset(MODELS)

__all__ = ["DEFAULT_VERSION", "MODELS", "NEVER_ASKS", "Registry", "Understanding", "normalize_text"]


@dataclass
class Understanding:
    """One turn, in the shape the rest of the backend expects."""

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
    # answer width, read off the words ("full report" vs "just the temperature")
    detail: str = ""
    assumed: list = field(default_factory=list)
    activity: str = "NONE"           # what to decide, for the advice engine
    sub_activity: str = ""           # descriptive, tunes a threshold at most
    entities: dict = field(default_factory=dict)
    family: str = "data"             # data | conversational | control | declined
    reply: str = ""                  # canned answer for a turn that needs no weather

    @property
    def needs_weather(self) -> bool:
        """False for a greeting, a reset, or a question we decline - skip the whole pipeline."""
        return self.family == "data"

    @property
    def weather_intent(self) -> str:
        """Which temporal operation this turn needs: NONE|CURRENT|TOMORROW|FORECAST|HISTORICAL.

        A property, not a field, and that is the point: it is a function of the time slot, so
        derived it can never disagree with `times_normalized` the way a second predicted head
        could. `src.v4.schema.weather_intent_for` has always existed and `backend.nlu.llm` has
        always computed this - into a dict nothing downstream read. Here it is, on the object
        the pipeline actually gets, which is what lets a route be chosen at all.

        A turn that needs no weather has no window: `weather_intent_for(None)` would say
        FORECAST, which is the right default for a question and wrong for "hey there".
        """
        if not self.needs_weather:
            return "NONE"
        return weather_intent_for(
            self.times_normalized[0] if self.times_normalized else None).value

    @property
    def venue(self) -> str:
        """"outdoor" or "indoor" - whether the weather reaches this activity at all.

        Derived from the words and the sub-activity, both of which are already extracted. See
        `src.v4.schema.venue_for` for why outdoor is the safe default.
        """
        return venue_for(self.activity, self.sub_activity, self.text)

    def fields(self) -> list:
        """Columns to fetch. Model 3 speaks the v4 schema too - it is prompted with those
        enums - so both versions map the same way."""
        return v4_fields_for(self.variables, self.detail or "NORMAL")


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
            self._loaded[version] = V4Model.load(spec["path"])
        return self._loaded[version]

    def understand(self, text: str, version: str | None = None) -> Understanding:
        version = version if version in MODELS else DEFAULT_VERSION
        return _understand_v4(self.get(version), text)
