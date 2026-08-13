"""
Model registry - v1 and v2 side by side, chosen per request.

The two models disagree about shape, not about the job: v1 folds the weather variable into a
14-class intent, v2 keeps a coarse intent and a multi-label variable slot. `Understanding`
is the common denominator the rest of the backend consumes, so respond.py, insights.py and
the context engine never learn which model answered.

    v1  weather_intent=RAIN, action=GET        -> variables=[RAIN],            action=GET
    v2  intent=FORECAST, variables=[RAIN,TEMP] -> variables=[RAIN,TEMPERATURE], action=GET
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import locations as locations_module          # re-exported for replay tests
from src.nlu import BUNDLE_PATH as V1_PATH, NLUModel
from src.v2.model import BUNDLE_PATH as V2_PATH, V2Model

DEFAULT_VERSION = "v1"

# v2 says GENERAL where v1 says CURRENT_CONDITIONS/FORECAST; the field map is keyed the v1 way.
VARIABLE_TO_FIELDS_KEY = {"GENERAL": "CURRENT_CONDITIONS"}


@dataclass
class Understanding:
    """One turn, in the shape the rest of the backend expects, whichever model produced it."""

    text: str
    version: str
    intent: str                      # for display: v1 weather_intent or v2 coarse intent
    action: str                      # GET | COMPARE | ALERT
    aggregation: str
    variables: list[str]             # one entry for v1, one or more for v2
    locations: list[str]
    times: list[str]
    times_normalized: list[str]
    confidence: float
    scores: dict = field(default_factory=dict)

    @property
    def field_keys(self) -> list[str]:
        """Keys into respond.INTENT_FIELDS, in the order the user asked for them."""
        return [VARIABLE_TO_FIELDS_KEY.get(variable, variable) for variable in self.variables]


def _from_v1(model, text: str) -> Understanding:
    parsed = model.predict(text)
    return Understanding(
        text=text, version="v1",
        intent=parsed.weather_intent.value,
        action=parsed.action.value,
        aggregation=parsed.aggregation.value,
        variables=[parsed.weather_intent.value],
        locations=list(parsed.entities.location),
        times=list(parsed.entities.time),
        times_normalized=list(parsed.entities.time_normalized),
        confidence=model.confidence(text),
        scores=dict(model.top_intents(text, k=5)),
    )


def _from_v2(model, text: str) -> Understanding:
    parsed = model.predict(text)
    # v2's coarse intent carries the action: COMPARE and ALERT are intents there
    action = {"COMPARE": "COMPARE", "ALERT": "ALERT"}.get(parsed.intent.value, "GET")
    return Understanding(
        text=text, version="v2",
        intent=parsed.intent.value,
        action=action,
        aggregation=parsed.aggregation.value,
        variables=[variable.value for variable in parsed.slots.variables],
        locations=list(parsed.slots.locations),
        times=list(parsed.slots.times),
        times_normalized=list(parsed.slots.times_normalized),
        confidence=parsed.confidence.get("intent", 0.0),
        scores=parsed.scores,
    )


class Registry:
    """Lazy-loads each bundle once. A missing v2 simply means v2 is not offered."""

    def __init__(self):
        self._models: dict[str, object] = {}
        self._adapters = {"v1": _from_v1, "v2": _from_v2}
        self._paths = {"v1": V1_PATH, "v2": V2_PATH}

    def available(self) -> list[dict]:
        return [
            {"version": version,
             "loaded": version in self._models,
             "present": path.exists(),
             "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else None,
             "description": DESCRIPTIONS[version]}
            for version, path in self._paths.items()
        ]

    def get(self, version: str | None):
        version = version if version in self._paths else DEFAULT_VERSION
        if version not in self._models:
            if not self._paths[version].exists():
                raise FileNotFoundError(f"{version} bundle missing at {self._paths[version]}")
            loader = NLUModel.load if version == "v1" else V2Model.load
            self._models[version] = loader(self._paths[version])
        return self._models[version]

    def understand(self, text: str, version: str | None = None) -> Understanding:
        version = version if version in self._paths else DEFAULT_VERSION
        return self._adapters[version](self.get(version), text)


DESCRIPTIONS = {
    "v1": "14-class intent, one variable per query (MODEL_RULES v1)",
    "v2": "coarse intent + multi-label variables, multi-value slots",
}


def demo():
    """Self-check: both models answer, and v2 can return two variables where v1 cannot."""
    registry = Registry()
    query = "rain and temperature in Guntur tomorrow"

    for version in ("v1", "v2"):
        if not registry._paths[version].exists():
            print(f"skip {version}: bundle not built")
            continue
        understanding = registry.understand(query, version)
        assert understanding.version == version
        assert understanding.locations == ["Guntur"], understanding.locations
        assert understanding.field_keys, understanding
        print(f"  {version}: intent={understanding.intent} action={understanding.action} "
              f"variables={understanding.variables} conf={understanding.confidence:.2f}")

    if registry._paths["v2"].exists():
        v2 = registry.understand(query, "v2")
        assert len(v2.variables) >= 2, f"v2 should see both variables: {v2.variables}"
    print("registry demo OK")


if __name__ == "__main__":
    demo()
