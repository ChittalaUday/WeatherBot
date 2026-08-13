"""
Model registry - one model, Model 1.

Model 1 is the v3 architecture: a coarse intent, a multi-label variable slot, and the three
presentation decisions (detail, chart, insights) the model makes instead of Python. The
earlier 14-class v1 and the slot-only v2 are gone; what survives of them is library code
Model 1 still imports - `clean_text` and the vectorizer from src/nlu.py, the span tagger
from src/tagger.py, the slot enums from src/v2/schema.py, the generator in src/v2/dataset.py.

`Understanding` stays the common denominator the rest of the backend consumes, so respond.py,
insights.py and the context engine never touch the model directly.

The bundle keeps its on-disk name (models/nlu_v3.joblib) and its version id ("v3"): those
name the architecture, and the conversation store has years of turns tagged that way.
"Model 1" is what it is called everywhere a human looks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import locations as locations_module          # re-exported for replay tests
from src.v3.model import BUNDLE_PATH, V3Model
from src.v3.schema import Detail, fields_for

VERSION = "v3"           # architecture id - names the bundle and tags stored turns
NAME = "Model 1"         # what humans see
DEFAULT_VERSION = VERSION
DESCRIPTION = "coarse intent + multi-label variables, and it picks detail, chart and insights - decides, never asks"

# Model 1 commits to a reading rather than stopping to ask.
NEVER_ASKS = {VERSION}


@dataclass
class Understanding:
    """One turn, in the shape the rest of the backend expects."""

    text: str
    version: str
    intent: str                      # coarse intent, for display
    action: str                      # GET | COMPARE | ALERT
    aggregation: str
    variables: list[str]             # one or more weather variables
    locations: list[str]
    times: list[str]
    times_normalized: list[str]
    confidence: float
    scores: dict = field(default_factory=dict)
    # the presentation decisions the model makes instead of Python
    detail: str = ""
    chart: str = ""
    insights: list = field(default_factory=list)
    assumed: list = field(default_factory=list)

    @property
    def decides_presentation(self) -> bool:
        return bool(self.detail)

    def fields(self) -> list:
        """Columns to fetch. The model's detail head picks the width."""
        return fields_for(self.variables, Detail(self.detail))


def _understand(model, text: str) -> Understanding:
    parsed = model.predict(text)
    action = {"COMPARE": "COMPARE", "ALERT": "ALERT"}.get(parsed.intent.value, "GET")
    return Understanding(
        text=text, version=VERSION,
        intent=parsed.intent.value,
        action=action,
        aggregation=parsed.aggregation.value,
        variables=[variable.value for variable in parsed.slots.variables],
        locations=list(parsed.slots.locations),
        times=list(parsed.slots.times),
        times_normalized=list(parsed.slots.times_normalized),
        confidence=parsed.confidence.get("intent", 0.0),
        scores=parsed.scores,
        detail=parsed.presentation.detail.value,
        chart=parsed.presentation.chart.value,
        insights=[insight.value for insight in parsed.presentation.insights],
    )


class Registry:
    """Lazy-loads the bundle once. Kept as a class so the serving path stays unchanged."""

    def __init__(self):
        self._model = None
        self._path = BUNDLE_PATH

    def available(self) -> list[dict]:
        return [{
            "version": VERSION,
            "name": NAME,
            "loaded": self._model is not None,
            "present": self._path.exists(),
            "size_mb": round(self._path.stat().st_size / 1e6, 1) if self._path.exists() else None,
            "description": DESCRIPTION,
        }]

    def get(self, version: str | None = None):
        """`version` is accepted and ignored - there is one model. Callers still pass it."""
        if self._model is None:
            if not self._path.exists():
                raise FileNotFoundError(
                    f"{NAME} bundle missing at {self._path} - "
                    f"rebuild with `python -m src.v3.model --export`")
            self._model = V3Model.load(self._path)
        return self._model

    def understand(self, text: str, version: str | None = None) -> Understanding:
        return _understand(self.get(), text)


def demo():
    """Self-check: Model 1 answers, reads both variables, and decides its own presentation."""
    registry = Registry()
    if not registry._path.exists():
        print("skip: bundle not built")
        return

    understanding = registry.understand("rain and temperature in Guntur tomorrow")
    assert understanding.version == VERSION
    assert understanding.locations == ["Guntur"], understanding.locations
    assert len(understanding.variables) >= 2, understanding.variables
    assert understanding.decides_presentation, "Model 1 must pick a detail level"
    assert understanding.fields(), understanding
    print(f"  {NAME}: intent={understanding.intent} action={understanding.action} "
          f"variables={understanding.variables} detail={understanding.detail} "
          f"chart={understanding.chart} conf={understanding.confidence:.2f}")
    print("registry demo OK")


if __name__ == "__main__":
    demo()
