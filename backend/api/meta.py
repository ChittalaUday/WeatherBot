"""
What this deployment is: health, the models it serves, their label sets, place autocomplete.

    GET /api/health            is it up, and what is configured
    GET /api/models            every model, its size, and how it scored at export time
    GET /api/labels?model=v4   the label sets a correction form has to offer
    GET /api/suggest?q=gun     location autocomplete for the picker
"""

from __future__ import annotations

import json

from fastapi import APIRouter

from backend import config
from backend.api.deps import registry
from backend.api.schemas import (
    HealthResponse,
    ModelListResponse,
    SuggestResponse,
    V4LabelsResponse,
)
from backend.nlu import DEFAULT_VERSION, MODELS, catalogue
from backend.pipeline import sources

router = APIRouter()


def served() -> list[dict]:
    """The switchable models. The local one is only offered if it actually answered at
    startup - a dropdown entry for a model that is not installed is a turn that fails."""
    from backend.api import GENERATION

    return catalogue(registry, bool(GENERATION.get("ok")))


@router.get("/api/health", response_model=HealthResponse)
def health():
    """Up, which bundles are present, and whether the wording layer is actually working.

    `generation.ok` is false when the local model is missing or misnamed. Answers still go out
    - the deterministic sentence is correct - but they are not phrased, and that is worth
    seeing on a dashboard rather than discovering by reading replies.
    """
    from backend.api import DUCKLING, GENERATION

    # `config.summary()` first: it carries a plain `duckling` URL and the probe below carries
    # whether that URL actually answered, which is the more useful of the two.
    return {"status": "ok", **config.summary(), "models": served(),
            "generation": dict(GENERATION), "duckling": dict(DUCKLING)}


@router.get("/api/models", response_model=ModelListResponse)
def list_models():
    """Every served model and its exported metrics.

    Per model, read from that model's own file - not one shared metrics blob, because a
    deployment serving one model's numbers under whichever model answered is a lie that
    survives for months without anyone noticing.
    """
    metrics = {}
    for version, spec in MODELS.items():
        path = spec["path"].with_name(f"metrics_{version}.json")
        if path.exists():
            metrics[version] = json.loads(path.read_text())
    return {"available": served(), "default": DEFAULT_VERSION, "metrics": metrics}


@router.get("/api/labels", response_model=V4LabelsResponse)
def labels(model: str = DEFAULT_VERSION):
    """The label sets the correction form offers. `model` is accepted and ignored - it is kept
    so a client that pins a version keeps working, and so a second label set has a place to
    go if a v5 head ever disagrees with this one."""
    from src.v4.schema import Activity, Aggregation, Intent, Variable, WeatherIntent

    return V4LabelsResponse(model="v4", name=MODELS["v4"]["name"],
                            intents=[i.value for i in Intent],
                            weather_intents=[w.value for w in WeatherIntent],
                            variables=[v.value for v in Variable],
                            activities=[a.value for a in Activity],
                            aggregations=[a.value for a in Aggregation])


@router.get("/api/suggest", response_model=SuggestResponse)
async def suggest(q: str):
    """Location autocomplete for the frontend picker."""
    async with sources.client() as http:
        return {"suggestions": await sources.suggest_locations(http, q)}
