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
from backend.nlu import DEFAULT_VERSION, MODELS
from backend.pipeline import sources

router = APIRouter()


@router.get("/api/health")
def health():
    """Up, which bundles are present, and whether the wording layer is actually working.

    `generation.ok` is false when the local model is missing or misnamed. Answers still go out
    - the deterministic sentence is correct - but they are not phrased, and that is worth
    seeing on a dashboard rather than discovering by reading replies.
    """
    from backend.api import GENERATION

    return {"status": "ok", "models": registry.available(), "generation": dict(GENERATION),
            **config.summary()}


@router.get("/api/models")
def list_models():
    """Every served model and its exported metrics.

    Per model, read from that model's own file. Serving one model's metrics under whichever
    model answered is how a deployment defaulting to v4 spent months reporting v3's numbers.
    """
    metrics = {}
    for version, spec in MODELS.items():
        path = spec["path"].with_name(f"metrics_{version}.json")
        if path.exists():
            metrics[version] = json.loads(path.read_text())
    return {"available": registry.available(), "default": DEFAULT_VERSION, "metrics": metrics}


@router.get("/api/labels")
def labels(model: str = DEFAULT_VERSION):
    """The label sets the correction form offers, for the model that answered.

    Per model, because they do not agree: v3 has 6 intents and 13 variables, v4 has 16 and 10.
    Serving v3's list against a v4 turn offers labels that model cannot predict.
    """
    if model == "v3":
        from src.v2.schema import Intent, Variable
        from src.v3.schema import ChartKind, Detail, Insight

        return {"model": "v3", "name": MODELS["v3"]["name"],
                "intents": [i.value for i in Intent],
                "variables": [v.value for v in Variable],
                "detail": [d.value for d in Detail],
                "chart": [c.value for c in ChartKind],
                "insights": [i.value for i in Insight]}

    from src.v4.schema import Activity, Aggregation, Intent, Variable, WeatherIntent

    return {"model": "v4", "name": MODELS["v4"]["name"],
            "intents": [i.value for i in Intent],
            "weather_intents": [w.value for w in WeatherIntent],
            "variables": [v.value for v in Variable],
            "activities": [a.value for a in Activity],
            "aggregations": [a.value for a in Aggregation]}


@router.get("/api/suggest")
async def suggest(q: str):
    """Location autocomplete for the frontend picker."""
    async with sources.client() as http:
        return {"suggestions": await sources.suggest_locations(http, q)}
