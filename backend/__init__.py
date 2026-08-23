"""
WeatherBot backend.

    backend.config      every setting, loaded from .env once
    backend.nlu         text -> Understanding, and what the conversation remembers
    backend.pipeline    Understanding -> an answer: places, plan, fetch, quality, analysis
    backend.generation  the answer, said in words a person would use
    backend.api         the HTTP surface
    backend.store       the turn log and the retraining export

The repo root goes on sys.path here, once, so that `src.*` resolves however the process was
started - `uvicorn backend.api:app`, `python -m backend.pipeline.plan`, or a notebook. Eight
modules each did this for themselves before, which meant eight chances to get it wrong and no
single place to fix it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
