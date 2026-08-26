"""
Every setting this service reads, in one place, resolved once.

Two rules, and both were bugs before they were rules:

  - `.env` is loaded before any module reads a variable. Modules used to call `os.getenv` at
    import time, so importing one of them before the file was read left it permanently
    misconfigured - the hosted NLU model spent a week reporting "not configured" that way.
  - nothing else calls `os.getenv`. A setting that is read in two places drifts in one of
    them; a setting that is read here is the same value everywhere.

`.env` parsing is six lines rather than python-dotenv: the file is KEY=value, and the
existing environment always wins, so a real deployment sets variables the normal way.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def load_env(path: Path = ENV_FILE) -> list[str]:
    """Fill anything not already in the environment. Returns the keys it set."""
    if not path.exists():
        return []
    filled = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value
            filled.append(key)
    return filled


load_env()          # before the constants below are evaluated


# --- paths -------------------------------------------------------------------
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "conversations.db"

# --- weather sources ---------------------------------------------------------
GFS_URL = os.getenv("GFS_API_URL", "https://gfsapi.niruthiapptesting.com")
GFS_HISTORICAL_URL = os.getenv("GFS_HISTORICAL_URL", "https://stg-gfs.niruthiapptesting.com")
SOLR_URL = os.getenv("SOLR_URL", "https://solr.apps.niruthi.com")
SOLR_AUTH = os.getenv("SOLR_AUTH_HEADER", "Basic YXBpOk5pcnV0aGlAMjRVc2Vy")
INFEST_URL = os.getenv("INFEST_SERVER_URL", "https://infestserver.apps.niruthi.com")
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "20"))
HTTP_CONNECT_TIMEOUT = float(os.getenv("HTTP_CONNECT_TIMEOUT", "10"))

# The archive lives on an internal address. Without the key it is unreachable, and the query
# planner refuses old dates up front rather than after a timeout.
ZARR_URL = os.getenv("ZARR_URL", "http://172.16.16.111:8550")
ZARR_KEY = os.getenv("ZARR_API_KEY", "e87aef62-d100-4703-b577-8df5f2418332")          # server-side only; never sent to a client
ARCHIVE_REACHABLE = bool(ZARR_KEY)

# --- generation (the local model that words every reply) ---------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "20"))

# --- hosted NLU (Model 3, the comparison column) -----------------------------
AI_BASE_URL = os.getenv("AI_BASE_URL", "https://integrate.api.nvidia.com/v1")
AI_API_KEY = os.getenv("API_KEY", "")
AI_MODEL = os.getenv("AI_MODEL", "z-ai/glm-5.2")
AI_TIMEOUT = float(os.getenv("AI_TIMEOUT", "45"))

# --- serving -----------------------------------------------------------------
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "v4")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]

# Confidence routing, from `python src/nlu.py --calibrate` on the hand-written eval set:
#   >= 0.95   98.9% accurate over 83% of turns   -> answer
#   0.45-0.95 ~75% accurate                      -> answer, but flag it for review
#   < 0.45    0-67% accurate                     -> too thin to build an answer on
# Re-run the calibration after every retrain rather than trusting these numbers forever.
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.45"))
CONFIDENT = float(os.getenv("CONFIDENT", "0.95"))

# How many rows one answer may carry back. Not a database limit - a limit on what the
# aggregator, the chart and the reply can do something useful with.
MAX_ROWS = int(os.getenv("MAX_ROWS", "400"))


def summary() -> dict:
    """What is configured, with no secret values in it. Served by /api/health."""
    return {
        "default_model": DEFAULT_MODEL,
        "archive": ARCHIVE_REACHABLE,
        "hosted_nlu": bool(AI_API_KEY),
        "generation_model": OLLAMA_MODEL,
    }


if __name__ == "__main__":
    print("filled from .env:", load_env())
    for key, value in summary().items():
        print(f"  {key:18s} {value}")
