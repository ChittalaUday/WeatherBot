"""
The same sentence through every model, streamed as each one finishes.

    python -m backend.api.compare "should i spray in Guntur tomorrow"

Two trained models answer in milliseconds offline; the hosted one is a network round trip to a
model with no training on this label set at all, working purely from the schema in its prompt.
Putting them side by side on the same sentence is the only honest way to know what the trained
models are worth - and where a general model is simply better, which is worth knowing too.

Each column carries the *same* payload a chat turn produces, built by the same
`Answer.payload`, so a compared answer and a chatted answer are the same object and the UI
renders them with the same components.

Nothing waits for the slowest:

    compare_start   the columns, empty, so the UI can lay them out
    compare_result  one column, complete, as soon as it is
    compare_done    the disagreements, once every column is in
"""

from __future__ import annotations

import asyncio
import time

from backend.api.deps import registry
from backend.nlu import MODELS, Understanding, normalize_text
from backend.nlu import llm as hosted
from backend.pipeline import run, sources
from src.v4.schema import weather_intent_for

# Slots compared field by field. Scalars must match exactly; the three below are sets, because
# ["Guntur", "Vizag"] and ["Vizag", "Guntur"] are the same reading.
SCALAR_FIELDS = ("intent", "weather_intent", "activity", "aggregation")
LIST_FIELDS = ("variables", "locations", "times")


def _from_hosted(column: dict, text: str) -> Understanding | None:
    """The hosted model's JSON, in the same Understanding the trained models produce.

    Built here rather than in `nlu.llm` so that module stays a pure client - and so the
    downstream pipeline genuinely cannot tell which model it is running for.
    """
    if not column.get("ok"):
        return None
    return Understanding(
        text=text, version="llm", intent=column["intent"],
        action="COMPARE" if column["intent"] == "COMPARISON" else "GET",
        aggregation=column["aggregation"], variables=column["variables"],
        locations=column["locations"], times=column["times"],
        times_normalized=column["times_normalized"], confidence=1.0,
        activity=column["activity"], sub_activity=column.get("sub_activity", ""),
        entities=column.get("entities", {}), family=column["family"],
        reply=column.get("reply", ""), detail="NORMAL")


def contenders() -> list[dict]:
    """Every model this deployment can compare, trained ones first."""
    entries = [{"version": v, "name": spec["name"], "kind": "local", "provider": ""}
               for v, spec in MODELS.items()]
    entries.append({"version": "llm", "name": hosted.NAME, "kind": "hosted",
                    "provider": hosted.model_name()})
    return entries


async def _column(version: str, text: str) -> dict:
    """One model, end to end. Never raises - a dead column must not take the others."""
    began = time.perf_counter()
    try:
        if version == "llm":
            got = await hosted.understand(text)
            if not got.get("ok"):
                return {"version": version, "ok": False, "error": got["error"],
                        "latency_ms": got.get("latency_ms", 0)}
            understanding = _from_hosted(got, text)
            nlu_ms, usage = got.get("latency_ms", 0), got.get("usage", {})
        else:
            understanding = registry.understand(text, version)
            nlu_ms, usage = int((time.perf_counter() - began) * 1000), {}
    except Exception as exc:                                        # noqa: BLE001
        return {"version": version, "ok": False, "latency_ms": 0,
                "error": f"{type(exc).__name__}: {exc}"}

    if understanding is None:
        return {"version": version, "ok": False, "latency_ms": nlu_ms, "error": "Null understanding"}

    slots = {
        # v3 has no weather_intent head; deriving it from the window is what makes the column
        # comparable at all rather than blank
        "weather_intent": (getattr(understanding, "weather_intent", "")
                           or weather_intent_for(
                               (understanding.times_normalized or [None])[0]).value),
        "activity": understanding.activity, "sub_activity": understanding.sub_activity,
        "variables": understanding.variables, "aggregation": understanding.aggregation,
        "locations": understanding.locations, "times": understanding.times,
        "times_normalized": understanding.times_normalized,
        "entities": understanding.entities, "family": understanding.family,
        "confidence": round(understanding.confidence, 3),
    }
    try:
        async with sources.client() as http:
            answer = await run(http, understanding)
    except Exception as exc:                                        # noqa: BLE001
        return {"version": version, "ok": True, "latency_ms": nlu_ms, "usage": usage, **slots,
                "answer": None,
                "pipeline": {"ok": False, "failed_at": "pipeline",
                             "error": f"{type(exc).__name__}: {exc}", "stages": {}}}

    return {"version": version, "ok": True, "latency_ms": nlu_ms, "usage": usage, **slots,
            # a turn that never reached the weather API has stages but no answer to render
            "answer": answer.payload(understanding) if answer.answered else None,
            "pipeline": {"ok": answer.ok, "total_ms": answer.total_ms,
                         "failed_at": answer.failed_at, "error": answer.error,
                         "short_circuit": answer.short_circuit, "reply": answer.reply,
                         "needs_location": answer.needs_location,
                         "stopped_by_plan": answer.stopped_by, "summary": answer.summary,
                         "stages": answer.stages}}


def disagreements(columns: list[dict]) -> list[str]:
    """Which slots the answering models read differently. Order-insensitive for the lists."""
    answered = [c for c in columns if c.get("ok")]
    if len(answered) < 2:
        return []
    out = [f for f in SCALAR_FIELDS if len({str(c.get(f) or "") for c in answered}) > 1]
    out += [f for f in LIST_FIELDS
            if len({frozenset(str(x).lower() for x in (c.get(f) or [])) for c in answered}) > 1]
    return out


async def columns(text: str):
    """Every column, streamed as it lands. The event stream `POST /api/compare` returns."""
    started = time.perf_counter()
    ms = lambda: int((time.perf_counter() - started) * 1000)
    cleaned = normalize_text(text)
    order = contenders()

    yield {"type": "compare_start", "text": text,
           "normalized": cleaned.normalized if cleaned.replacements else None,
           "models": order}

    done: list[dict] = []
    tasks = [asyncio.create_task(_column(entry["version"], cleaned.normalized))
             for entry in order]
    for finished in asyncio.as_completed(tasks):
        got = await finished
        done.append(got)
        yield {"type": "compare_result", **got, "elapsed_ms": ms()}

    found = disagreements(done)
    yield {"type": "compare_done", "disagreements": found,
           "agreed": not found and sum(1 for c in done if c.get("ok")) > 1,
           "total_ms": ms()}
