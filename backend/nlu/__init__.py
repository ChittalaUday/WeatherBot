"""
NLU - text in, an `Understanding` out, plus what the conversation remembers.

    understand(registry, text, version)           whichever model the switch picked
    registry.Registry.understand(text, version)   the trained classifier (v4)
    llm.understand(text, spec=...)                a prompted classifier, local or hosted
    context.apply(state, ...)                     how one turn changes the conversation

Everything downstream reads `Understanding` and nothing else, so a classifier can be
swapped, added or compared without the pipeline learning that more than one exists.
"""

from __future__ import annotations

import asyncio

from backend.nlu import llm  # after registry: llm reads Understanding from it
from backend.nlu.registry import (
    DEFAULT_VERSION,
    MODELS,
    NEVER_ASKS,
    Registry,
    Understanding,
)
from src.normalize import normalize as normalize_text

__all__ = ["DEFAULT_VERSION", "MODELS", "NEVER_ASKS", "Registry", "Understanding",
           "catalogue", "llm", "normalize_text", "understand"]


async def understand(registry: Registry, text: str, version: str | None = None) -> Understanding:
    """One turn, read by whichever model `version` names.

    The routing lives here rather than in the endpoint so the chat turn and the comparison
    cannot drift apart on which id means what.
    """
    spec = llm.SPECS.get(version or "")
    if spec is None:
        # sklearn, CPU-bound: off the event loop, so whatever is running concurrently with
        # this - Duckling, on a chat turn - actually makes progress while it runs.
        return await asyncio.to_thread(registry.understand, text, version)
    reading = llm.to_understanding(await llm.understand(text, spec=spec), text)
    if reading is None:
        # Loud on purpose. Falling back to the trained head would answer under the wrong
        # model's name, and a switch that silently does not switch is worse than an error.
        raise RuntimeError(f"{spec.name} could not read that turn")
    return reading


def catalogue(registry: Registry, local_ok: bool) -> list[dict]:
    """Every model a turn can be routed to, in switch order.

    One list, served by /api/models and /api/health both - two endpoints answering different
    lists of what this deployment serves is how a dropdown ends up offering a model the turn
    endpoint rejects.
    """
    return [*registry.available(), llm.entry(llm.LOCAL, local_ok)]
