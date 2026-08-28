"""
NLU - text in, an `Understanding` out, plus what the conversation remembers.

    registry.Registry.understand(text, version)   the trained model (v4)
    llm.understand(text)                          the hosted model (Model 3), same contract
    context.apply(state, ...)                     how one turn changes the conversation

Everything downstream reads `Understanding` and nothing else, so a model can be swapped,
added or compared without the pipeline learning that more than one exists.
"""

from backend.nlu.registry import (
    DEFAULT_VERSION,
    MODELS,
    NEVER_ASKS,
    Registry,
    Understanding,
)
from src.normalize import normalize as normalize_text

__all__ = ["DEFAULT_VERSION", "MODELS", "NEVER_ASKS", "Registry", "Understanding",
           "normalize_text"]
