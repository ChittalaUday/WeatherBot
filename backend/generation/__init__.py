"""
Generation - the deterministic answer, said the way a person would say it.

    context.build(result) -> Context      what the model is allowed to know
    prompts.system/user                   the instructions, assembled per turn
    llm.stream / llm.say / llm.explain    the local model, and the fallbacks
    llm.probe()                           is it actually reachable

Retrieval-augmented in the literal sense: the model states nothing it was not handed. What was
retrieved also decides what it is told - figures for a turn with figures, near names for a
turn with near names, and neither block for a turn with nothing to go on.

Every failure inside a turn is silent and falls back to the rule-built sentence, which is
correct but blunt. `probe()` exists so a *deployment* is not silent about it.
"""

from backend.generation.context import Context, build
from backend.generation.llm import (
                                    TROUBLE_LINES,
                                    explain,
                                    grounded,
                                    probe,
                                    say,
                                    stream,
                                    trouble_line,
                                    usable,
)

__all__ = ["Context", "build", "explain", "grounded", "probe", "say", "stream",
           "trouble_line", "usable", "TROUBLE_LINES"]
