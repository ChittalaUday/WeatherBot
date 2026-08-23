"""
Retrieval: what the model is allowed to know about this turn, and nothing else.

    python -m backend.generation.context        # self-check

The generation step is retrieval-augmented in the literal sense - it generates nothing from
its own weights except the wording. Everything it may state is retrieved here from what the
pipeline already computed, and assembled into labelled sections.

Why sections rather than one blob: the deterministic summary is one sentence about one thing,
so a model given only that sentence can only say that one thing back - "compare A and B" came
out with A's number and a shrug. But a model handed an unlabelled heap of numbers narrates the
heap. Labelled sections let the prompt say "answer from Comparison, then Range" and let a
0.6b keep them apart.

    Places        who this is about
    Period        when, and at what resolution
    Comparison    the answer to a two-place question - always first, when there is one
    Range         the shape of the series: total or average, and the spread
    Notable       thresholds crossed - heat, heavy rain, strong wind
    Dry spell     how much of the window was dry
    Decision      the advice verdict and the readings behind it
    Caution       what the data could not support
    Figures       the table itself, when it is small enough to be read

Two limits, because a table is too big in two different ways: too many time points to keep
straight (a week of days is followable, a day of hours is not), and too wide across places and
columns. Over either, the rows are dropped entirely rather than truncated - a small model
handed the first fortnight of a month will describe it as the month, which is a wrong answer
assembled from true numbers. The sections above already cover the whole window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ponytail: both limits are judgement calls, not measurements - 24 hourly rows demonstrably
# thinned the answer, 7 daily rows demonstrably helped it. Raise them with the model.
MAX_FACT_ROWS = 10
MAX_FACT_CELLS = 60

# Note.kind -> the heading it appears under. Order is the order it is read in, and the
# comparison leads because it is the answer to the question that produced it.
SECTIONS = (
    ("COMPARISON", "Comparison"),
    ("RANGE", "Range"),
    ("THRESHOLD", "Notable"),
    ("DRY_SPELL", "Dry spell"),
)


@dataclass
class Context:
    """The retrieved sections for one turn, in reading order."""

    sections: list = field(default_factory=list)      # [(heading, [lines])]

    def __bool__(self) -> bool:
        return bool(self.sections)

    def render(self) -> str:
        """The sections as text. Text, not JSON: a small model reads a table it can see the
        shape of, and copies braces it cannot."""
        out = []
        for heading, lines in self.sections:
            if not lines:
                continue
            out.append(f"{heading}:" if len(lines) > 1 else f"{heading}: {lines[0]}")
            if len(lines) > 1:
                out += [f"- {line}" for line in lines]
        return "\n".join(out)

    def mentions(self) -> str:
        """Everything retrieved, lowercased - what a reply is allowed to have got its facts
        from. `generation.llm` checks a no-data reply against this."""
        return self.render().lower()


def _table_lines(table: dict, max_rows: int, max_cells: int) -> list[str]:
    columns, body = table.get("columns") or [], table.get("rows") or []
    if not body or not columns:
        return []
    if len(body) <= max_rows and len(body) * len(columns) <= max_cells:
        return [" | ".join(c["label"] for c in columns)] + \
               [" | ".join(str(row.get(c["key"], "")) for c in columns) for row in body]
    return [f"({len(body)} rows of {', '.join(c['label'] for c in columns[1:])} - too many to "
            f"list; the figures above cover the whole period)"]


def build(result: dict, *, max_rows: int = MAX_FACT_ROWS,
          max_cells: int = MAX_FACT_CELLS) -> Context:
    """The retrieval context for one finished pipeline result."""
    context = Context()

    def add(heading: str, lines) -> None:
        """A heading with nothing under it is noise the model has to read past."""
        if (kept := [line for line in lines if line]):
            context.sections.append((heading, kept))

    if (places := result.get("places")):
        add("Places", [", ".join(p["name"] + (f" ({p['state']})" if p.get("state") else "")
                                 for p in places)])
    if result.get("when"):
        add("Period", [f"{result['when']} "
                       f"({'hourly' if result.get('hourly') else 'daily'} readings)"])

    if (reduced := result.get("reduced")):
        add("Figure asked for", [reduced["text"]])

    notes = result.get("insights") or []
    for kind, heading in SECTIONS:
        add(heading, [n.text for n in notes if getattr(n, "kind", None) == kind])

    if (advice := result.get("advice")):
        # The headline and the reasons, not `advice.evidence`. Its keys are machine names -
        # `peak_mm`, `wet_readings`, `in_window_readings` - and a small model handed them
        # writes them out: "with one wet reading from peak mm 1.9 and total mm 1.9". Every
        # figure in there is already in the conclusion or the sections above, said in words.
        add("Decision", [advice.headline, *advice.reasons])
        # The stretch the verdict points at, named on its own line. Without it the model has
        # a yes and no idea when, and "pick your moment" is the half of the answer that is
        # actually useful.
        add("Best window", [advice.window] if advice.window else [])
        add("Caution", advice.caveats or [])

    if (table := result.get("table")):
        add("Figures", _table_lines(table, max_rows, max_cells))
    return context


def demo():
    """Self-check: the sections keep their order, and a big table is summarised not truncated."""
    from backend.pipeline.analysis import Note

    def month(days):
        return {"places": [{"name": "Guntur", "state": "Andhra Pradesh"}],
                "when": "this week", "hourly": False,
                "insights": [Note("RANGE", "average rainfall 2.0mm, range 0.0-6.0mm"),
                             Note("THRESHOLD", "heavy rain on 1 of 7 days (peak 12.0mm)")],
                "table": {"columns": [{"key": "time", "label": "When"},
                                      {"key": "Rainfall", "label": "Rainfall (mm)"}],
                          "rows": [{"time": f"{d} Aug", "Rainfall": "2"}
                                   for d in range(1, days + 1)]}}

    week = build(month(7)).render()
    assert "Places: Guntur (Andhra Pradesh)" in week, week
    assert "Range: average rainfall" in week and "Notable: heavy rain" in week, week
    assert "1 Aug | 2" in week, week
    assert week.index("Range") < week.index("Notable") < week.index("Figures"), week

    # too much table: the rows go, the summary of them stays
    year = build(month(365)).render()
    assert "1 Aug | 2" not in year and "365 rows" in year, year
    assert "average rainfall" in year, "the summary must survive when the rows do not"
    assert "|" not in build(month(24)).render(), "a day of hourly rows is a large set"

    # a comparison leads, whatever order the notes arrived in
    compared = build({"places": [{"name": "A"}, {"name": "B"}], "when": "tomorrow",
                      "insights": [Note("RANGE", "A: average 2mm", "A"),
                                   Note("COMPARISON", "A leads B by 2mm on rainfall")]}).render()
    assert compared.index("Comparison") < compared.index("Range"), compared

    # an advice turn carries the verdict and what it was read off
    from backend.pipeline.advice import Advice
    decided = build({"places": [{"name": "Guntur"}], "when": "tomorrow",
                     "advice": Advice("YES", "Yes, but pick your moment.", [],
                                      {"mean_wind_ms": 6.0, "total_mm": 0.0},
                                      window="06:00 to 12:00 (6 hours)",
                                      caveats=["partial data: 80% coverage"])}).render()
    assert "Decision:" in decided and "pick your moment" in decided, decided
    assert "Best window: 06:00 to 12:00 (6 hours)" in decided, decided
    # the evidence dict's machine keys must never reach the prompt - the model reads them out
    assert "mean wind ms" not in decided and "mean_wind_ms" not in decided, decided
    assert "Caution:" in decided, decided
    # one line renders inline, several render as a list - so a heading never sits alone
    assert "Period: tomorrow" in decided, decided

    assert not build({}), "nothing retrieved is an empty context, not an empty heading list"
    print("context demo OK\n")
    print("\n".join("  " + line for line in week.splitlines()))


if __name__ == "__main__":
    demo()
