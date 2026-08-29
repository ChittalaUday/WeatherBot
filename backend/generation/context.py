"""
Retrieval: what the model is allowed to know about this turn, and nothing else.

    python tests/test_generation_units.py          # the checks for this module

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

    def headings(self) -> list:
        """The section names this turn actually retrieved.

        `prompts.grounding` takes these so the model is only ever told what the sections it
        HAS mean. Told what a "Best window" is on a turn that has none, a small model produces
        one: a plain "will it rain tomorrow" came back "the best window for this rain is the
        27th of August".
        """
        return [heading for heading, lines in self.sections if lines]

    def mentions(self) -> str:
        """Everything retrieved, lowercased - what a reply is allowed to have got its facts
        from. `generation.llm` checks a no-data reply against this."""
        return self.render().lower()


def _table_lines(table: dict, max_rows: int, max_cells: int,
                 hourly: bool = False) -> list[str]:
    columns, body = table.get("columns") or [], table.get("rows") or []
    if not body or not columns:
        return []
    if len(body) <= max_rows and len(body) * len(columns) <= max_cells:
        return [" | ".join(c["label"] for c in columns)] + \
               [" | ".join(str(row.get(c["key"], "")) for c in columns) for row in body]
    return _digest_lines(columns, body, hourly)


def _digest_lines(columns: list[dict], body: list[dict], hourly: bool = False) -> list[str]:
    """Low, high and mean for every column - one line each, in the table's own order.

    A digest rather than a sample: fifteen of nineteen rows dropped is a summary that quietly
    describes four hours of a day, and the reader cannot tell which four.
    """
    lines = [f"{len(body)} readings, summarised per measurement:"]
    keys = {c["key"] for c in columns}
    skip = {"Tmin", "Tmax"} if {"Tmin", "Tmax", "Tavg"} <= keys and hourly else set()
    for column in columns[1:]:
        if column["key"] in skip:
            continue
        numbers = []
        for row in body:
            try:
                numbers.append(float(str(row.get(column["key"], "")).replace(",", "")))
            except (TypeError, ValueError):
                continue
        if not numbers:
            continue
        low, high = min(numbers), max(numbers)
        mean = sum(numbers) / len(numbers)
        lines.append(f"  {column['label']}: {low:g} to {high:g}, averaging {mean:.1f}"
                     if low != high else f"  {column['label']}: {low:g} throughout")
    return lines


def build(result: dict, *, max_rows: int = MAX_FACT_ROWS,
          max_cells: int = MAX_FACT_CELLS) -> Context:
    """The retrieval context for one finished pipeline result."""
    context = Context()

    def add(heading: str, lines) -> None:
        """A heading with nothing under it is noise the model has to read past."""
        if (kept := [line for line in lines if line]):
            context.sections.append((heading, kept))

    def day_lines() -> list[str]:
        """The day's own figures, from the daily feed an hourly answer also pulled.

        The hourly rows carry the shape of the day; these carry the things only a daily row
        has - the real high and low, sunshine hours, how long the day is.
        """
        rows = result.get("day_rows") or []
        out = []
        for place, day in zip(result.get("places") or [], rows):
            first = (day or [None])[0]
            if not isinstance(first, dict):
                continue
            said = [f"{label}: {first[key]:g}" for key, label in
                    (("Tmax", "high"), ("Tmin", "low"), ("SunSD", "sunshine hrs"),
                     ("DayLength", "day length hrs"), ("Rainfall", "rain mm"))
                    if isinstance(first.get(key), (int, float))]
            if said:
                out.append(f"{place['name']} - " + ", ".join(said))
        return out

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
        add("Decision", [advice.headline, *advice.reasons])
        from backend.pipeline.advice import TIMED
        if advice.window:
            add("Best window" if advice.activity in TIMED else "When", [advice.window])
        add("Caution", advice.caveats or [])

    if (table := result.get("table")):
        add("Figures", _table_lines(table, max_rows, max_cells, bool(result.get("hourly"))))
    add("Today", day_lines())
    return context
