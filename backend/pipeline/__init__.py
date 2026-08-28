"""
The answer path: one Understanding in, one Answer out. No transport, no conversation state.

    python tests/test_live_stack.py          # the checks for this module

    route -> places -> plan -> fetch -> columns -> quality -> analyse -> decide -> summarise

This is the *only* implementation. The chat endpoint passes the places the conversation already
resolved; the comparison view passes none and lets it resolve. Two copies of this sequence
drifted the moment one was fixed, and they did.

Nothing here raises: a stage that fails records why and the later stages are skipped, because
this runs several times side by side and one dead column must not take the others with it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime

from backend.nlu import times
from backend.pipeline import advice as advice_engine
from backend.pipeline import analysis, params, profiles, quality, render, sources
from backend.pipeline import places as place_index
from backend.pipeline import plan as planner
from backend.pipeline.timewindow import resolve as resolve_window
from backend.pipeline.timewindow import select_rows

__all__ = ["Answer", "run", "resolve_places"]


@dataclass
class Answer:
    """Everything one turn produced, and how it got there.

    `stages` is the audit trail - what each step decided and how long it took. It is what the
    comparison view renders and what a debugger reads; it is never the source of any value the
    UI shows, which all come from the named fields.
    """

    ok: bool = True
    stages: dict = field(default_factory=dict)
    total_ms: int = 0

    # how it ended, when it did not end in an answer
    failed_at: str = ""
    error: str = ""
    short_circuit: str = ""          # the intent family, for a turn that needs no weather
    reply: str = ""                  # the canned answer for that turn
    needs_location: bool = False
    stopped_by: str = ""             # a plan verdict of REJECT or ASK

    # the answer itself
    summary: str = ""
    places: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    fields: list = field(default_factory=list)
    when: str = ""
    aggregation: str = "RAW"
    hourly: bool = False
    plan: planner.QueryPlan | None = None
    quality: quality.Quality | None = None
    advice: advice_engine.Advice | None = None
    reduced: dict | None = None
    insights: list = field(default_factory=list)
    table: dict = field(default_factory=dict)
    chart: dict | None = None
    # what to open on screen. The rule fills it here; the chat endpoint overwrites it with
    # the model's pick when there is one. Never empty, so a client always has an answer.
    presentation: dict = field(default_factory=dict)
    served_by: str = ""
    fell_back_from: str = ""
    rows: list = field(default_factory=list)     # the selected rows, one list per place

    @property
    def answered(self) -> bool:
        """True only when there is a forecast to render - not a greeting, not a refusal."""
        return self.ok and self.plan is not None and not self.stopped_by

    def as_context(self) -> dict:
        """The shape `backend.generation.context.build` reads. One place defines it."""
        return {"places": self.places, "when": self.when, "hourly": self.hourly,
                "insights": self.insights, "table": self.table, "reduced": self.reduced,
                "advice": self.advice}

    def payload(self, understanding, *, variables=None) -> dict:
        """The wire body a client renders - identical for a chat turn and a compare column.

        The chat endpoint adds the conversation-specific fields on top (turn_id, chat_id,
        operation, metrics); everything else comes from here, so a compared column and a
        chatted answer are literally the same object shape.
        """
        plan, checked = self.plan, self.quality
        return {
            "model": understanding.version,
            "variables": variables if variables is not None else understanding.variables,
            "intent": understanding.intent,
            "action": understanding.action,
            "when": self.when,
            "places": self.places,
            "granularity": "hourly" if self.hourly else "daily",
            "summary": self.summary,
            "confidence": round(understanding.confidence, 3),
            "aggregation": self.aggregation,
            "reduced": self.reduced,
            "chart": self.chart,
            "insights": [note.as_dict() for note in self.insights],
            "unresolved": self.unresolved,
            "assumed": understanding.assumed,
            "advice": ({"verdict": self.advice.verdict, "headline": self.advice.headline,
                        "reasons": self.advice.reasons, "evidence": self.advice.evidence,
                        "activity": self.advice.activity,
                        "sub_activity": self.advice.sub_activity,
                        "window": self.advice.window,
                        "caveats": self.advice.caveats} if self.advice else None),
            "plan": {**(plan.as_dict() if plan else {}),
                     "served_by": self.served_by, "fell_back_from": self.fell_back_from},
            "quality": {"status": checked.status, "rows": checked.rows,
                        "coverage": {k: round(v, 2) for k, v in checked.coverage.items()},
                        "unusable": checked.unusable, "gaps": checked.gaps,
                        "message": checked.message} if checked else None,
            "table": self.table,
            "presentation": self.presentation or render.presentation(self),
            "series": [{"place": place["name"],
                        "points": [{"t": r["Date_time"], "v": r.get(self.fields[0])}
                                   for r in rows]}
                       for place, rows in zip(self.places, self.rows)] if self.fields else [],
            "stages": self.stages,
        }


async def resolve_places(http, names: list[str]) -> tuple[list[dict], list[str]]:
    """Raw location spans -> resolved places, plus the ones the index did not know.

    The one implementation. The chat endpoint used to carry a verbatim copy of this, so a fix
    to either was a fix to one of them.
    """
    usable = [n for n in names
              if not place_index.is_relative(n)]
    if not usable:
        return [], []
    solr = lambda query, rows=8: sources.solr_query(http, query, rows)
    # a comma span can be one address or two places - the resolver decides
    parts = [part for name in usable for part in place_index.split_span(name)]
    resolved = await asyncio.gather(*(place_index.resolve(solr, part) for part in parts))
    return [p for p in resolved if p], [n for n, p in zip(parts, resolved) if not p]


def served_fields(fields: list[str], selected: list[list[dict]]) -> tuple[list[str], list[str]]:
    """Split the asked-for columns into (kept, never sent), by what actually came back.

    Asking a source for what it does not serve cost twice: a table column of dashes, and a
    SPARSE verdict printing "not enough data" under an answer that had every number it needed.
    The capability table says what a source *claims*; this trusts the rows. When nothing came
    back at all, nothing is dropped - that is a data problem, and quality has to say so.
    """
    absent = [f for f in fields if not any(quality.values(rows, f) for rows in selected)]
    if not absent or len(absent) == len(fields):
        return fields, []
    return [f for f in fields if f not in absent], absent


async def run(http, understanding, *, places: list[dict] | None = None,
              aggregation: str | None = None, now: datetime | None = None) -> Answer:
    """Everything after the model, for one reading of one sentence."""
    started = time.perf_counter()
    now = now or datetime.now()
    answer = Answer()
    elapsed = lambda: int((time.perf_counter() - started) * 1000)

    def finish(**changes) -> Answer:
        for key, value in changes.items():
            setattr(answer, key, value)
        answer.total_ms = elapsed()
        return answer

    # 1. routing - does this turn want weather at all? A greeting that reaches the location
    # resolver comes back asking which city you meant, which is the worst thing this bot does.
    if not understanding.needs_weather:
        answer.stages["routing"] = {"family": understanding.family,
                                    "note": "answered without touching the weather API"}
        return finish(short_circuit=understanding.family, reply=understanding.reply)

    # 2. places - unless the caller already has them from conversation state
    began = time.perf_counter()
    unresolved: list[str] = []
    if places is None:
        try:
            places, unresolved = await resolve_places(http, understanding.locations)
        except Exception as exc:                                   # noqa: BLE001
            return finish(ok=False, failed_at="locations",
                          error=f"{type(exc).__name__}: {exc}")
    answer.places, answer.unresolved = places, unresolved
    answer.stages["locations"] = {
        "ms": int((time.perf_counter() - began) * 1000),
        "asked_for": understanding.locations,
        "resolved": [{"name": p["name"], "state": p.get("state"),
                      "lat": p["lat"], "lon": p["lon"]} for p in places],
        "unresolved": unresolved,
    }
    if not places:
        answer.stages["locations"]["note"] = "no place resolved - the caller must ask for one"
        return finish(needs_location=True)

    # 3. parameters - which API, how many places, which reduction, over which window, with
    # the reason for each decision beside it. The profile is chosen from the window, so the
    # window is resolved first; a turn that named no time gets one from the profile.
    #
    # A span the rule tables could not canonicalise gets one chance from a model, and if that
    # fails the turn stops rather than guessing. Every `time_resolution` correction in the
    # feedback table was an unplaceable expression answered with next week.
    said = understanding.times_normalized[0] if understanding.times_normalized else ""
    spoken, how = await times.place(said, understanding.text, now=now,
                                    hint=getattr(understanding, "time_hint", ""))
    if how != "rules":
        answer.stages["time"] = {"span": said or None, "canonical": spoken or None, "by": how}
    if how == "unplaceable":
        # Nothing can place it. Saying so is the answer - `generation.explain("time")` has
        # carried the line for this since before anything could reach it.
        return finish(ok=False, failed_at="time",
                      error=f"could not place the time in {understanding.text!r}")
    understanding.times_normalized = [spoken] if spoken else []
    window = resolve_window(spoken, now) if spoken else None
    profile = profiles.pick(understanding, window, now)
    chosen = params.resolve(understanding, profile, places, now=now, aggregation=aggregation)
    fields, plan = chosen.fields, chosen.plan
    normalized = chosen.window
    answer.aggregation = chosen.aggregation
    answer.plan = plan
    # what was assumed on their behalf is the answer's to admit, not the audit trail's to bury
    understanding.assumed.extend(chosen.assumed)
    answer.stages["params"] = chosen.as_dict()
    answer.stages["plan"] = {**(plan.as_dict() if plan else {}), "fields": fields}
    if plan and plan.verdict in (planner.Verdict.REJECT, planner.Verdict.ASK):
        return finish(stopped_by=plan.verdict.value)

    # 4. fetch
    began = time.perf_counter()
    fetched = await sources.fetch_for(http, plan, places)
    answer.served_by, answer.fell_back_from = fetched.source, fetched.fell_back_from
    answer.stages["fetch"] = {
        "ms": int((time.perf_counter() - began) * 1000),
        "served_by": fetched.source, "ok": fetched.ok, "error": fetched.error,
        "fell_back_from": fetched.fell_back_from, "note": fetched.note,
        "rows_returned": [len(rows) for rows in fetched.per_place],
    }
    if not fetched.ok:
        return finish(ok=False, failed_at="fetch", error=fetched.error)

    answer.hourly = plan.hourly if plan else False
    answer.rows = [select_rows(feed, normalized or "", now)[0] for feed in fetched.per_place]
    answer.when = (plan.label if plan else "") or select_rows(fetched.per_place[0], normalized or "", now)[1]

    # 5. columns - everything downstream reads `fields`, so the ones the feed never sent are
    # dropped here, once.
    fields, absent = served_fields(fields, answer.rows)
    answer.fields = fields
    if absent:
        answer.stages["fields_dropped"] = {"absent": absent, "kept": fields}

    # 6. quality - what actually came back, before anything is computed from it
    checked = quality.assess(answer.rows[0], fields,
                             expect_daily=0 if answer.hourly else len(answer.rows[0]))
    answer.quality = checked
    answer.stages["quality"] = {
        "status": checked.status, "rows": checked.rows, "gaps": checked.gaps,
        "coverage": {k: round(v, 2) for k, v in checked.coverage.items()},
        "unusable": checked.unusable, "message": checked.message,
    }

    # 7. analysis - the reduction, the observations
    answer.reduced = analysis.apply_aggregation(answer.rows[0], fields[0], answer.aggregation)
    answer.insights = analysis.build_insights(answer.rows, places, fields, answer.aggregation,
                                              answer.hourly)
    answer.stages["analysis"] = {
        "aggregation": answer.aggregation, "reduced": answer.reduced,
        "insights": [n.as_dict() for n in answer.insights],
        "sample": {f: quality.values(answer.rows[0], f)[:5] for f in fields[:3]},
    }

    # 8. the decision, if this was an advice turn
    # `hourly` matters: the same rule reads an hourly feed and a daily one, and only one of
    # them can honestly say "from 14:00"
    verdict = advice_engine.evaluate(understanding.activity, answer.rows[0],
                                     sub_activity=understanding.sub_activity,
                                     hourly=answer.hourly)
    answer.advice = verdict
    answer.stages["advice"] = ({"verdict": verdict.verdict, "headline": verdict.headline,
                               "reasons": verdict.reasons, "evidence": verdict.evidence,
                               "window": verdict.window, "caveats": verdict.caveats}
                              if verdict else {"note": "not an advice turn"})

    # 9. the conclusion. Order matters: the number that was asked for leads, then the
    # decision, then anything the data could not support.
    compare_two = understanding.action == "COMPARE" and len(places) > 1
    shown = places if compare_two else places[:1]
    parts = [render.summarize(understanding.action,
                              answer.rows if compare_two else answer.rows[0],
                              fields, shown, answer.when, answer.aggregation)]
    if answer.reduced:
        parts.insert(0, answer.reduced["text"])          # the figure that was asked for leads
    if verdict and verdict.verdict != advice_engine.UNKNOWN:
        parts.insert(0, verdict.headline)                # ...unless a decision was asked for
    elif verdict:
        # UNKNOWN: the rule could not run, so there are no figures to stand behind. Saying why
        # is the whole answer - the readings below it would read as one.
        parts = [verdict.headline, *verdict.reasons]
    parts += _caveats(checked, fetched, plan, absent, unresolved)
    answer.summary = _sentences(parts)

    answer.table = render.build_table(answer.rows if compare_two else answer.rows[0], fields,
                                      shown, answer.hourly)
    # Built whenever there is a series, and *shown* only when the presentation below says so.
    # Suppressing it here on a keyword left two places deciding what appears on screen.
    answer.chart = analysis.build_chart(answer.rows, places, fields[0], answer.hourly)
    # A reader who asked to *see* it has already decided; the model is not consulted about it.
    answer.presentation = render.presentation(
        answer, "chart" if answer.chart and analysis.wants_chart(understanding.text) else "",
        "asked to see it", "question")
    answer.stages["answer"] = {"summary": answer.summary, "when": answer.when,
                               "table_rows": len(answer.rows[0]), "columns": fields}
    return finish()


def _sentences(parts: list[str]) -> str:
    """Join fragments into readable prose, punctuating the ones that forgot to.

    Each piece is built by a different rule and only some end in a full stop; concatenated
    with a bare space they run together into one broken sentence.
    """
    out = []
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        # a trailing bracket is already punctuation of its own kind
        out.append(part if part[-1] in ".!?)" else part + ".")
    return " ".join(out)


def _caveats(checked, fetched, plan, absent: list[str], unresolved: list[str]) -> list[str]:
    """Everything the answer is owed but the conclusion could not carry, in reading order."""
    out = []
    if (caveat := quality.caveat(checked)):
        out.append(caveat)
    if fetched.fell_back_from:
        out.append(f"({fetched.note or 'served from a fallback source'}.)")
    if plan.unservable:
        out.append(f"(No {', '.join(plan.unservable)} in that archive.)")
    if absent:
        # a footnote, not a failure - said once, plainly
        out.append("(No " + ", ".join(render.label(f).lower() for f in absent) +
                   " in this feed.)")
    if unresolved:
        out.append(f"(Could not find {', '.join(unresolved)} - showing the rest.)")
    return out
