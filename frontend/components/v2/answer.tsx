"use client";

import { BarChart3, Clock, Database, MapPin, ShieldAlert, Sparkles, Table2, Timer, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { ResultChart } from "@/components/result-chart";
import { ResultTable } from "@/components/result-table";
import { Badge } from "@/components/ui/badge";
import { VerdictCard } from "@/components/v2/verdict-card";
import type { Result } from "@/lib/v2/types";

/**
 * One answered turn.
 *
 * The table and chart are the v1 components: v2 emits the same forecast payload shape on
 * purpose, so those render it unchanged and there is one implementation of a forecast table in
 * this app rather than two that drift.
 */

/** Sub-second answers read better in milliseconds; anything slower reads better in seconds. */
export function formatMs(ms: number): string {
  return ms >= 1000 ? `${(ms / 1000).toFixed(ms >= 10_000 ? 0 : 1)} s` : `${ms} ms`;
}

const unit = (result: Result) => {
  const many = result.table.rows.length !== 1;
  if (result.granularity === "daily") return many ? "days" : "day";
  return many ? "hours" : "hour";
};

const QUALITY: Record<Result["quality"]["status"], string> = {
  OK: "",
  PARTIAL: "text-amber-600 dark:text-amber-400",
  SPARSE: "text-amber-600 dark:text-amber-400",
  NO_DATA: "text-rose-600 dark:text-rose-400",
};

export function Answer({ result }: { result: Result }) {
  // The backend decided what this answer needs; these start where it said and stay togglable.
  const view = result.presentation;
  const [showTable, setShowTable] = useState(view?.table === "open");
  const [showChart, setShowChart] = useState(view?.chart === "open");
  const [showTiming, setShowTiming] = useState(false);
  const [showCaveats, setShowCaveats] = useState(false);
  const place = result.places[0];
  const metrics = result.metrics ?? { total_ms: 0 };
  // Stage breakdown, in the order a turn actually runs. Names match what the backend emits.
  const stages = (
    [
      ["read the question", metrics.parse_ms],
      ["find the place", metrics.locate_ms],
      ["fetch the weather", metrics.fetch_ms],
      ["check daylight", metrics.daylight_ms],
      ["apply the policy", metrics.policy_ms],
    ] as [string, number | undefined][]
  ).filter(([, value]) => typeof value === "number" && value > 0);

  return (
    <div className="space-y-3">
      {result.advice ? (
        <VerdictCard advice={result.advice} policyId={result.policy_id} />
      ) : (
        <div className="rounded-2xl border bg-card p-4">
          <p className="text-sm leading-relaxed">{result.summary}</p>
        </div>
      )}

      {/* For a decision turn the sentence sits under the card: the card is the answer, the
          sentence is how it would be said out loud. */}
      {result.advice && (
        <p className="px-1 text-sm leading-relaxed text-muted-foreground">{result.summary}</p>
      )}

      {result.assumed.length > 0 && (
        <p className="px-1 text-[11px] text-muted-foreground">
          Assumed: {result.assumed.join("; ")}
        </p>
      )}

      {result.insights.length > 0 && (
        <ul className="space-y-1 px-1">
          {result.insights.map((insight) => (
            <li key={insight.text} className="flex gap-2 text-xs text-muted-foreground">
              <span className="mt-[6px] h-1 w-1 shrink-0 rounded-full bg-current opacity-40" />
              {insight.text}
            </li>
          ))}
        </ul>
      )}

      {/* A chart when the shape is the point, a table when the values are. Both stay one
          click away when they are not - "only what is needed" is a default, not a restriction. */}
      {result.chart && showChart && (
        <div className="rounded-2xl border bg-card p-3">
          <ResultChart chart={result.chart} />
        </div>
      )}

      {/* Provenance strip: which place, which feed, at what resolution, how fresh, how long. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-1 text-[11px] text-muted-foreground">
        {place && (
          <span className="inline-flex items-center gap-1">
            <MapPin className="h-3 w-3" />
            {place.name}
            {place.district && place.district !== place.name ? `, ${place.district}` : ""}
          </span>
        )}
        <span className="inline-flex items-center gap-1">
          <Clock className="h-3 w-3" />
          {result.when}
        </span>
        <span className="inline-flex items-center gap-1">
          <Database className="h-3 w-3" />
          {result.plan.served_by} · {result.plan.resolution} · {result.plan.rows}{" "}
          {result.plan.rows === 1 ? "row" : "rows"}
        </span>
        {result.plan.fell_back_from && (
          <span className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400">
            <TriangleAlert className="h-3 w-3" />
            hourly only reaches 24 h, served daily
          </span>
        )}
        {result.quality.status !== "OK" && (
          <span className={QUALITY[result.quality.status]}>
            {result.quality.status.toLowerCase()}
            {result.quality.message ? ` · ${result.quality.message}` : ""}
          </span>
        )}
        <Badge variant="outline" className="text-[10px] font-normal">
          {result.model}
        </Badge>
        {/* Whether a model wrote this wording or the deterministic template did. A silent
            fallback would be a lie by omission, so it says which. */}
        <span
          className="inline-flex items-center gap-1"
          title={
            result.generation_note
              ? `Note: ${result.generation_note}`
              : result.generated
                ? "Worded by the model from the measured values"
                : "Worded by the deterministic template"
          }
        >
          <Sparkles className="h-3 w-3" />
          {result.generated ? "model wording" : "template wording"}
          {result.generation_note && <span className="text-amber-600 dark:text-amber-400">·</span>}
        </span>
        {result.confidence > 0 && (
          <span className="tabular-nums">confidence {result.confidence}</span>
        )}
        <button
          type="button"
          onClick={() => setShowTiming((on) => !on)}
          title="Where this turn's time went"
          className="inline-flex items-center gap-1 rounded-full border px-1.5 py-0.5 tabular-nums transition-colors hover:text-foreground"
        >
          <Timer className="h-3 w-3" />
          {formatMs(metrics.total_ms)}
        </button>
      </div>

      {showTiming && (
        <div className="flex flex-wrap gap-x-4 gap-y-1 rounded-2xl border bg-card px-3 py-2 text-[11px] text-muted-foreground">
          {stages.length === 0 ? (
            <span>No stage breakdown for this turn.</span>
          ) : (
            stages.map(([label, value]) => (
              <span key={label} className="tabular-nums">
                {label} <span className="font-medium text-foreground">{value} ms</span>
              </span>
            ))
          )}
          <span className="tabular-nums">
            total <span className="font-medium text-foreground">{metrics.total_ms} ms</span>
          </span>
        </div>
      )}

      {/* Toggles, plus the caveats for answers that have no verdict card to carry them. */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 px-1">
        {result.chart && (
          <button
            type="button"
            onClick={() => setShowChart((on) => !on)}
            className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <BarChart3 className="h-3 w-3" />
            {showChart ? "Hide the graph" : "Show the graph"}
          </button>
        )}
        {result.table.rows.length > 0 && (
          <button
            type="button"
            onClick={() => setShowTable((on) => !on)}
            className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <Table2 className="h-3 w-3" />
            {showTable
              ? "Hide the values"
              : `${result.table.rows.length} ${unit(result)} of values`}
          </button>
        )}
        {!result.advice && result.caveats.length > 0 && (
          <button
            type="button"
            onClick={() => setShowCaveats((on) => !on)}
            className="inline-flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <ShieldAlert className="h-3 w-3" />
            {showCaveats ? "Hide" : `What this cannot see (${result.caveats.length})`}
          </button>
        )}
      </div>

      {!result.advice && showCaveats && (
        <ul className="space-y-1.5 border-l-2 border-border pl-3">
          {result.caveats.map((caveat) => (
            <li key={caveat} className="text-[11px] leading-relaxed text-muted-foreground">
              {caveat}
            </li>
          ))}
        </ul>
      )}

      {showTable && (
        <div className="overflow-x-auto rounded-2xl border bg-card">
          <ResultTable data={result.table} />
        </div>
      )}
    </div>
  );
}
