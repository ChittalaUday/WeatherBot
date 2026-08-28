"use client";

import type { CompareColumn } from "@/lib/types";
import { ResultChart } from "@/components/result-chart";
import { ResultTable } from "@/components/result-table";

/**
 * Every classifier's reading of one sentence, side by side.
 *
 * The trained one is 46MB of TF-IDF and linear heads answering offline in milliseconds; a
 * prompted one is a round trip to a model that has never seen this label set and works only
 * from the schema in its prompt. The latency column is not decoration - it is half the
 * comparison.
 *
 * Fields they disagree on are highlighted, because agreement is the boring case and the
 * disagreements are the entire reason to look.
 */

const FIELDS: { key: keyof CompareColumn; label: string; list?: boolean }[] = [
  { key: "intent", label: "intent" },
  { key: "weather_intent", label: "window" },
  { key: "activity", label: "activity" },
  { key: "sub_activity", label: "sub-activity" },
  { key: "variables", label: "variables", list: true },
  { key: "aggregation", label: "aggregation" },
  { key: "locations", label: "locations", list: true },
  { key: "times", label: "times", list: true },
  { key: "times_normalized", label: "normalised", list: true },
];

function show(column: CompareColumn, key: keyof CompareColumn, list?: boolean) {
  const value = column[key];
  if (list) {
    const items = (value as string[] | undefined) ?? [];
    return items.length ? items.join(", ") : "—";
  }
  const text = (value as string | undefined) ?? "";
  return text && text !== "NONE" ? text.toLowerCase().replace(/_/g, " ") : "—";
}

const VERDICT_TONE: Record<string, string> = {
  YES: "text-emerald-600 dark:text-emerald-500",
  NO: "text-rose-600 dark:text-rose-500",
  CAUTION: "text-amber-600 dark:text-amber-500",
  UNKNOWN: "text-muted-foreground",
};

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 px-3 py-1">
      <span className="shrink-0 text-[11px] text-muted-foreground">{label}</span>
      <span className="min-w-0 text-right text-xs">{children}</span>
    </div>
  );
}

/**
 * What the reading above actually did: which source it routed to, what that cost, whether the
 * data came back usable, and what fell out. Two models can agree on every slot and still plan
 * different queries - and a model that reads the sentence wrong produces a fluent, confident,
 * useless answer, which only shows up here.
 */
function Pipeline({ pipeline }: { pipeline: CompareColumn["pipeline"] }) {
  if (!pipeline) return null;
  const s = pipeline.stages ?? {};

  if (pipeline.short_circuit) {
    return (
      <div className="border-t px-3 py-2">
        <div className="text-[11px] font-medium text-muted-foreground">pipeline</div>
        <p className="mt-1 text-xs">
          answered as <span className="font-medium">{pipeline.short_circuit}</span> — no weather
          call
        </p>
        {pipeline.reply && <p className="mt-1 text-xs italic text-muted-foreground">{pipeline.reply}</p>}
      </div>
    );
  }

  return (
    <div className="border-t">
      <div className="flex items-baseline justify-between px-3 pt-2">
        <span className="text-[11px] font-medium text-muted-foreground">pipeline</span>
        {typeof pipeline.total_ms === "number" && (
          <span className="text-[10px] tabular-nums text-muted-foreground">{pipeline.total_ms}ms</span>
        )}
      </div>

      {!pipeline.ok && (
        <p className="px-3 py-2 text-xs text-destructive">
          stopped at {pipeline.failed_at}: {pipeline.error}
        </p>
      )}

      {s.locations && (
        <Row label="places">
          {s.locations.resolved?.length
            ? s.locations.resolved.map((p) => p.name).join(", ")
            : (s.locations.note ?? "none resolved")}
          {s.locations.unresolved?.length ? (
            <span className="text-amber-600 dark:text-amber-500">
              {" "}· unknown: {s.locations.unresolved.join(", ")}
            </span>
          ) : null}
        </Row>
      )}
      {s.plan?.fields && (
        <Row label="columns">
          <span className="font-mono text-[11px]">{s.plan.fields.join(" ")}</span>
        </Row>
      )}
      {s.plan && (
        <Row label="source">
          {(s.plan.source ?? "—").replace(/_/g, " ").toLowerCase()}
          <span className="text-muted-foreground">
            {" "}· {(s.plan.resolution ?? "").toLowerCase()} · ~{s.plan.estimated_rows} rows
          </span>
        </Row>
      )}
      {s.plan?.window && <Row label="window">{s.plan.window}</Row>}
      {s.plan?.unservable?.length ? (
        <Row label="unservable">
          <span className="text-amber-600 dark:text-amber-500">{s.plan.unservable.join(", ")}</span>
        </Row>
      ) : null}
      {s.fetch && (
        <Row label="fetched">
          {s.fetch.ok === false ? (
            <span className="text-destructive">{s.fetch.error?.slice(0, 40)}</span>
          ) : (
            <>
              {(s.fetch.rows_returned ?? []).join(" + ")} rows
              <span className="text-muted-foreground"> · {s.fetch.ms}ms</span>
              {s.fetch.fell_back_from ? (
                <span className="text-amber-600 dark:text-amber-500"> · fell back</span>
              ) : null}
            </>
          )}
        </Row>
      )}
      {s.quality && (
        <Row label="data">
          <span
            className={
              s.quality.status === "OK" ? "" : "text-amber-600 dark:text-amber-500"
            }
          >
            {s.quality.status?.toLowerCase()}
          </span>
          {s.quality.gaps ? (
            <span className="text-muted-foreground"> · {s.quality.gaps} gaps</span>
          ) : null}
        </Row>
      )}
      {s.analysis?.reduced && (
        <Row label={s.analysis.reduced.kind.toLowerCase()}>
          <span className="tabular-nums">
            {s.analysis.reduced.value}
            {s.analysis.reduced.unit}
          </span>
        </Row>
      )}
      {s.advice && (
        <Row label="verdict">
          {s.advice.verdict ? (
            <span className={`font-semibold ${VERDICT_TONE[s.advice.verdict] ?? ""}`}>
              {s.advice.verdict}
            </span>
          ) : (
            <span className="text-muted-foreground">{s.advice.note}</span>
          )}
        </Row>
      )}
      {/* the sentence itself belongs to <Answer/> below - repeating it here printed it twice */}
      <div className="pb-1" />
    </div>
  );
}

/** The chat answer itself - same payload, same components as a single-chat turn. */
function Answer({ answer }: { answer: NonNullable<CompareColumn["answer"]> }) {
  return (
    <div className="border-t px-3 py-2">
      <p className="text-xs font-medium leading-snug">{answer.summary}</p>

      {answer.reduced && (
        <p className="mt-1 text-lg font-semibold tabular-nums">
          {answer.reduced.value}
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            {answer.reduced.unit} · {answer.reduced.kind.toLowerCase()}
          </span>
        </p>
      )}
      {answer.chart && (
        <div className="mt-2">
          <ResultChart chart={answer.chart} />
        </div>
      )}
      {answer.table?.rows?.length > 0 && (
        <div className="mt-2 max-h-64 overflow-auto">
          <ResultTable data={answer.table} />
        </div>
      )}
      {answer.insights?.length > 0 && (
        <ul className="mt-2 space-y-0.5">
          {answer.insights.map((insight) => (
            <li
              key={`${insight.kind}-${insight.text}`}
              className="flex gap-1.5 text-[11px] text-muted-foreground"
            >
              <span className="text-primary">•</span>
              {insight.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function Compare({
  text,
  models,
  disagreements,
  totalMs,
  pending = 0,
}: {
  text: string;
  models: CompareColumn[];
  disagreements: string[];
  totalMs: number;
  pending?: number;
}) {
  const disputed = new Set(disagreements);
  const answered = models.filter((m) => m.ok);
  const fastest = answered.length
    ? Math.min(...answered.map((m) => m.latency_ms))
    : 0;

  return (
    <section className="w-full">
      <div className="mb-3 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 className="text-sm font-semibold">{text}</h2>
        <span className="text-xs text-muted-foreground">{totalMs}ms total</span>
        {pending > 0 ? (
          <span className="rounded bg-muted px-1.5 py-0.5 text-[11px] text-muted-foreground">
            {pending} still running
          </span>
        ) : disagreements.length > 0 ? (
          <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[11px] font-medium text-amber-600 dark:text-amber-500">
            disagree on {disagreements.join(", ")}
          </span>
        ) : (
          <span className="rounded bg-emerald-500/10 px-1.5 py-0.5 text-[11px] font-medium text-emerald-600 dark:text-emerald-500">
            all three agree
          </span>
        )}
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        {models.map((column) => (
          <div key={column.version} className="min-w-0 rounded-xl border bg-card">
            <header className="flex items-center justify-between gap-2 border-b px-3 py-2">
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold">{column.name}</div>
                <div className="truncate text-[11px] text-muted-foreground">
                  {column.kind === "hosted" ? column.provider ?? "hosted" : `${column.version} · on-device`}
                </div>
              </div>
              <div className="shrink-0 text-right">
                <div
                  className={`text-xs font-medium tabular-nums ${
                    column.ok && column.latency_ms === fastest
                      ? "text-emerald-600 dark:text-emerald-500"
                      : "text-muted-foreground"
                  }`}
                >
                  {column.latency_ms}ms
                </div>
                {column.usage?.total_tokens ? (
                  <div className="text-[10px] text-muted-foreground">
                    {column.usage.total_tokens} tok
                  </div>
                ) : null}
              </div>
            </header>

            {!column.ok && !column.error ? (
              <div className="flex items-center gap-2 px-3 py-4">
                <span className="h-2 w-2 animate-pulse rounded-full bg-muted-foreground" />
                <p className="text-xs text-muted-foreground">still thinking…</p>
              </div>
            ) : !column.ok ? (
              <div className="px-3 py-4">
                <p className="text-xs text-muted-foreground">
                  <span className="font-medium text-destructive">unavailable</span> — {column.error}
                </p>
              </div>
            ) : (
              <dl className="divide-y">
                {FIELDS.map((field) => {
                  const value = show(column, field.key, field.list);
                  const flagged = disputed.has(field.key as string) && value !== "—";
                  return (
                    <div
                      key={field.key as string}
                      className={`flex items-baseline justify-between gap-3 px-3 py-1.5 ${
                        flagged ? "bg-amber-500/5" : ""
                      }`}
                    >
                      <dt className="shrink-0 text-[11px] text-muted-foreground">{field.label}</dt>
                      <dd
                        className={`min-w-0 truncate text-right text-xs ${
                          flagged ? "font-semibold text-amber-700 dark:text-amber-400" : ""
                        }`}
                        title={value}
                      >
                        {value}
                      </dd>
                    </div>
                  );
                })}
                {column.entities && Object.keys(column.entities).length > 0 && (
                  <div className="px-3 py-1.5">
                    <div className="text-[11px] text-muted-foreground">entities</div>
                    <div className="mt-1 flex flex-wrap gap-1">
                      {Object.entries(column.entities).flatMap(([kind, terms]) =>
                        terms.map((term) => (
                          <span
                            key={`${kind}-${term}`}
                            className="rounded bg-muted px-1.5 py-0.5 text-[10px]"
                          >
                            {kind}: {term}
                          </span>
                        )),
                      )}
                    </div>
                  </div>
                )}
                {typeof column.confidence === "number" && (
                  <div className="flex items-baseline justify-between gap-3 px-3 py-1.5">
                    <dt className="text-[11px] text-muted-foreground">confidence</dt>
                    <dd className="text-xs tabular-nums">
                      {Math.round(column.confidence * 100)}%
                    </dd>
                  </div>
                )}
              </dl>
            )}
            <Pipeline pipeline={column.pipeline} />
            {column.answer ? <Answer answer={column.answer} /> : null}
          </div>
        ))}
      </div>
    </section>
  );
}
