"use client";

import { Check, Minus, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { LoaderCircleIcon } from "@/components/ui/loader-circle-icon";
import { cn } from "@/lib/utils";
import type { CompareColumn } from "@/lib/v2/types";

/**
 * The rules cascade and Rasa, on one sentence, side by side.
 *
 * Location resolution, the forecast, the policy engine and the renderer are shared, so any
 * difference between these columns is a difference in how the sentence was read - which is the
 * only thing this view is for. The disagreement chips name the fields that actually differ.
 */

const VERDICT_CHIP: Record<string, string> = {
  YES: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  CAUTION: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  NO: "bg-rose-500/15 text-rose-700 dark:text-rose-400",
  UNKNOWN: "bg-muted text-muted-foreground",
};

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex gap-2 border-t px-3 py-1.5 text-xs first:border-t-0">
      <span className="w-24 shrink-0 text-[11px] text-muted-foreground">{label}</span>
      <span className="min-w-0 flex-1 break-words">{children}</span>
    </div>
  );
}

const dash = <span className="text-muted-foreground">—</span>;

export function ComparePanel({
  columns,
  disagreements,
  agreed,
  pending,
  totalMs,
}: {
  columns: CompareColumn[];
  disagreements: string[];
  agreed: boolean;
  pending: number;
  totalMs: number;
}) {
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 px-1">
        {pending > 0 ? (
          <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
            <LoaderCircleIcon size={13} isAnimated />
            reading with {pending} parser{pending === 1 ? "" : "s"}…
          </span>
        ) : agreed ? (
          <span className="inline-flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400">
            <Check className="h-3.5 w-3.5" /> both parsers read it the same way
          </span>
        ) : (
          <span className="inline-flex flex-wrap items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400">
            <X className="h-3.5 w-3.5" /> they disagree on
            {disagreements.map((field) => (
              <Badge key={field} variant="outline" className="text-[10px] font-normal">
                {field}
              </Badge>
            ))}
          </span>
        )}
        {totalMs > 0 && (
          <span className="text-[11px] tabular-nums text-muted-foreground">{totalMs} ms</span>
        )}
      </div>

      <div className="grid gap-2 sm:grid-cols-2">
        {columns.map((column) => {
          const advice = column.pipeline?.stages?.advice;
          const stopped = column.pipeline?.short_circuit;
          return (
            <div key={column.version} className="overflow-hidden rounded-2xl border bg-card">
              <div className="flex items-center justify-between gap-2 bg-muted/40 px-3 py-2">
                <span className="text-xs font-medium">{column.name}</span>
                <span className="flex items-center gap-1.5">
                  {column.latency_ms > 0 && (
                    <span className="text-[10px] tabular-nums text-muted-foreground">
                      {column.latency_ms} ms
                    </span>
                  )}
                  <Badge variant="outline" className="font-mono text-[10px] font-normal">
                    {column.version}
                  </Badge>
                </span>
              </div>

              {!column.ok ? (
                <p className="px-3 py-3 text-xs text-muted-foreground">
                  {column.error === "parser unavailable"
                    ? "Not running. Start it with: docker compose --profile rasa up -d"
                    : (column.error ?? "no reading")}
                </p>
              ) : (
                <div>
                  <Row label="task">
                    <span className="font-mono text-[11px]">{column.intent ?? "—"}</span>
                  </Row>
                  <Row label="subject">
                    {column.activity || column.sub_activity ? (
                      <span className="font-mono text-[11px]">
                        {column.activity || column.sub_activity}
                      </span>
                    ) : (
                      dash
                    )}
                  </Row>
                  <Row label="place">
                    {column.locations?.length ? (
                      <>
                        <span className="font-mono text-[11px]">{column.locations.join(", ")}</span>
                        {column.pipeline?.stages?.locations?.resolved?.[0] && (
                          <span className="text-muted-foreground">
                            {" → "}
                            {column.pipeline.stages.locations.resolved[0].name}
                          </span>
                        )}
                      </>
                    ) : (
                      <span className="text-amber-600 dark:text-amber-400">
                        none extracted
                      </span>
                    )}
                  </Row>
                  <Row label="when">
                    {column.times?.length ? (
                      <span className="font-mono text-[11px]">{column.times.join(", ")}</span>
                    ) : (
                      dash
                    )}
                  </Row>
                  <Row label="confidence">
                    <span className="tabular-nums">{column.confidence ?? dash}</span>
                  </Row>
                  <Row label="verdict">
                    {advice?.verdict ? (
                      <span
                        className={cn(
                          "rounded px-1.5 py-0.5 text-[10px] font-medium",
                          VERDICT_CHIP[advice.verdict],
                        )}
                      >
                        {advice.verdict}
                      </span>
                    ) : stopped ? (
                      <span className="inline-flex items-center gap-1 text-muted-foreground">
                        <Minus className="h-3 w-3" />
                        {stopped.replace(/_/g, " ")}
                      </span>
                    ) : (
                      dash
                    )}
                  </Row>
                  {(column.pipeline?.summary || column.pipeline?.reply) && (
                    <Row label="answer">
                      <span className="text-muted-foreground">
                        {column.pipeline.summary ?? column.pipeline.reply}
                      </span>
                    </Row>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
