"use client";

import { AlertTriangle, Check, HelpCircle, ShieldAlert, X } from "lucide-react";
import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { Advice } from "@/lib/v2/types";

/**
 * The decision, and every number it was read off.
 *
 * The evidence block is not a debug panel - it is the point. v2's claim is that a weather
 * recommendation should be auditable, so the policy that decided, the values it read and the
 * things it structurally could not see are all on screen next to the verdict.
 */

const LOOK: Record<Advice["verdict"], { icon: typeof Check; ring: string; chip: string }> = {
  YES: {
    icon: Check,
    ring: "border-emerald-500/30 bg-emerald-500/[0.06]",
    chip: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  },
  CAUTION: {
    icon: AlertTriangle,
    ring: "border-amber-500/30 bg-amber-500/[0.06]",
    chip: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  },
  NO: {
    icon: X,
    ring: "border-rose-500/30 bg-rose-500/[0.06]",
    chip: "bg-rose-500/15 text-rose-700 dark:text-rose-400",
  },
  UNKNOWN: {
    icon: HelpCircle,
    ring: "border-border bg-muted/40",
    chip: "bg-muted text-muted-foreground",
  },
};

/** Aggregate key -> how a person reads it. Anything unmapped is shown as-is rather than hidden. */
const LABELS: Record<string, [string, string]> = {
  rain_mm_total: ["Rain expected", "mm"],
  rain_mm_max: ["Heaviest hour", "mm"],
  rain_rows: ["Wet rows", ""],
  temp_max_c: ["Max temp", "°C"],
  temp_min_c: ["Min temp", "°C"],
  apparent_max_c: ["Feels like", "°C"],
  humidity_mean_pct: ["Humidity", "%"],
  wind_kmh: ["Wind", "km/h"],
  wind_gust_kmh: ["Gusts", "km/h"],
  cloud_pct: ["Low cloud", "%"],
  daylight_pct: ["Daylight in window", "%"],
  soil_moisture_mean: ["Soil moisture", ""],
  soil_temperature_mean_c: ["Soil temp", "°C"],
};
// Internal duplicates of a friendlier key, or plumbing. Not worth a tile.
const HIDE = new Set([
  "rows",
  "wind_ms_max",
  "wind_gust_ms_max",
  "cloud_frac_mean",
  "daylight_fraction",
]);

export function VerdictCard({ advice, policyId }: { advice: Advice; policyId: string | null }) {
  const [open, setOpen] = useState(false);
  const look = LOOK[advice.verdict];
  const Icon = look.icon;

  const tiles = Object.entries(advice.evidence)
    .filter(([key, value]) => !HIDE.has(key) && value !== null && value !== undefined)
    .map(([key, value]) => {
      const [label, unit] = LABELS[key] ?? [key.replace(/_/g, " "), ""];
      return { key, label, unit, value };
    });

  return (
    <div className={cn("rounded-2xl border p-4", look.ring)}>
      <div className="flex items-start gap-3">
        <span className={cn("mt-0.5 grid h-8 w-8 shrink-0 place-items-center rounded-full", look.chip)}>
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-base font-semibold leading-tight">{advice.headline}</span>
            <Badge variant="secondary" className={cn("text-[10px] font-medium", look.chip)}>
              {advice.verdict}
            </Badge>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {advice.activity ? `${advice.activity.replace(/_/g, " ")} · ` : ""}
            {advice.window}
          </p>

          {advice.reasons.length > 0 && (
            <ul className="mt-3 space-y-1.5">
              {advice.reasons.map((reason) => (
                <li key={reason} className="flex gap-2 text-sm">
                  <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-current opacity-40" />
                  <span className="min-w-0">{reason}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {tiles.length > 0 && (
        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
          {tiles.map((tile) => (
            <div key={tile.key} className="rounded-xl bg-background/60 px-2.5 py-2">
              <div className="truncate text-[10px] uppercase tracking-wide text-muted-foreground">
                {tile.label}
              </div>
              <div className="text-sm font-medium tabular-nums">
                {tile.value}
                {tile.unit && <span className="ml-0.5 text-[11px] font-normal text-muted-foreground">{tile.unit}</span>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* What the feed cannot see. Collapsed, but never absent - a quiet answer would imply
          lightning and rain probability had been checked. */}
      {advice.caveats.length > 0 && (
        <div className="mt-3">
          <button
            type="button"
            onClick={() => setOpen((on) => !on)}
            className="inline-flex items-center gap-1.5 text-[11px] text-muted-foreground hover:text-foreground"
          >
            <ShieldAlert className="h-3 w-3" />
            {open ? "Hide" : `What this cannot see (${advice.caveats.length})`}
          </button>
          {open && (
            <ul className="mt-2 space-y-1.5 border-l-2 border-border pl-3">
              {advice.caveats.map((caveat) => (
                <li key={caveat} className="text-[11px] leading-relaxed text-muted-foreground">
                  {caveat}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {policyId && (
        <p className="mt-3 font-mono text-[10px] text-muted-foreground">policy {policyId}</p>
      )}
    </div>
  );
}
