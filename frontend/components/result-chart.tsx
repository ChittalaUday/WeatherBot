"use client";

import { useId } from "react";
import type { Chart } from "@/lib/types";

const COLORS = ["var(--chart-1, #2563eb)", "var(--chart-2, #16a34a)", "var(--chart-3, #ea580c)"];
const WIDTH = 560;
const HEIGHT = 170;
const PAD = { top: 12, right: 12, bottom: 26, left: 38 };

const stamp = (iso: string, hourly: boolean) => {
  const date = new Date(iso);
  return hourly
    ? `${date.getHours()}:00`
    : date.toLocaleDateString(undefined, { day: "numeric", month: "short" });
};

/**
 * Hand-rolled SVG rather than a charting dependency: two shapes are needed (a line over
 * time, grouped bars for a comparison) and both are a dozen lines of maths.
 */
export function ResultChart({ chart }: { chart: Chart }) {
  const gradientId = useId();
  const hourly = chart.granularity === "hourly";
  const values = chart.series.flatMap((s) => s.points.map((p) => p.v));
  if (values.length === 0) return null;

  const max = Math.max(...values);
  const min = Math.min(...values);
  const span = max - min || 1;
  const top = max + span * 0.15;
  const bottom = chart.field === "Rainfall" ? 0 : min - span * 0.15;
  const plotW = WIDTH - PAD.left - PAD.right;
  const plotH = HEIGHT - PAD.top - PAD.bottom;

  const x = (index: number, count: number) =>
    PAD.left + (count <= 1 ? plotW / 2 : (index / (count - 1)) * plotW);
  const y = (value: number) => PAD.top + plotH - ((value - bottom) / (top - bottom || 1)) * plotH;

  const labels = chart.series[0].points;
  const tickEvery = Math.max(1, Math.ceil(labels.length / 7));

  return (
    <figure className="mt-3 rounded-lg border bg-background/60 p-2">
      <figcaption className="flex items-center justify-between px-1 pb-1 text-[11px] text-muted-foreground">
        <span>
          {chart.label} ({chart.unit})
        </span>
        {chart.series.length > 1 && (
          <span className="flex gap-3">
            {chart.series.map((series, index) => (
              <span key={series.name} className="inline-flex items-center gap-1">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ background: COLORS[index % COLORS.length] }}
                />
                {series.name}
              </span>
            ))}
          </span>
        )}
      </figcaption>

      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="w-full" role="img"
           aria-label={`${chart.label} over time`}>
        <defs>
          <linearGradient id={gradientId} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor={COLORS[0]} stopOpacity="0.25" />
            <stop offset="100%" stopColor={COLORS[0]} stopOpacity="0" />
          </linearGradient>
        </defs>

        {[0, 0.5, 1].map((fraction) => {
          const value = bottom + (top - bottom) * (1 - fraction);
          return (
            <g key={fraction}>
              <line x1={PAD.left} x2={WIDTH - PAD.right} y1={y(value)} y2={y(value)}
                    stroke="currentColor" strokeOpacity="0.12" strokeWidth="1" />
              <text x={PAD.left - 6} y={y(value) + 3} textAnchor="end"
                    className="fill-muted-foreground text-[9px]">
                {value.toFixed(value >= 10 ? 0 : 1)}
              </text>
            </g>
          );
        })}

        {chart.type === "bar"
          ? chart.series.map((series, seriesIndex) =>
              series.points.map((point, index) => {
                const groupWidth = plotW / series.points.length;
                const barWidth = Math.max(3, (groupWidth * 0.6) / chart.series.length);
                const left =
                  PAD.left + index * groupWidth + groupWidth * 0.2 + seriesIndex * barWidth;
                return (
                  <rect key={`${series.name}-${point.t}`} x={left} y={y(point.v)}
                        width={barWidth} height={Math.max(1, y(bottom) - y(point.v))}
                        rx="2" fill={COLORS[seriesIndex % COLORS.length]} fillOpacity="0.85" />
                );
              }),
            )
          : chart.series.map((series, seriesIndex) => {
              const path = series.points
                .map((point, index) =>
                  `${index === 0 ? "M" : "L"} ${x(index, series.points.length)} ${y(point.v)}`)
                .join(" ");
              const area =
                `${path} L ${x(series.points.length - 1, series.points.length)} ${y(bottom)} ` +
                `L ${x(0, series.points.length)} ${y(bottom)} Z`;
              return (
                <g key={series.name}>
                  {chart.series.length === 1 && <path d={area} fill={`url(#${gradientId})`} />}
                  <path d={path} fill="none" strokeWidth="2" strokeLinejoin="round"
                        strokeLinecap="round" stroke={COLORS[seriesIndex % COLORS.length]} />
                  {series.points.map((point, index) => (
                    <circle key={point.t} cx={x(index, series.points.length)} cy={y(point.v)}
                            r="2.5" fill={COLORS[seriesIndex % COLORS.length]}>
                      <title>{`${stamp(point.t, hourly)} · ${point.v}${chart.unit}`}</title>
                    </circle>
                  ))}
                </g>
              );
            })}

        {labels.map((point, index) =>
          index % tickEvery === 0 ? (
            <text key={point.t} x={x(index, labels.length)} y={HEIGHT - 8} textAnchor="middle"
                  className="fill-muted-foreground text-[9px]">
              {stamp(point.t, hourly)}
            </text>
          ) : null,
        )}
      </svg>
    </figure>
  );
}
