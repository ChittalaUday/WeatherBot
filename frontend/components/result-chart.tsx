"use client";

import { useMemo } from "react";
import { EvilAreaChart } from "@/components/evilcharts/charts/recharts-area-chart";
import { EvilBarChart } from "@/components/evilcharts/charts/recharts-bar-chart";
import { EvilComposedChart } from "@/components/evilcharts/charts/recharts-composed-chart";
import { EvilLineChart } from "@/components/evilcharts/charts/recharts-line-chart";
import { EvilRadarChart } from "@/components/evilcharts/charts/recharts-radar-chart";
import type { Chart, Point, Series } from "@/lib/types";

/**
 * The picture, drawn with EvilCharts (Recharts under it).
 *
 * The backend decides the *shape* - see `backend/pipeline/analysis.pick_chart` - and this
 * only renders it. Every shape here is interactive for free: hover reads the value, the
 * legend selects a series, and a long run of readings gets a brush to zoom with.
 *
 * Recharts wants rows and the pipeline emits series, so `toRows` pivots once at the top of
 * each renderer rather than each of them reshaping in its own way.
 */

const COLORS = ["var(--chart-1)", "var(--chart-2)", "var(--chart-3)"];
const BOX = "h-[210px] w-full";
// Past this many readings a series is unreadable without being able to zoom into it.
const BRUSH_AFTER = 24;

const stamp = (iso: string, hourly: boolean) => {
  const date = new Date(iso);
  return hourly
    ? `${date.getHours()}:00`
    : date.toLocaleDateString(undefined, { day: "numeric", month: "short" });
};

/** Series of points -> one row per timestamp, which is the shape Recharts reads. */
function toRows(series: Series[], hourly: boolean) {
  const byTime = new Map<string, Record<string, unknown>>();
  for (const one of series) {
    for (const point of one.points) {
      const row = byTime.get(point.t) ?? { t: point.t, when: stamp(point.t, hourly) };
      row[one.name] = point.v;
      byTime.set(point.t, row);
    }
  }
  return [...byTime.values()];
}

/** A colour per series, per theme. Not annotated as `ChartConfig`: the chart components
 *  check every config key against the data row type, and an index signature erases that. */
const paint = (color: string) => ({ light: [color], dark: [color] });

const configFor = (names: string[], unit: string) =>
  Object.fromEntries(
    names.map((name, index) => [
      name,
      { label: unit ? `${name} (${unit})` : name, colors: paint(COLORS[index % COLORS.length]) },
    ]),
  );

function Frame({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <figure className="mt-3 rounded-lg border bg-background/60 p-2">
      <figcaption className="px-1 pb-1 text-[11px] text-muted-foreground">{title}</figcaption>
      {children}
    </figure>
  );
}

const caption = (chart: Chart) => `${chart.label}${chart.unit ? ` (${chart.unit})` : ""}`;

// --- a series over time: line, filled line, or grouped bars ------------------

function SeriesChart({ chart, series }: { chart: Chart; series: Series[] }) {
  const hourly = chart.granularity === "hourly";
  const rows = useMemo(() => toRows(series, hourly), [series, hourly]);
  const names = useMemo(() => series.map((s) => s.name), [series]);
  const config = useMemo(() => configFor(names, chart.unit), [names, chart.unit]);
  const zoomable = rows.length > BRUSH_AFTER;

  if (chart.type === "bar") {
    return (
      <Frame title={caption(chart)}>
        <EvilBarChart config={config} data={rows} className={BOX} xDataKey="when">
          <EvilBarChart.Grid />
          <EvilBarChart.XAxis dataKey="when" />
          <EvilBarChart.YAxis />
          <EvilBarChart.Tooltip />
          {names.length > 1 && <EvilBarChart.Legend />}
          {names.map((name) => (
            <EvilBarChart.Bar key={name} dataKey={name} isClickable enableHoverHighlight />
          ))}
          {zoomable && <EvilBarChart.Brush />}
        </EvilBarChart>
      </Frame>
    );
  }

  if (chart.type === "area") {
    return (
      <Frame title={`${caption(chart)} · running total`}>
        <EvilAreaChart config={config} data={rows} className={BOX} xDataKey="when">
          <EvilAreaChart.Grid />
          <EvilAreaChart.XAxis dataKey="when" />
          <EvilAreaChart.YAxis />
          <EvilAreaChart.Tooltip />
          {names.length > 1 && <EvilAreaChart.Legend />}
          {names.map((name) => (
            <EvilAreaChart.Area key={name} dataKey={name} variant="gradient" isClickable />
          ))}
          {zoomable && <EvilAreaChart.Brush />}
        </EvilAreaChart>
      </Frame>
    );
  }

  return (
    <Frame title={caption(chart)}>
      <EvilLineChart config={config} data={rows} className={BOX} xDataKey="when">
        <EvilLineChart.Grid />
        <EvilLineChart.XAxis dataKey="when" />
        <EvilLineChart.YAxis />
        <EvilLineChart.Tooltip />
        {names.length > 1 && <EvilLineChart.Legend />}
        {names.map((name) => (
          <EvilLineChart.Line key={name} dataKey={name} isClickable>
            <EvilLineChart.ActiveDot />
          </EvilLineChart.Line>
        ))}
        {zoomable && <EvilLineChart.Brush />}
      </EvilLineChart>
    </Frame>
  );
}

// --- the day's low and high, as a floating column ---------------------------

function BandChart({ chart, points }: {
  chart: Chart;
  points: { t: string; lo: number; hi: number; v: number }[];
}) {
  const hourly = chart.granularity === "hourly";
  const rows = points.map((p) => ({
    when: stamp(p.t, hourly), range: [p.lo, p.hi] as [number, number], lo: p.lo, hi: p.hi,
  }));
  // A floating bar from the low to the high: the swing is the answer, and a single line
  // through the average hides the thing the question was usually about.
  const config = { range: { label: `Low to high (${chart.unit})`, colors: paint(COLORS[2]) } };

  return (
    <Frame title={`${caption(chart)} · daily low to high`}>
      <EvilBarChart config={config} data={rows} className={BOX} xDataKey="when" barRadius={4}>
        <EvilBarChart.Grid />
        <EvilBarChart.XAxis dataKey="when" />
        <EvilBarChart.YAxis domain={["dataMin - 2", "dataMax + 2"]}
                            tickFormatter={(value: number) => `${Math.round(value)}`} />
        <EvilBarChart.Tooltip />
        <EvilBarChart.Bar dataKey="range" variant="gradient" enableHoverHighlight />
      </EvilBarChart>
    </Frame>
  );
}

// --- rain as bars, temperature as a line over them --------------------------

function ComboChart({ chart, bars, line }: {
  chart: Chart;
  bars: { label: string; unit: string; points: Point[] };
  line: { label: string; unit: string; points: Point[] };
}) {
  const hourly = chart.granularity === "hourly";
  const warm = new Map(line.points.map((p) => [p.t, p.v]));
  const rows = bars.points.map((p) => ({
    when: stamp(p.t, hourly), rain: p.v, temp: warm.get(p.t) ?? null,
  }));
  const config = {
    rain: { label: `${bars.label} (${bars.unit})`, colors: paint(COLORS[0]) },
    temp: { label: `${line.label} (${line.unit})`, colors: paint(COLORS[2]) },
  };

  return (
    <Frame title={`${bars.label} and ${line.label.toLowerCase()}`}>
      <EvilComposedChart config={config} data={rows} className={BOX} xDataKey="when">
        <EvilComposedChart.Grid />
        <EvilComposedChart.XAxis dataKey="when" />
        {/* Two axes because two units: millimetres and degrees cannot share a scale. */}
        <EvilComposedChart.YAxis yAxisId="rain" />
        <EvilComposedChart.YAxis yAxisId="temp" orientation="right" />
        <EvilComposedChart.Tooltip />
        <EvilComposedChart.Legend />
        <EvilComposedChart.Bar dataKey="rain" barProps={{ yAxisId: "rain" }} />
        <EvilComposedChart.Line dataKey="temp" lineProps={{ yAxisId: "temp" }} connectNulls>
          <EvilComposedChart.ActiveDot />
        </EvilComposedChart.Line>
      </EvilComposedChart>
    </Frame>
  );
}

// --- wind direction as a rose -----------------------------------------------

function RoseChart({ buckets }: { buckets: { bucket: string; share: number }[] }) {
  const config = { share: { label: "Share of readings (%)", colors: paint(COLORS[0]) } };
  return (
    <Frame title="Wind direction · share of readings">
      <EvilRadarChart config={config} data={buckets} className={BOX}>
        <EvilRadarChart.PolarGrid />
        <EvilRadarChart.PolarAngleAxis dataKey="bucket" />
        <EvilRadarChart.Tooltip />
        <EvilRadarChart.Radar dataKey="share" variant="filled" isGlowing>
          <EvilRadarChart.ActiveDot />
        </EvilRadarChart.Radar>
      </EvilRadarChart>
    </Frame>
  );
}

// --- hour of day across days ------------------------------------------------

/**
 * Hand-drawn, because a grid of hours is not one of EvilCharts' shapes and a week of hourly
 * readings is 168 points - a smear as a line, and "it rains every afternoon" only shows up
 * as a grid. Hovering a cell reads it.
 */
function HeatmapChart({ chart, cells, days }: {
  chart: Chart;
  cells: { d: string; h: number; v: number }[];
  days: string[];
}) {
  const biggest = Math.max(...cells.map((c) => c.v)) || 1;
  return (
    <Frame title={`${caption(chart)} · by hour`}>
      <div className="grid gap-[2px] px-1 pb-1"
           style={{ gridTemplateColumns: `2.4rem repeat(${days.length}, minmax(0, 1fr))` }}>
        {Array.from({ length: 24 }, (_, hour) => (
          <div key={hour} className="contents">
            <span className="pr-1 text-right text-[9px] leading-[11px] text-muted-foreground">
              {hour % 6 === 0 ? `${hour}:00` : ""}
            </span>
            {days.map((day) => {
              const cell = cells.find((c) => c.d === day && c.h === hour);
              return (
                <div key={`${day}-${hour}`}
                     className="h-[11px] rounded-[2px] transition-opacity hover:opacity-100 hover:ring-1 hover:ring-primary"
                     style={{
                       background: COLORS[0],
                       opacity: cell ? 0.06 + (cell.v / biggest) * 0.84 : 0.03,
                     }}
                     title={cell
                       ? `${stamp(day, false)} ${hour}:00 · ${cell.v}${chart.unit}`
                       : `${stamp(day, false)} ${hour}:00 · no reading`} />
              );
            })}
          </div>
        ))}
        <span />
        {days.map((day) => (
          <span key={day} className="pt-1 text-center text-[9px] text-muted-foreground">
            {stamp(day, false)}
          </span>
        ))}
      </div>
    </Frame>
  );
}

export function ResultChart({ chart }: { chart: Chart }) {
  switch (chart.type) {
    case "band":
      return chart.points.length ? <BandChart chart={chart} points={chart.points} /> : null;
    case "combo":
      return <ComboChart chart={chart} bars={chart.bars} line={chart.line} />;
    case "rose":
      return chart.buckets.length ? <RoseChart buckets={chart.buckets} /> : null;
    case "heatmap":
      return chart.cells.length ? (
        <HeatmapChart chart={chart} cells={chart.cells} days={chart.days} />
      ) : null;
    default:
      return chart.series.length ? <SeriesChart chart={chart} series={chart.series} /> : null;
  }
}
