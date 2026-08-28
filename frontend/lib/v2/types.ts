// Wire format of Backend-v2. The forecast-shaped parts (table, chart, insights) are identical
// to v1's, so those types are imported rather than restated - if they drift, this fails to
// compile, which is the point.

import type { Chart, Insight, Place, Reduced, TableData } from "@/lib/types";

/**
 * Where a v2 turn's time went. Its own type, not v1's: the stages are different, and reusing
 * v1's key names (`nlu_ms`, `solr_ms`, `api_ms`) rendered an empty breakdown because nothing
 * on the wire is called that.
 */
export type Metrics = {
  parse_ms?: number;      // normalization, patterns, entities - or the Rasa round trip
  locate_ms?: number;     // the location index
  fetch_ms?: number;      // the weather feed, or the archive
  daylight_ms?: number;   // sunrise/sunset, only when a policy needs light
  policy_ms?: number;     // aggregate + evaluate
  total_ms: number;
};

export type Verdict = "YES" | "NO" | "CAUTION" | "UNKNOWN";
export type Parser = "rules" | "rasa" | "llm";

/** The decision, and every number it was read off. v2's whole reason for existing. */
export type Advice = {
  verdict: Verdict;
  headline: string;
  reasons: string[];
  evidence: Record<string, number>;
  activity: string;
  /** the versioned policy that produced this - `cricket_outdoor_v1` */
  sub_activity: string;
  window: string;
  caveats: string[];
};

/** Which feed answered, at what resolution, and what it structurally cannot serve. */
export type PlanInfo = {
  verdict: string;
  source: string | null;
  served_by: string;
  fell_back_from: string;
  unservable: string[];
  resolution: string | null;
  rows: number;
  notes: string[];
};

export type Quality = {
  status: "OK" | "PARTIAL" | "SPARSE" | "NO_DATA";
  rows: number;
  coverage: Record<string, number>;
  unusable: string[];
  message: string;
};

export type Nlu = {
  model: string;
  intent: string;
  variables: string[];
  entities: { location: string[]; time: string[]; time_normalized: string[] };
  confidence: number;
};

export type Candidate = {
  name: string;
  level: string;
  district: string | null;
  state: string | null;
  lat: number;
  lon: number;
};

/** What the backend decided this answer needs on screen. Computed there so it is decided once. */
export type Presentation = {
  detail: "brief" | "table" | "chart";
  chart: "open" | "available" | "none";
  table: "open" | "available" | "none";
  rows: number;
  columns: number;
  why: { chart: string; table: string };
};

export type Result = {
  type: "result";
  turn_id: number;
  model: string;
  intent: string;
  variables: string[];
  when: string;
  places: Place[];
  granularity: "hourly" | "daily";
  summary: string;
  uncertain: boolean;
  confidence: number;
  reduced: Reduced | null;
  chart: Chart | null;
  insights: Insight[];
  /** slots carried over from the previous turn, so the answer can admit what it assumed */
  assumed: string[];
  table: TableData;
  metrics: Metrics;
  advice: Advice | null;
  plan: PlanInfo;
  quality: Quality;
  caveats: string[];
  policy_id: string | null;
  forecast_updated_at: string;
  presentation: Presentation;
  /** Did a model word this reply, or did it fall back to the deterministic sentence? */
  generated: boolean;
  /** Why it fell back, when it did - shown, because a silent fallback is a lie by omission. */
  generation_note: string;
  /** The template sentence, always computed. Kept so the two can be compared. */
  deterministic_summary?: string;
};

export type CompareColumn = {
  version: Parser;
  name: string;
  ok: boolean;
  error?: string;
  latency_ms: number;
  intent?: string;
  activity?: string;
  sub_activity?: string;
  locations?: string[];
  times?: string[];
  confidence?: number;
  pipeline?: {
    ok: boolean;
    total_ms?: number;
    short_circuit?: string | null;
    reply?: string | null;
    summary?: string | null;
    stages?: {
      locations?: { resolved?: { name: string; state?: string | null }[]; unresolved?: string[] };
      plan?: { source?: string | null; resolution?: string | null; estimated_rows?: number };
      advice?: { verdict?: Verdict; headline?: string; caveats?: string[] } | null;
    };
  };
};

export type ServerEvent =
  | { type: "status"; stage: "understanding" | "locating" | "fetching" | "writing" }
  // the model's reasoning, on its own channel so it is never shown as the answer
  | { type: "thinking"; text: string }
  | { type: "thinking_done" }
  // the answer as the model writes it; the `result` below repeats it in full
  | { type: "delta"; text: string }
  // the streamed wording failed verification - drop what was shown, the result carries the truth
  | { type: "delta_reset"; reason: string }
  | ({ type: "nlu" } & Nlu)
  | Result
  | {
      type: "chat";
      turn_id: number;
      chat_id: string;
      model: string;
      intent: string;
      family: "conversational" | "control" | "declined";
      message: string;
      confidence: number;
      metrics?: Metrics;
    }
  | {
      type: "clarify";
      message: string;
      text: string;
      reason?: string;
      candidates?: Candidate[];
    }
  | { type: "need_location"; reason: string; text: string; message: string }
  | { type: "compare_start"; text: string; models: { version: Parser; name: string }[] }
  | ({ type: "compare_result"; elapsed_ms: number } & CompareColumn)
  | { type: "compare_done"; disagreements: string[]; agreed: boolean; total_ms: number }
  | { type: "error"; message: string };

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "status"; stage: string }
  | { id: string; role: "thinking"; text: string; done: boolean }
  | { id: string; role: "streaming"; text: string }
  | { id: string; role: "nlu"; nlu: Nlu }
  | { id: string; role: "answer"; result: Result }
  | {
      id: string;
      role: "chat";
      turnId: number;
      family: "conversational" | "control" | "declined";
      message: string;
    }
  | {
      id: string;
      role: "clarify";
      message: string;
      text: string;
      reason?: string;
      candidates: Candidate[];
    }
  | { id: string; role: "ask-location"; text: string; message: string; answered: boolean }
  | {
      id: string;
      role: "compare";
      text: string;
      columns: CompareColumn[];
      disagreements: string[];
      agreed: boolean;
      pending: number;
      totalMs: number;
    }
  | { id: string; role: "error"; message: string };
