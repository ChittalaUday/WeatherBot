// Wire format of backend/api. Kept in one place so the transport and the UI agree.

export type Place = {
  query?: string;
  raw?: string;
  normalized?: string;
  name: string;
  level: string;
  lat: number;
  lon: number;
  district: string | null;
  state: string | null;
};

export type TableColumn = { key: string; label: string };
export type TableData = { columns: TableColumn[]; rows: Record<string, string>[] };

export type Chart = {
  type: "line" | "bar";
  field: string;
  label: string;
  unit: string;
  granularity: "hourly" | "daily";
  series: { name: string; points: { t: string; v: number }[] }[];
};

/** One observation, with the kind of observation it is - so the UI can group or filter them
 *  the same way the generation layer does. */
export type Insight = {
  kind: "RANGE" | "THRESHOLD" | "DRY_SPELL" | "COMPARISON";
  text: string;
  place: string;
};

export type Reduced = {
  kind: "SUM" | "AVG" | "MAX" | "MIN" | "TREND";
  value: number;
  unit: string;
  text: string;
  at?: string;
};

export type Nlu = {
  model?: string;
  variables?: string[];
  intent: string;
  action: "GET" | "COMPARE" | "ALERT";
  aggregation?: string;
  entities: { location: string[]; time: string[]; time_normalized: string[] };
  confidence: number;
};

/** Model 2 only: the decision, when to act on it, and the numbers it was read off. */
export type Advice = {
  verdict: "YES" | "NO" | "CAUTION" | "UNKNOWN";
  headline: string;
  reasons: string[];
  evidence: Record<string, string | number | null>;
  activity: string;
  sub_activity: string;
  /** The stretch the verdict points at - "06:00 to 12:00 (6 hours)". Empty when the answer
   *  is not about timing (soil state) or when there is no usable stretch at all. */
  window: string;
  caveats: string[];
};

/** Which source answered, at what resolution, and what it could not serve. */
export type PlanInfo = {
  verdict: "EXECUTE" | "COARSEN" | "ASK" | "REJECT";
  source: string | null;
  served_by: string;
  fell_back_from: string;
  unservable: string[];
  resolution: string | null;
  span_days: number;
  rows: number;
  reason: string;
  notes: string[];
};

/** What actually came back - every feed here can return rows missing most of their columns. */
export type Quality = {
  status: "OK" | "PARTIAL" | "SPARSE" | "NO_DATA";
  rows: number;
  coverage: Record<string, number>;
  unusable: string[];
  gaps: number;
  message: string;
};

/** Where a turn's wall clock went: the intent model, the location index, the weather feed,
 *  the local model that phrased the answer. Only `total_ms` is on every turn - a greeting
 *  never touches Solr, the feed or the LLM. */
export type Metrics = {
  nlu_ms?: number;
  solr_ms?: number;
  api_ms?: number;
  llm_ms?: number;
  db_ms?: number;
  total_ms: number;
};

/** One model's reading of a sentence, in the three-way comparison. */
export type CompareColumn = {
  version: string;
  name: string;
  kind: "local" | "hosted";
  provider?: string;
  ok: boolean;
  error?: string;
  latency_ms: number;
  intent?: string;
  weather_intent?: string;
  activity?: string;
  sub_activity?: string;
  variables?: string[];
  aggregation?: string;
  locations?: string[];
  times?: string[];
  times_normalized?: string[];
  entities?: Record<string, string[]>;
  family?: string;
  confidence?: number;
  usage?: { total_tokens?: number };
  /** The same payload a single chat turn produces - rendered with the same components. */
  answer?: Omit<Extract<ServerEvent, { type: "result" }>, "type" | "turn_id"> | null;
  /** What this reading actually produced downstream - source, cost, data, verdict, answer. */
  pipeline?: {
    ok: boolean;
    total_ms?: number;
    failed_at?: string;
    error?: string;
    short_circuit?: string;
    reply?: string;
    needs_location?: boolean;
    stopped_by_plan?: string;
    summary?: string;
    stages?: {
      routing?: { family?: string; note?: string };
      locations?: {
        ms?: number; asked_for?: string[]; unresolved?: string[]; note?: string;
        resolved?: { name: string; state?: string | null }[]
      };
      plan?: {
        verdict?: string; source?: string | null; resolution?: string | null;
        window?: string; span_days?: number; estimated_rows?: number;
        unservable?: string[]; reason?: string; fields?: string[]
      };
      fetch?: {
        ms?: number; served_by?: string; ok?: boolean; error?: string;
        fell_back_from?: string; note?: string; rows_returned?: number[]
      };
      quality?: {
        status?: string; rows?: number; gaps?: number; unusable?: string[];
        coverage?: Record<string, number>; message?: string
      };
      analysis?: {
        aggregation?: string; insights?: string[];
        reduced?: { kind: string; value: number; unit: string } | null
      };
      advice?: { verdict?: string; headline?: string; note?: string; caveats?: string[] };
      answer?: { summary?: string; when?: string; table_rows?: number; columns?: string[] };
    };
  };
};

export type ServerEvent =
  | { type: "status"; stage: "understanding" | "locating" | "fetching" | "writing"; places?: Place[] }
  | ({ type: "nlu" } & Nlu)
  | { type: "need_location"; reason: string; text: string; message: string }
  // the answer as the local model writes it; the `result` below repeats it in full
  | { type: "delta"; text: string }
  // the same model's reasoning, on its own channel so it is shown as thinking, not as answer
  | { type: "thinking"; text: string }
  | {
    type: "result";
    turn_id: number;
    model: string;
    variables: string[];
    intent: string;
    action: string;
    when: string;
    places: Place[];
    granularity: "hourly" | "daily";
    summary: string;
    uncertain: boolean;
    confidence: number;
    aggregation: string;
    reduced: Reduced | null;
    chart: Chart | null;
    insights: Insight[];
    // v3 only: what the model chose, and what it committed to instead of asking
    assumed?: string[];
    unresolved: string[];
    table: TableData;
    series: { place: string; points: { t: string; v: number | null }[] }[];
    metrics?: Metrics;
    // Model 2 only
    advice?: Advice | null;
    plan?: PlanInfo;
    quality?: Quality;
  }
  | {
    // a turn answered without touching the weather API: greeting, control, or declined
    type: "chat";
    turn_id: number;
    chat_id: string;
    model: string;
    intent: string;
    family: "conversational" | "control" | "declined";
    message: string;
    confidence: number;
    locations: string[];
    metrics?: Metrics;
  }
  | {
    // the query planner could not serve that question - too far ahead, too many rows, or an
    // archive that is not reachable. Not a confidence problem, so there is nothing to rate.
    type: "clarify";
    message: string;
    text: string;
  }
  | {
    // the columns, empty, so the UI can lay them out before any model has answered
    type: "compare_start";
    text: string;
    normalized: string | null;
    models: { version: string; name: string; kind: "local" | "hosted"; provider?: string }[];
  }
  | ({ type: "compare_result"; elapsed_ms: number } & CompareColumn)
  | { type: "compare_done"; disagreements: string[]; agreed: boolean; total_ms: number }
  | { type: "error"; message: string };

export type ChatMessage =
  | { id: string; role: "user"; text: string }
  | { id: string; role: "status"; stage: string }
  | { id: string; role: "nlu"; nlu: Nlu }
  // the answer mid-write; replaced by the "assistant" message when the result lands
  | { id: string; role: "streaming"; text: string }
  // the reasoning behind it, kept in the transcript and collapsed once the answer is in
  | { id: string; role: "thinking"; text: string }
  | { id: string; role: "assistant"; result: Extract<ServerEvent, { type: "result" }> }
  | { id: string; role: "ask-location"; text: string; message: string; answered: boolean }
  | {
    id: string;
    role: "clarify";
    message: string;
    text: string;
  }
  | {
    id: string;
    role: "chat";
    turnId: number;
    intent: string;
    family: "conversational" | "control" | "declined";
    message: string;
    locations: string[];
    metrics?: Metrics;
  }
  | {
    id: string;
    role: "compare";
    text: string;
    models: CompareColumn[];
    disagreements: string[];
    totalMs: number;
    pending: number;
  }
  | { id: string; role: "error"; message: string };
