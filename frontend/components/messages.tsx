"use client";

import { Brain, HelpCircle, PenLine, Radar, Search, Sigma, Volume2, Square } from "lucide-react";
import { AlarmClockIcon } from "@/components/ui/alarm-clock-icon";
import { CloudSunRainIcon } from "@/components/ui/cloud-sun-rain-icon";
import { LoaderCircleIcon } from "@/components/ui/loader-circle-icon";
import { MapPinIcon } from "@/components/ui/map-pin-icon";
import { NavigationIcon } from "@/components/ui/navigation-icon";
import { SparklesIcon } from "@/components/ui/sparkles-icon";
import { ThumbsDownIcon } from "@/components/ui/thumbs-down-icon";
import { ThumbsUpIcon } from "@/components/ui/thumbs-up-icon";
import { TriangleAlertIcon } from "@/components/ui/triangle-alert-icon";
import { useState, useRef } from "react";
import { useQuery } from "@tanstack/react-query";
import { Correction } from "@/components/correction";
import { useFeedback } from "@/lib/use-feedback";

import { ResultChart } from "@/components/result-chart";
import { ResultTable } from "@/components/result-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { ChatMessage, Metrics as TurnMetrics, Nlu, PlanInfo, Quality } from "@/lib/types";
import { Compare } from "@/components/compare";
import { apiUrl } from "@/lib/utils";

const STAGE_COPY: Record<string, { icon: typeof Brain; label: string }> = {
  understanding: { icon: Brain, label: "Reading your question" },
  locating: { icon: Search, label: "Finding the place" },
  fetching: { icon: Radar, label: "Pulling the forecast" },
  writing: { icon: PenLine, label: "Writing the answer" },
};

/** Where the turn's time went. Only the stages that ran are listed - a greeting has an NLU
 *  number and nothing else, and a zero next to "Data API" would be a lie about work not done. */
function Metrics({ metrics }: { metrics: TurnMetrics }) {
  const stages: [string, number | undefined][] = [
    ["NLU", metrics.nlu_ms],
    ["Solr", metrics.solr_ms],
    ["Data API", metrics.api_ms],
    ["LLM", metrics.llm_ms],
    ["DB", metrics.db_ms],
  ];
  return (
    <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 border-t border-border/50 pt-2 text-[10px] text-muted-foreground">
      {stages
        .filter(([, ms]) => ms !== undefined)
        .map(([label, ms]) => (
          <span key={label}>
            {label} <span className="font-medium tabular-nums text-foreground">{ms}ms</span>
          </span>
        ))}
      <span className="ml-auto">
        total <span className="font-medium tabular-nums text-foreground">{metrics.total_ms}ms</span>
      </span>
    </div>
  );
}

function Bubble({ children, mine = false }: { children: React.ReactNode; mine?: boolean }) {
  return (
    <div className={mine ? "flex justify-end" : "flex justify-start"}>
      <div
        className={
          mine
            ? "max-w-[75%] rounded-2xl rounded-br-md bg-primary px-4 py-2.5 text-sm text-primary-foreground shadow-sm"
            : "w-full rounded-2xl rounded-bl-md border bg-card px-4 py-3.5 text-sm shadow-sm"
        }
      >
        {children}
      </div>
    </div>
  );
}

/** The 4 model targets, shown so you can see what the NLU actually understood. */
function NluChips({ nlu }: { nlu: Nlu }) {
  const { entities } = nlu;
  return (
    <div className="flex flex-wrap items-center gap-1.5 px-1 text-xs">
      {nlu.model && (
        <Badge variant="outline" className="font-mono text-[10px] uppercase">
          {nlu.model}
        </Badge>
      )}
      <Badge variant="secondary" className="gap-1 font-mono text-[11px]">
        <SparklesIcon size={12} isAnimated />
        {nlu.intent}
      </Badge>
      {(nlu.variables ?? []).length > 1 &&
        nlu.variables!.map((variable) => (
          <Badge key={variable} variant="secondary" className="font-mono text-[11px]">
            {variable}
          </Badge>
        ))}
      <Badge variant="outline" className="font-mono text-[11px]">
        {nlu.action}
      </Badge>
      {nlu.aggregation && nlu.aggregation !== "RAW" && (
        <Badge variant="outline" className="gap-1 font-mono text-[11px]">
          <Sigma className="h-3 w-3" />
          {nlu.aggregation}
        </Badge>
      )}
      {entities.location.map((place, index) => (
        <Badge key={`${place}-${index}`} variant="outline" className="gap-1 text-[11px]">
          <MapPinIcon size={12} isAnimated />
          {place}
        </Badge>
      ))}
      {entities.time.map((raw, index) => (
        <Badge key={raw + index} variant="outline" className="gap-1 text-[11px]">
          <AlarmClockIcon size={12} isAnimated />
          {raw}
          {entities.time_normalized[index] && entities.time_normalized[index] !== raw && (
            <span className="text-muted-foreground">→ {entities.time_normalized[index]}</span>
          )}
        </Badge>
      ))}
      <span className="text-muted-foreground">{Math.round(nlu.confidence * 100)}%</span>
    </div>
  );
}

/**
 * What the local model is reasoning, while it reasons. Shown rather than hidden: phrasing the
 * answer is the longest step of a turn, and its thinking is the only honest thing to put in
 * that gap.
 *
 * Open while it is being written, shut once the answer is in - a finished thought is a
 * one-line footnote, not the middle of the transcript. `<details>` rather than state: the
 * browser already has this widget.
 */
export function Thinking({ text, live = false }: { text: string; live?: boolean }) {
  return (
    <details
      open={live}
      className="group rounded-xl border border-dashed bg-muted/30 px-3 py-1.5 [&[open]]:py-2"
    >
      <summary className="flex cursor-pointer list-none items-center gap-1.5 text-[11px] text-muted-foreground">
        <Brain className={`h-3.5 w-3.5 ${live ? "animate-pulse" : ""}`} />
        {live ? "Thinking" : "Thought this through"}
        <span className="text-[10px] opacity-50 group-open:hidden">· show</span>
      </summary>
      <p className="mt-1.5 max-h-40 overflow-y-auto whitespace-pre-wrap text-[11px] leading-relaxed text-muted-foreground">
        {text}
      </p>
    </details>
  );
}

/** Browser geolocation, asked for only when the text had no usable place (Rule 4.1).
 *  Exported because the plain chat asks the same question and must ask it the same way. */
export function LocationPrompt({
  message,
  answered,
  onShare,
}: {
  message: string;
  answered: boolean;
  onShare: (lat: number, lon: number) => void;
}) {
  const [state, setState] = useState<"idle" | "asking" | "denied">("idle");

  const share = () => {
    if (!navigator.geolocation) return setState("denied");
    setState("asking");
    navigator.geolocation.getCurrentPosition(
      (position) => onShare(position.coords.latitude, position.coords.longitude),
      () => setState("denied"),
      { enableHighAccuracy: true, timeout: 10000 },
    );
  };

  return (
    <Card className="max-w-full gap-0 border-dashed p-4">
      <div className="flex items-center gap-2 text-sm font-medium">
        <NavigationIcon size={16} isAnimated />
        {message}
      </div>
      {answered ? (
        <p className="mt-2 text-xs text-muted-foreground">Location shared - re-running your question.</p>
      ) : state === "denied" ? (
        <p className="mt-2 text-xs text-destructive">
          Location permission denied. Name a place instead, e.g. &ldquo;rain in Guntur tomorrow&rdquo;.
        </p>
      ) : (
        <Button size="sm" className="mt-3 w-fit gap-2" onClick={share} disabled={state === "asking"}>
          {state === "asking" ? <LoaderCircleIcon size={14} isAnimated /> : <NavigationIcon size={14} />}
          Use my location
        </Button>
      )}
    </Card>
  );
}

/**
 * Thumbs on an answer, and the correction form behind the thumbs-down.
 *
 * A bare "wrong" is triage, not training data: it says something failed without saying what
 * the answer should have been. Thumbs-down therefore opens the form pre-filled with the
 * model's own reading, so fixing it is usually one tap.
 */
function Rate({
  turnId,
  model,
  intent,
  action,
  variables,
  locations,
  times,
}: {
  turnId: number;
  model: string;
  intent: string;
  action: string;
  variables: string[];
  locations: string[];
  times: string[];
}) {
  const feedback = useFeedback();
  const [sent, setSent] = useState<"up" | "down" | "corrected" | null>(null);
  const [correcting, setCorrecting] = useState(false);
  // an answer restored from an older build may predate stored turn ids
  const rateable = Number.isInteger(turnId);

  // what the user already said about this turn - a reopened chat shows its own ratings
  const existing = useQuery({
    queryKey: ["feedback", turnId],
    queryFn: async () =>
      (await fetch(`${apiUrl()}/api/feedback/${turnId}`)).json() as Promise<{
        feedback: { kind: string; revisions: number } | null;
      }>,
    enabled: rateable && sent === null,
    staleTime: 60_000,
  });
  const stored = existing.data?.feedback ?? null;

  if (correcting) {
    return (
      <Correction
        turnId={turnId}
        model={model}
        variables={variables}
        locations={locations}
        times={times}
        onDone={() => {
          setCorrecting(false);
          setSent("corrected");
        }}
      />
    );
  }

  if (feedback.isError) {
    return (
      <span className="mt-2 block text-[11px] text-destructive">
        Could not record that: {(feedback.error as Error).message}
      </span>
    );
  }
  const verdict = sent ?? (stored ? (stored.kind === "up" ? "up" : stored.kind === "down" ? "down" : "corrected") : null);
  if (verdict) {
    // only claimed once the request actually succeeded - a silent 422 used to read as "saved"
    const saved = !feedback.isPending;
    return (
      <span className="mt-2 flex items-center gap-2 text-[11px] text-muted-foreground">
        {!saved
          ? "Saving…"
          : verdict === "up"
            ? "Marked correct - thanks."
            : verdict === "corrected"
              ? "Correction saved - it joins the next training run."
              : "Marked wrong."}
        {saved && (
          <button
            onClick={() => {
              setSent(null);
              setCorrecting(verdict !== "up");   // changing a correction reopens the form
            }}
            className="underline decoration-dotted underline-offset-2 hover:text-foreground"
          >
            change
          </button>
        )}
      </span>
    );
  }
  if (!rateable) return null;

  return (
    <div className="mt-2 flex items-center gap-1">
      <button
        onClick={() => {
          setSent("up");
          feedback.mutate({
            turn_id: turnId, kind: "up", intent, action, model,
            variables, location: locations, time: times
          });
        }}
        aria-label="Correct answer"
        className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-emerald-600"
      >
        <ThumbsUpIcon size={14} isAnimated />
      </button>
      <button
        onClick={() => {
          feedback.mutate({ turn_id: turnId, kind: "down", intent, action, model });
          setCorrecting(true);      // ask what it should have been, while it is still on screen
        }}
        aria-label="Wrong answer"
        className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive"
      >
        <ThumbsDownIcon size={14} isAnimated />
      </button>
      <span className="ml-1 text-[10px] text-muted-foreground">was this right?</span>
    </div>
  );
}

function TtsButton({ text }: { text: string }) {
  const [loading, setLoading] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [ttsMs, setTtsMs] = useState<number | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handlePlay = async () => {
    if (playing && audioRef.current) {
      audioRef.current.pause();
      setPlaying(false);
      return;
    }

    if (audioRef.current) {
      audioRef.current.play();
      setPlaying(true);
      return;
    }

    try {
      setLoading(true);
      const ttsUrl = process.env.NEXT_PUBLIC_TTS_URL;
      if (!ttsUrl) {
        console.warn("NEXT_PUBLIC_TTS_URL is not set");
        return;
      }

      const start = performance.now();
      const response = await fetch(`${ttsUrl}/tts`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "audio/*"
        },
        body: JSON.stringify({
          text,
          language: "en",
          gender: "female"
        })
      });

      if (!response.ok) throw new Error("TTS failed");

      const arrayBuffer = await response.arrayBuffer();
      setTtsMs(Math.round(performance.now() - start));
      const blob = new Blob([arrayBuffer], { type: "audio/wav" });
      const url = URL.createObjectURL(blob);

      const audio = new Audio(url);
      audio.onended = () => setPlaying(false);
      audioRef.current = audio;

      audio.play();
      setPlaying(true);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      {ttsMs !== null && (
        <span className="text-[10px] text-muted-foreground whitespace-nowrap">TTS: {ttsMs}ms</span>
      )}
      <button
        onClick={handlePlay}
        className="inline-flex h-6 w-6 items-center justify-center rounded-md text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
        disabled={loading}
        aria-label="Play audio"
      >
        {loading ? <LoaderCircleIcon size={14} isAnimated /> : (playing ? <Square size={14} fill="currentColor" /> : <Volume2 size={14} />)}
      </button>
    </div>
  );
}

// The advice verdict is deliberately NOT rendered as a card here.
//
// It is still computed, and it still leads the answer: `pipeline.run` puts the headline at the
// front of the summary and `generation` words it. Showing it a second time as a coloured
// YES/NO badge made a correct answer look wrong - the sentence would say "you will probably be
// fine, there is a little rain around" while a red NO sat underneath it, because the rule's
// verdict is a threshold and the sentence is a description, and a reader trusts the badge.
// One statement of the decision, in the answer, is the whole point of the wording layer.
//
// The verdict, its reasons and its evidence are all still on the payload (`result.advice`) and
// are rendered per column in compare.tsx, where comparing what each model decided is the job.

/** Where the numbers came from and how complete they were - shown only when it is not routine. */
function SourceNote({ plan, quality }: { plan?: PlanInfo; quality?: Quality }) {
  const chips: string[] = [];
  if (plan?.served_by) chips.push(plan.served_by.replace(/_/g, " ").toLowerCase());
  if (plan?.resolution) chips.push(plan.resolution.toLowerCase());
  if (plan?.rows) chips.push(`${plan.rows} rows`);
  const problems = [
    plan?.fell_back_from ? `fell back from ${plan.fell_back_from.replace(/_/g, " ").toLowerCase()}` : "",
    plan?.unservable?.length ? `no ${plan.unservable.join(", ")} in that source` : "",
    quality && quality.status !== "OK" ? quality.message || quality.status.toLowerCase() : "",
  ].filter(Boolean);
  if (!chips.length && !problems.length) return null;
  return (
    <div className="mt-2 border-t pt-1.5">
      <div className="flex flex-wrap items-center gap-1.5">
        {chips.map((chip) => (
          <span key={chip} className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
            {chip}
          </span>
        ))}
      </div>
      {problems.map((problem) => (
        <p key={problem} className="mt-1 text-[11px] text-amber-600 dark:text-amber-500">{problem}</p>
      ))}
    </div>
  );
}

export function Messages({
  messages,
  onShareLocation,
}: {
  messages: ChatMessage[];
  onShareLocation: (text: string, lat: number, lon: number) => void;
}) {
  return (
    <div className="flex flex-col gap-3">
      {messages.map((message, index) => {
        // the last message is the one still being written, and the only one that animates
        const live = index === messages.length - 1;
        switch (message.role) {
          case "user":
            return (
              <Bubble key={message.id} mine>
                {message.text}
              </Bubble>
            );

          case "status": {
            const stage = STAGE_COPY[message.stage] ?? { icon: Brain, label: message.stage };
            const Icon = stage.icon;
            return (
              <div key={message.id} className="flex items-center gap-2 px-1 text-xs text-muted-foreground">
                <Icon className="h-3.5 w-3.5 animate-pulse" />
                {stage.label}
                <span className="inline-flex gap-0.5">
                  <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.3s]" />
                  <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground [animation-delay:-0.15s]" />
                  <span className="h-1 w-1 animate-bounce rounded-full bg-muted-foreground" />
                </span>
              </div>
            );
          }

          case "nlu":
            return <NluChips key={message.id} nlu={message.nlu} />;

          case "thinking":
            return <Thinking key={message.id} text={message.text} live={live} />;

          case "streaming":
            // the same bubble the finished answer arrives in, so nothing jumps when it does
            return (
              <Bubble key={message.id}>
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0">
                    <CloudSunRainIcon size={17} isAnimated />
                  </span>
                  <p className="min-w-0 font-medium">
                    {message.text}
                    <span className="ml-0.5 inline-block h-3.5 w-0.5 translate-y-0.5 animate-pulse rounded-full bg-primary" />
                  </p>
                </div>
              </Bubble>
            );

          case "ask-location":
            return (
              <LocationPrompt
                key={message.id}
                message={message.message}
                answered={message.answered}
                onShare={(lat, lon) => onShareLocation(message.text, lat, lon)}
              />
            );

          // The only clarify left: the query planner cannot serve that question. Both models
          // commit to a reading rather than asking which one you meant (Rule 1.1), so there
          // is no intent to pick here and nothing to label.
          case "clarify":
            return (
              <Card key={message.id} className="max-w-full gap-0 border-dashed p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <HelpCircle className="h-4 w-4 text-amber-500" />
                  {message.message}
                </div>
              </Card>
            );

          case "compare":
            return (
              <div key={message.id} className="w-full">
                <Compare
                  text={message.text}
                  models={message.models}
                  disagreements={message.disagreements}
                  totalMs={message.totalMs}
                  pending={message.pending}
                />
              </div>
            );

          case "chat":
            return (
              <Bubble key={message.id}>
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0">
                    <CloudSunRainIcon size={17} isAnimated />
                  </span>
                  <div className="min-w-0">
                    <p className="text-sm">{message.message}</p>
                    <div className="mt-1 flex flex-wrap items-center gap-1.5">
                      <span className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                        {message.intent.replace(/_/g, " ").toLowerCase()}
                      </span>
                      {message.locations.map((place) => (
                        <span key={place} className="rounded bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground">
                          {place}
                        </span>
                      ))}
                    </div>
                    {message.metrics && <Metrics metrics={message.metrics} />}
                  </div>
                </div>
              </Bubble>
            );

          case "error":
            return (
              <Bubble key={message.id}>
                <span className="flex items-center gap-2 text-destructive">
                  <TriangleAlertIcon size={16} isAnimated />
                  {message.message}
                </span>
              </Bubble>
            );

          case "assistant": {
            const { result } = message;
            return (
              <Bubble key={message.id}>
                <div className="flex items-start gap-2">
                  <span className="mt-0.5 shrink-0">
                    <CloudSunRainIcon size={17} isAnimated />
                  </span>
                  <div className="min-w-0">
                    <div className="flex items-start justify-between gap-2">
                      <p className="font-medium">{result.summary}</p>
                      <div className="shrink-0 pt-0.5">
                        <TtsButton text={result.summary} />
                      </div>
                    </div>
                    {result.reduced && (
                      <p className="mt-1 text-2xl font-semibold tabular-nums">
                        {result.reduced.value}
                        <span className="ml-1 text-sm font-normal text-muted-foreground">
                          {result.reduced.unit} · {result.reduced.kind.toLowerCase()}
                        </span>
                      </p>
                    )}
                    <div className="mt-1 flex flex-wrap gap-1.5 text-[11px] text-muted-foreground">
                      {result.places.map((place) => (
                        <span key={place.name} className="inline-flex items-center gap-1">
                          <MapPinIcon size={12} />
                          {place.name}
                          {place.district && place.district !== place.name ? `, ${place.district}` : ""}
                          <span className="opacity-60">
                            ({place.lat.toFixed(3)}, {place.lon.toFixed(3)})
                          </span>
                        </span>
                      ))}
                      <span>· {result.granularity}</span>
                      <span>· {result.when}</span>
                    </div>
                    {result.assumed && result.assumed.length > 0 && (
                      <p className="mt-1.5 text-[11px] text-muted-foreground">
                        Assumed: {result.assumed.join(" · ")}
                      </p>
                    )}
                    {result.chart && <ResultChart chart={result.chart} />}
                    <ResultTable data={result.table} />
                    {result.insights.length > 0 && (
                      <ul className="mt-2 space-y-1">
                        {result.insights.map((insight) => (
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
                    <SourceNote plan={result.plan} quality={result.quality} />
                    {result.presentation && (
                      <p className="mt-2 text-[10px] text-muted-foreground">
                        model chose: {result.presentation.detail.toLowerCase()} detail ·{" "}
                        {result.presentation.chart.toLowerCase().replace("_", " ")} ·{" "}
                        {result.presentation.insights.join(", ").toLowerCase() || "no insights"}
                      </p>
                    )}
                    {result.uncertain && (
                      <p className="mt-2 text-[11px] text-amber-600 dark:text-amber-500">
                        Not fully sure I read that right ({Math.round(result.confidence * 100)}%) -
                        the thumbs below correct it.
                      </p>
                    )}
                    <Rate
                      turnId={result.turn_id}
                      model={result.model}
                      intent={result.intent}
                      action={result.action}
                      variables={result.variables}
                      locations={result.places.map((place) => place.raw ?? place.name)}
                      times={result.when ? [result.when] : []}
                    />
                    {result.metrics && <Metrics metrics={result.metrics} />}
                  </div>
                </div>
              </Bubble>
            );
          }
        }
      })}
    </div>
  );
}
