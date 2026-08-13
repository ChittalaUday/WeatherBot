"use client";

import { Brain, HelpCircle, Radar, Search, Sigma } from "lucide-react";
import { AlarmClockIcon } from "@/components/ui/alarm-clock-icon";
import { CloudSunRainIcon } from "@/components/ui/cloud-sun-rain-icon";
import { LoaderCircleIcon } from "@/components/ui/loader-circle-icon";
import { MapPinIcon } from "@/components/ui/map-pin-icon";
import { NavigationIcon } from "@/components/ui/navigation-icon";
import { SparklesIcon } from "@/components/ui/sparkles-icon";
import { ThumbsDownIcon } from "@/components/ui/thumbs-down-icon";
import { ThumbsUpIcon } from "@/components/ui/thumbs-up-icon";
import { TriangleAlertIcon } from "@/components/ui/triangle-alert-icon";
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Correction } from "@/components/correction";
import { useFeedback } from "@/lib/use-feedback";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://127.0.0.1:8787";
import { ResultChart } from "@/components/result-chart";
import { ResultTable } from "@/components/result-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import type { ChatMessage, Nlu } from "@/lib/types";

const STAGE_COPY: Record<string, { icon: typeof Brain; label: string }> = {
  understanding: { icon: Brain, label: "Reading your question" },
  locating: { icon: Search, label: "Finding the place" },
  fetching: { icon: Radar, label: "Pulling the forecast" },
};

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

/** Browser geolocation, asked for only when the text had no usable place (Rule 4.1). */
function LocationPrompt({
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
      (await fetch(`${API}/api/feedback/${turnId}`)).json() as Promise<{
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
        intent={intent}
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
          feedback.mutate({ turn_id: turnId, kind: "up", intent, action, model,
                            variables, location: locations, time: times });
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

export function Messages({
  messages,
  onShareLocation,
  onAsk,
}: {
  messages: ChatMessage[];
  onShareLocation: (text: string, lat: number, lon: number) => void;
  onAsk: (text: string) => void;
}) {
  const feedback = useFeedback();

  /** Picking an intent answers the model's question - the cheapest gold label there is. */
  const onChoose = (turnId: number, intent: string, followUp: string) => {
    feedback.mutate({ turn_id: turnId, kind: "choice", intent });
    onAsk(followUp);
  };

  return (
    <div className="flex flex-col gap-3">
      {messages.map((message) => {
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

          case "ask-location":
            return (
              <LocationPrompt
                key={message.id}
                message={message.message}
                answered={message.answered}
                onShare={(lat, lon) => onShareLocation(message.text, lat, lon)}
              />
            );

          case "clarify":
            return (
              <Card key={message.id} className="max-w-full gap-0 border-dashed p-4">
                <div className="flex items-center gap-2 text-sm font-medium">
                  <HelpCircle className="h-4 w-4 text-amber-500" />
                  {message.message}
                </div>
                {message.options.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {message.options.map((option) => (
                      <Button
                        key={option.label}
                        size="sm"
                        variant="outline"
                        className="h-7 text-xs"
                        onClick={() => onChoose(message.turnId, option.intent, `${message.text} ${option.label}`)}
                      >
                        {option.label}
                        <span className="ml-1 text-muted-foreground">
                          {Math.round(option.confidence * 100)}%
                        </span>
                      </Button>
                    ))}
                  </div>
                )}
              </Card>
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
                    <p className="font-medium">{result.summary}</p>
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
                          <li key={insight} className="flex gap-1.5 text-[11px] text-muted-foreground">
                            <span className="text-primary">•</span>
                            {insight}
                          </li>
                        ))}
                      </ul>
                    )}
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
