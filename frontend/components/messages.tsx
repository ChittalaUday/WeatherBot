"use client";

import {
  AlertTriangle,
  Brain,
  CloudSunRain,
  Clock,
  HelpCircle,
  Loader2,
  MapPin,
  Navigation,
  Radar,
  Search,
  Sigma,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { useState } from "react";
import { useFeedback } from "@/lib/use-feedback";
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
            ? "max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-4 py-2.5 text-sm text-primary-foreground"
            : "max-w-full rounded-2xl rounded-bl-sm bg-muted/60 px-4 py-3 text-sm"
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
      <Badge variant="secondary" className="gap-1 font-mono text-[11px]">
        <Sparkles className="h-3 w-3" />
        {nlu.intent}
      </Badge>
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
          <MapPin className="h-3 w-3" />
          {place}
        </Badge>
      ))}
      {entities.time.map((raw, index) => (
        <Badge key={raw + index} variant="outline" className="gap-1 text-[11px]">
          <Clock className="h-3 w-3" />
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
        <Navigation className="h-4 w-4 animate-pulse text-primary" />
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
          {state === "asking" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Navigation className="h-3.5 w-3.5" />}
          Use my location
        </Button>
      )}
    </Card>
  );
}

/** Thumbs on an answer. Every click is a label the retraining loop can use. */
function Rate({ turnId, intent, action }: { turnId: number; intent: string; action: string }) {
  const feedback = useFeedback();
  const [sent, setSent] = useState<"up" | "down" | null>(null);

  const send = (kind: "up" | "down") => {
    setSent(kind);
    feedback.mutate({ turn_id: turnId, kind, intent, action });
  };

  if (sent) {
    return (
      <span className="mt-2 block text-[11px] text-muted-foreground">
        {sent === "up" ? "Marked correct - thanks." : "Marked wrong - it goes into the fix list."}
      </span>
    );
  }
  return (
    <div className="mt-2 flex items-center gap-1">
      <button
        onClick={() => send("up")}
        aria-label="Correct answer"
        className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-emerald-600"
      >
        <ThumbsUp className="h-3.5 w-3.5" />
      </button>
      <button
        onClick={() => send("down")}
        aria-label="Wrong answer"
        className="rounded p-1 text-muted-foreground transition-colors hover:bg-muted hover:text-destructive"
      >
        <ThumbsDown className="h-3.5 w-3.5" />
      </button>
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
            const stage = STAGE_COPY[message.stage] ?? { icon: Loader2, label: message.stage };
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
                  <AlertTriangle className="h-4 w-4" />
                  {message.message}
                </span>
              </Bubble>
            );

          case "assistant": {
            const { result } = message;
            return (
              <Bubble key={message.id}>
                <div className="flex items-start gap-2">
                  <CloudSunRain className="mt-0.5 h-4 w-4 shrink-0 text-primary" />
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
                          <MapPin className="h-3 w-3" />
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
                    <Rate turnId={result.turn_id} intent={result.intent} action={result.action} />
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
