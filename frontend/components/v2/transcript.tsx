"use client";

import { AlertCircle, MapPin, MessageCircleQuestion } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LoaderCircleIcon } from "@/components/ui/loader-circle-icon";
import { Answer } from "@/components/v2/answer";
import { ComparePanel } from "@/components/v2/compare-panel";
import { Thinking } from "@/components/v2/thinking";
import type { ChatMessage, Candidate } from "@/lib/v2/types";

/** What each stage is actually waiting on, so the line says something true. */
const STAGE: Record<string, string> = {
  understanding: "reading the question",
  locating: "resolving the place",
  fetching: "fetching the forecast",
  writing: "evaluating the policy",
};

function Bubble({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-3.5 py-2 text-sm text-primary-foreground">
        {children}
      </div>
    </div>
  );
}

export function Transcript({
  messages,
  onShareLocation,
  onPickPlace,
}: {
  messages: ChatMessage[];
  onShareLocation: (text: string, lat: number, lon: number) => void;
  onPickPlace: (text: string, candidate: Candidate) => void;
}) {
  return (
    <div className="space-y-5">
      {messages.map((message) => {
        switch (message.role) {
          case "user":
            return <Bubble key={message.id}>{message.text}</Bubble>;

          case "status":
            return (
              <div
                key={message.id}
                className="flex items-center gap-2 text-xs text-muted-foreground"
              >
                <LoaderCircleIcon size={13} isAnimated />
                {STAGE[message.stage] ?? message.stage}
              </div>
            );

          case "nlu":
            // What the parser read, before anything was fetched. Kept in the transcript
            // because it is the first thing to check when an answer is about the wrong thing.
            return (
              <div
                key={message.id}
                className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground"
              >
                <span className="font-mono">{message.nlu.intent}</span>
                {message.nlu.entities.location[0] && (
                  <span>place: {message.nlu.entities.location[0]}</span>
                )}
                {message.nlu.entities.time[0] && <span>when: {message.nlu.entities.time[0]}</span>}
                <span className="tabular-nums">conf {message.nlu.confidence}</span>
                <span className="font-mono opacity-60">{message.nlu.model}</span>
              </div>
            );

          case "thinking":
            return <Thinking key={message.id} text={message.text} done={message.done} />;

          case "streaming":
            // The answer as the model writes it. Replaced wholesale by the finished answer,
            // so the two never sit on screen together.
            return (
              <div key={message.id} className="rounded-2xl border bg-card px-4 py-3">
                <p className="text-sm leading-relaxed">
                  {message.text}
                  <span className="ml-0.5 inline-block h-3.5 w-[2px] animate-pulse bg-primary align-middle" />
                </p>
              </div>
            );

          case "answer":
            return <Answer key={message.id} result={message.result} />;

          case "chat":
            return (
              <div key={message.id} className="rounded-2xl border bg-card px-4 py-3">
                <p className="text-sm leading-relaxed">{message.message}</p>
              </div>
            );

          case "clarify":
            return (
              <div
                key={message.id}
                className="rounded-2xl border border-amber-500/30 bg-amber-500/[0.06] px-4 py-3"
              >
                <p className="flex gap-2 text-sm leading-relaxed">
                  <MessageCircleQuestion className="mt-0.5 h-4 w-4 shrink-0 text-amber-600 dark:text-amber-400" />
                  <span>{message.message}</span>
                </p>
                {/* An ambiguous place is answerable in one tap rather than by retyping. */}
                {message.candidates.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {message.candidates.map((candidate: Candidate) => (
                      <Button
                        key={`${candidate.name}-${candidate.lat}`}
                        variant="outline"
                        size="sm"
                        className="h-7 gap-1 text-xs"
                        onClick={() => onPickPlace(message.text, candidate)}
                      >
                        <MapPin className="h-3 w-3" />
                        {candidate.name}
                        {candidate.district && candidate.district !== candidate.name && (
                          <span className="text-muted-foreground">· {candidate.district}</span>
                        )}
                      </Button>
                    ))}
                  </div>
                )}
              </div>
            );

          case "ask-location":
            return (
              <div key={message.id} className="rounded-2xl border bg-card px-4 py-3">
                <p className="text-sm leading-relaxed">{message.message}</p>
                {!message.answered && (
                  <Button
                    size="sm"
                    variant="outline"
                    className="mt-2.5 h-7 gap-1.5 text-xs"
                    onClick={() =>
                      navigator.geolocation.getCurrentPosition(
                        (position) =>
                          onShareLocation(
                            message.text,
                            position.coords.latitude,
                            position.coords.longitude,
                          ),
                        () => undefined,
                        { timeout: 10_000 },
                      )
                    }
                  >
                    <MapPin className="h-3 w-3" />
                    Use my location
                  </Button>
                )}
              </div>
            );

          case "compare":
            return (
              <ComparePanel
                key={message.id}
                columns={message.columns}
                disagreements={message.disagreements}
                agreed={message.agreed}
                pending={message.pending}
                totalMs={message.totalMs}
              />
            );

          case "error":
            return (
              <div
                key={message.id}
                className="flex gap-2 rounded-2xl border border-rose-500/30 bg-rose-500/[0.06] px-4 py-3 text-sm"
              >
                <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-rose-600 dark:text-rose-400" />
                <span className="min-w-0 break-words">{message.message}</span>
              </div>
            );
        }
      })}
    </div>
  );
}
