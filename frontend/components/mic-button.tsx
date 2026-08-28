"use client";

import { useQuery } from "@tanstack/react-query";
import { Loader2, Mic, X } from "lucide-react";
import { useRef } from "react";
import { Waveform } from "@/components/waveform";
import { useDictation } from "@/lib/use-dictation";
import { cn, sttHealthUrl } from "@/lib/utils";

/**
 * Dictation for a composer: hold the text that was already typed, append what is spoken.
 *
 * The live trace is rendered here rather than left to each composer - both of them drop
 * this into the same flex row beside Send, so it only needs saying once.
 *
 * The speech service is polled before offering the mic, because the failure is otherwise
 * silent: the socket opens, audio goes out, and nothing ever comes back.
 */
export function MicButton({
  value,
  onChange,
  disabled,
}: {
  value: string;
  onChange: (text: string) => void;
  disabled?: boolean;
}) {
  const base = useRef("");
  const { recording, pending, error, analyser, toggle } = useDictation((spoken) =>
    onChange(`${base.current} ${spoken}`.trim()),
  );

  const { data, isPending, isError } = useQuery({
    queryKey: ["stt-health"],
    queryFn: async () => {
      // Don't treat a non-ok status as a failure - only an unreachable host is that.
      // The server binds its port after the weights are in, so "loading" is a moment, and
      // an unstarted container shows up as a fetch error rather than a bad status.
      const response = await fetch(sttHealthUrl());
      return (await response.json()) as { status: string; version: string };
    },
    // Check often while it is warming up, then settle down once it is answering.
    refetchInterval: (query) => (query.state.data?.status === "ok" ? 30_000 : 5_000),
    retry: false,
  });

  const unavailable = isPending
    ? "Checking the speech service…"
    : isError
      ? "Speech service unreachable - start it with: docker compose up -d in stt-cpp/"
      : data?.status !== "ok"
        ? `Speech service is not ready (${data?.status ?? "unknown"})…`
        : null;

  const label = pending ? "Transcribing" : recording ? "Stop dictation" : "Speak";

  return (
    <>
      {recording && analyser && (
        <Waveform analyser={analyser} className="h-5 w-32 shrink-0 text-red-500 sm:w-48" />
      )}
      {/* A disabled button swallows hover in Chrome and Safari, so the tooltip has to sit
          on a wrapper that is not itself disabled. */}
      <span
        className="inline-flex shrink-0"
        title={unavailable ?? error ?? (pending ? "Transcribing…" : label)}
      >
        <button
          type="button"
          suppressHydrationWarning
          // The closing words land just after the mic is released, so a stopped mic is
          // not a finished one: block a restart until the server has closed the utterance.
          disabled={Boolean(disabled || pending || unavailable !== null)}
          onClick={() => {
            if (!recording) base.current = value;
            toggle();
          }}
          aria-label={unavailable ?? label}
          aria-busy={pending}
          aria-pressed={recording}
          className={cn(
            "grid h-8 w-8 shrink-0 place-items-center rounded-full border transition-colors disabled:opacity-40",
            recording
              ? "animate-pulse border-red-500/40 bg-red-500/10 text-red-500"
              : "text-muted-foreground hover:text-foreground",
            pending && "disabled:opacity-100",
            error && !recording && "border-red-500/40 text-red-500",
          )}
        >
          {pending ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : recording ? (
            <X className="h-4 w-4" />
          ) : (
            <Mic className="h-4 w-4" />
          )}
        </button>
      </span>
    </>
  );
}
