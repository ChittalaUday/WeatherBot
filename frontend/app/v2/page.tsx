"use client";

import { LazyMotion, domAnimation, m } from "motion/react";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import { CloudSunRainIcon } from "@/components/ui/cloud-sun-rain-icon";
import { DropletsIcon } from "@/components/ui/droplets-icon";
import { MessageSquarePlusIcon } from "@/components/ui/message-square-plus-icon";
import { SparklesIcon } from "@/components/ui/sparkles-icon";
import { ThermometerIcon } from "@/components/ui/thermometer-icon";
import { formatMs } from "@/components/v2/answer";
import { ComposerV2 } from "@/components/v2/composer-v2";
import { HealthV2 } from "@/components/v2/health-v2";
import { Transcript } from "@/components/v2/transcript";
import { useChatV2 } from "@/lib/v2/use-chat-v2";
import type { Parser } from "@/lib/v2/types";

/**
 * The v2 chat: a separate conversation against Backend-v2 on port 8788.
 *
 * Deliberately not the v1 page with a different URL. v2 answers a different kind of question -
 * it returns a policy verdict with the values it was read from and the things it could not see -
 * and this page is built to show that rather than to hide it behind a paragraph. The v1 chat at
 * `/` is untouched and still talks to 8787.
 *
 * Its own chat id, its own history, its own parser switch (rules or Rasa).
 */

const EXAMPLES = [
  { icon: SparklesIcon, text: "Can I play cricket at 6 PM in Guntur?" },
  { icon: DropletsIcon, text: "Should I carry a raincoat tomorrow in Hyderabad?" },
  { icon: ThermometerIcon, text: "Will it rain in Warangal this evening?" },
  { icon: CloudSunRainIcon, text: "Can I spray pesticide tomorrow morning in Guntur?" },
];

export default function V2Page() {
  const {
    busy,
    messages,
    chatId,
    parser,
    chooseParser,
    ask,
    compare,
    sendLocation,
    pickPlace,
    newChat,
  } = useChatV2();
  // Only parsers the backend says are loaded can be selected; Rasa is an optional container.
  const [available, setAvailable] = useState<Parser[]>(["rules"]);
  const bottom = useRef<HTMLDivElement>(null);
  const started = messages.length > 0;

  // Every answer's time, and the total for the conversation. The per-turn number lives on each
  // answer; this is the running total, which is the one that tells you whether the thing is
  // getting slower as you use it.
  const answered = messages.filter((m) => m.role === "answer");
  const totalMs = answered.reduce(
    (sum, m) => sum + (m.role === "answer" ? (m.result.metrics?.total_ms ?? 0) : 0),
    0,
  );
  const slowest = answered.reduce(
    (worst, m) =>
      m.role === "answer" ? Math.max(worst, m.result.metrics?.total_ms ?? 0) : worst,
    0,
  );

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // If the chosen parser goes away (container stopped), fall back rather than sending to it.
  useEffect(() => {
    if (available.length && !available.includes(parser)) chooseParser(available[0]);
  }, [available, parser, chooseParser]);

  return (
    <LazyMotion features={domAnimation}>
      <main className="flex h-dvh flex-col bg-background">
        <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur">
          <div className="mx-auto flex w-full max-w-3xl items-center justify-between gap-3 px-4 py-2.5">
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-primary/10">
                <CloudSunRainIcon size={19} isAnimated />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold leading-tight">
                  WeatherSnap v2
                </span>
                <span className="block truncate text-[11px] text-muted-foreground">
                  {answered.length > 0
                    ? `${answered.length} answer${answered.length === 1 ? "" : "s"} · ${formatMs(
                        totalMs,
                      )} total · slowest ${formatMs(slowest)} · ${parser}`
                    : chatId
                      ? `chat ${chatId.slice(5, 13)} · ${parser}`
                      : "decision-first backend"}
                </span>
              </span>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              <Link
                href="/"
                className="rounded-lg border px-2 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground"
                title="The v1 chat, on port 8787"
              >
                v1
              </Link>
              <Button
                variant="ghost"
                size="sm"
                onClick={newChat}
                title="New chat - forgets the remembered place, time and activity"
                className="h-8 gap-1.5 px-2 text-xs text-muted-foreground"
              >
                <MessageSquarePlusIcon size={15} isAnimated />
                <span className="hidden sm:inline">New</span>
              </Button>
              <ThemeToggle />
              <HealthV2 onParsers={setAvailable} />
            </div>
          </div>
        </header>

        {started ? (
          <>
            <div className="flex flex-1 flex-col overflow-y-auto">
              <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-end px-4 py-6">
                <Transcript
                  messages={messages}
                  onShareLocation={sendLocation}
                  onPickPlace={pickPlace}
                />
                <div ref={bottom} className="h-2" />
              </div>
            </div>
            <div className="pointer-events-none h-6 shrink-0 bg-gradient-to-t from-background to-transparent" />
            <div className="sticky bottom-0 bg-background px-4">
              <ComposerV2
                parser={parser}
                onParserChange={chooseParser}
                available={available}
                onSubmit={ask}
                onCompare={compare}
                busy={busy}
              />
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center px-4">
            <m.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
              className="w-full max-w-2xl"
            >
              <div className="mb-7 text-center">
                <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                  Should you do it, and why?
                </h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Every answer is a versioned policy applied to forecast values you can see -
                  including what the feed could not tell it.
                </p>
              </div>

              <ComposerV2
                parser={parser}
                onParserChange={chooseParser}
                available={available}
                onSubmit={ask}
                onCompare={compare}
                busy={busy}
                centered
              />

              <div className="mt-6 grid gap-2 sm:grid-cols-2">
                {EXAMPLES.map(({ icon: Icon, text }) => (
                  <button
                    key={text}
                    onClick={() => ask(text)}
                    disabled={busy}
                    className="group flex items-center gap-2.5 rounded-2xl border bg-card px-3.5 py-3 text-left text-sm text-muted-foreground transition-colors hover:border-primary/30 hover:bg-muted hover:text-foreground disabled:opacity-50"
                  >
                    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-lg bg-muted transition-colors group-hover:bg-background">
                      <Icon size={16} isAnimated />
                    </span>
                    <span className="min-w-0 truncate">{text}</span>
                  </button>
                ))}
              </div>
            </m.div>
          </div>
        )}
      </main>
    </LazyMotion>
  );
}
