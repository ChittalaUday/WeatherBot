"use client";

import { LazyMotion, domAnimation, m } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { ChatHistory } from "@/components/chat-history";
import { Composer } from "@/components/composer";
import { HealthBadge } from "@/components/health-badge";
import { Messages } from "@/components/messages";
import { ThemeToggle } from "@/components/theme-toggle";
import { Button } from "@/components/ui/button";
import { CloudSunRainIcon } from "@/components/ui/cloud-sun-rain-icon";
import { DropletsIcon } from "@/components/ui/droplets-icon";
import { MessageSquarePlusIcon } from "@/components/ui/message-square-plus-icon";
import { ThermometerIcon } from "@/components/ui/thermometer-icon";
import { WindIcon } from "@/components/ui/wind-icon";
import { useChat } from "@/lib/use-chat";

const EXAMPLES = [
  { icon: DropletsIcon, text: "will it rain in Nokha tommorrow?" },
  { icon: ThermometerIcon, text: "rain and temperature in Guntur tomorrow" },
  { icon: WindIcon, text: "compare max temp between Hyderabad and Vizag next 3 days" },
  { icon: CloudSunRainIcon, text: "soil moisture in my field right now" },
];

export default function Page() {
  const { busy, messages, chatId, ask, compare, sendLocation, newChat, openChat } = useChat();
  // Compare mode sends the same sentence to every model and shows them side by side. It needs
  // the whole screen - three columns of slots do not fit a reading-width column.
  const [compareMode, setCompareMode] = useState(false);
  const send = (text: string) => (compareMode ? compare(text) : ask(text, model));
  const width = compareMode ? "max-w-none" : "max-w-3xl";
  // Model 2 (v4) is the default: it can route a greeting away from the weather API,
  // decide an activity, and say which source answered. Model 1 stays selectable so the
  // two can be compared on the same question.
  const [model, setModel] = useState("v4");
  const bottom = useRef<HTMLDivElement>(null);
  const started = messages.length > 0;

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  return (
    <LazyMotion features={domAnimation}>
      <main className="flex h-dvh flex-col bg-background">
        <header className="sticky top-0 z-30 border-b bg-background/80 backdrop-blur">
          <div className={`mx-auto flex w-full ${width} items-center justify-between gap-3 px-4 py-2.5`}>
            <div className="flex min-w-0 items-center gap-2.5">
              <span className="grid h-9 w-9 shrink-0 place-items-center rounded-xl bg-primary/10">
                <CloudSunRainIcon size={19} isAnimated />
              </span>
              <span className="min-w-0">
                <span className="block truncate text-sm font-semibold leading-tight">
                  WeatherSnap
                </span>
                <span className="block truncate text-[11px] text-muted-foreground">
                  {compareMode
                    ? "comparing Model 1 · Model 2 · Model 3"
                    : chatId
                      ? `chat ${chatId.slice(5, 13)} · ${model}`
                      : "NLU weather assistant"}
                </span>
              </span>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
              <button
                type="button"
                onClick={() => setCompareMode((on) => !on)}
                aria-pressed={compareMode}
                title="Ask all three models at once and compare what each understood"
                className={`rounded-lg border px-2 py-1 text-[11px] font-medium transition-colors ${
                  compareMode
                    ? "border-primary bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                Compare 3
              </button>
              <ChatHistory currentChatId={chatId} onOpenChat={openChat} />
              <Button
                variant="ghost"
                size="sm"
                onClick={newChat}
                title="New chat - forgets the remembered place and time"
                className="h-8 gap-1.5 px-2 text-xs text-muted-foreground"
              >
                <MessageSquarePlusIcon size={15} isAnimated />
                <span className="hidden sm:inline">New</span>
              </Button>
              <ThemeToggle />
              <HealthBadge />
            </div>
          </div>
        </header>

        {started ? (
          <>
            <div className="flex flex-1 flex-col overflow-y-auto">
              {/* messages grow upward from the composer instead of stranding one answer at
                  the top of a tall screen */}
              <div className={`mx-auto flex w-full ${width} flex-1 flex-col justify-end px-4 py-6`}>
                <Messages
                  messages={messages}
                  onShareLocation={(text, lat, lon) => sendLocation(text, lat, lon, model)}
                />
                <div ref={bottom} className="h-2" />
              </div>
            </div>
            {/* a fade instead of a rule: the transcript should look like it runs under the
                composer, not like it stops at a line */}
            <div className="pointer-events-none h-6 shrink-0 bg-gradient-to-t from-background to-transparent" />
            <div className="sticky bottom-0 bg-background px-4">
              <Composer model={model} onModelChange={setModel} onSubmit={send} busy={busy} />
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center px-4">
            <m.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
              className={`w-full ${width}`}
            >
              <div className="mb-7 text-center">
                <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                  What&rsquo;s the weather doing?
                </h1>
                <p className="mt-2 text-sm text-muted-foreground">
                  Rain, temperature, wind and soil for any village, block or district in India.
                </p>
              </div>

              <Composer
                model={model}
                onModelChange={setModel}
                onSubmit={send}
                busy={busy}
                centered
              />

              <div className="mt-6 grid gap-2 sm:grid-cols-2">
                {EXAMPLES.map(({ icon: Icon, text }) => (
                  <button
                    key={text}
                    onClick={() => send(text)}
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
