"use client";

import { LazyMotion, domAnimation, m } from "motion/react";
import { useEffect, useRef, useState } from "react";
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
import { useWeatherSocket } from "@/lib/use-weather-socket";

const EXAMPLES = [
  { icon: DropletsIcon, text: "will it rain in Nokha tommorrow?" },
  { icon: ThermometerIcon, text: "rain and temperature in Guntur tomorrow" },
  { icon: WindIcon, text: "compare max temp between Hyderabad and Vizag next 3 days" },
  { icon: CloudSunRainIcon, text: "soil moisture in my field right now" },
];

export default function Page() {
  const { connected, busy, messages, chatId, ask, sendLocation, newChat } = useWeatherSocket();
  const [model, setModel] = useState("v1");
  const bottom = useRef<HTMLDivElement>(null);
  const started = messages.length > 0;

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

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
                  WeatherSnap
                </span>
                <span className="block truncate text-[11px] text-muted-foreground">
                  {chatId ? `chat ${chatId.slice(5, 13)} · ${model}` : "NLU weather assistant"}
                </span>
              </span>
            </div>
            <div className="flex shrink-0 items-center gap-1.5">
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
              <HealthBadge connected={connected} />
            </div>
          </div>
        </header>

        {started ? (
          <>
            <div className="flex flex-1 flex-col overflow-y-auto">
              {/* messages grow upward from the composer instead of stranding one answer at
                  the top of a tall screen */}
              <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-end px-4 py-6">
                <Messages
                  messages={messages}
                  onShareLocation={(text, lat, lon) => sendLocation(text, lat, lon, model)}
                  onAsk={(text) => ask(text, model)}
                />
                <div ref={bottom} className="h-2" />
              </div>
            </div>
            <div className="sticky bottom-0 border-t bg-background/95 px-4 backdrop-blur">
              <Composer
                model={model}
                onModelChange={setModel}
                onSubmit={(text) => ask(text, model)}
                busy={busy}
                connected={connected}
              />
            </div>
          </>
        ) : (
          <div className="flex flex-1 flex-col items-center justify-center px-4">
            <m.div
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.35 }}
              className="w-full max-w-3xl"
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
                onSubmit={(text) => ask(text, model)}
                busy={busy}
                connected={connected}
                centered
              />

              <div className="mt-6 grid gap-2 sm:grid-cols-2">
                {EXAMPLES.map(({ icon: Icon, text }) => (
                  <button
                    key={text}
                    onClick={() => ask(text, model)}
                    disabled={!connected}
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
