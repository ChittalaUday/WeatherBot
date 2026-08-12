"use client";

import { CloudSunRain, CornerDownLeft, Loader2 } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { HealthBadge } from "@/components/health-badge";
import { Messages } from "@/components/messages";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { useWeatherSocket } from "@/lib/use-weather-socket";

const EXAMPLES = [
  "will it rain in Nokha tommorrow?",
  "compare max temp between Hyderabad and Vizag next 3 days",
  "soil moisture in my field right now",
  "humidity in Guntur at 6:45 pm",
  "alert me if wind speed crosses 40 kmph in Kakinada tonight",
];

export default function Page() {
  const { connected, busy, messages, ask, sendLocation } = useWeatherSocket();
  const [draft, setDraft] = useState("");
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    ask(draft);
    setDraft("");
  };

  return (
    <main className="mx-auto flex h-dvh max-w-3xl flex-col px-4">
      <header className="flex items-center justify-between gap-3 border-b py-4">
        <div className="flex items-center gap-2.5">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-primary/10">
            <CloudSunRain className="h-5 w-5 text-primary" />
          </div>
          <div>
            <h1 className="text-sm font-semibold leading-tight">WeatherSnap</h1>
            <p className="text-xs text-muted-foreground">Ask in plain English</p>
          </div>
        </div>
        <HealthBadge connected={connected} />
      </header>

      <div className="flex-1 overflow-y-auto py-5">
        {messages.length === 0 ? (
          <div className="flex h-full flex-col items-center justify-center gap-4 text-center">
            <div className="grid h-14 w-14 place-items-center rounded-2xl bg-muted">
              <CloudSunRain className="h-7 w-7 text-muted-foreground" />
            </div>
            <div>
              <p className="text-sm font-medium">Rain, temperature, soil, wind - anywhere in India</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Typos are fine. Say &ldquo;my field&rdquo; and it will ask for your location.
              </p>
            </div>
            <div className="flex flex-wrap justify-center gap-2">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  onClick={() => ask(example)}
                  className="rounded-full border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-muted hover:text-foreground"
                >
                  {example}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <Messages messages={messages} onShareLocation={sendLocation} onAsk={ask} />
        )}
        <div ref={bottom} />
      </div>

      <form onSubmit={submit} className="sticky bottom-0 flex gap-2 border-t bg-background py-4">
        <Input
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          placeholder={connected ? "will it rain in Guntur tomorrow?" : "connecting…"}
          disabled={!connected}
          className="flex-1"
        />
        <Button type="submit" disabled={!connected || busy || !draft.trim()} className="gap-1.5">
          {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <CornerDownLeft className="h-4 w-4" />}
          Ask
        </Button>
      </form>
    </main>
  );
}
