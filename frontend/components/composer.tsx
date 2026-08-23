"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowUp, ChevronDown, Layers, MapPin } from "lucide-react";
import { m } from "motion/react";
import { useRef, useState } from "react";
import { AIInputDropdown, AIInputPillButton } from "@/components/ai-input";
import { LoaderCircleIcon } from "@/components/ui/loader-circle-icon";
import { apiUrl, cn } from "@/lib/utils";

type ModelItem = { id: string; label: string; description: string; icon: typeof Layers };

/**
 * The composer from components/ai-input.tsx, wired to this app's transcript.
 *
 * Its own AIInput manages an internal message list and fakes a reply, which would fight the
 * real transcript above; the dropdown and pill primitives are reused instead so the look is
 * the component's and the state stays ours. The dropdown lists whatever /api/models serves.
 */
export function Composer({
  model,
  onModelChange,
  onSubmit,
  busy,
  centered = false,
}: {
  model: string;
  onModelChange: (id: string) => void;
  onSubmit: (text: string) => void;
  busy: boolean;
  centered?: boolean;
}) {
  const [value, setValue] = useState("");
  const [open, setOpen] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);

  const { data } = useQuery({
    queryKey: ["models"],
    queryFn: async () => (await fetch(`${apiUrl()}/api/models`)).json(),
    staleTime: 5 * 60_000,
  });

  const models: ModelItem[] = (data?.available ?? [])
    .filter((entry: { present: boolean }) => entry.present)
    .map((entry: { version: string; name?: string; description: string }) => ({
      id: entry.version,
      // name comes from the backend, so the pill never drifts from what is actually served
      label: entry.name ?? entry.version,
      description: entry.description,
      icon: Layers,
    }));
  const selected = models.find((item) => item.id === model);

  const submit = () => {
    if (!value.trim() || busy) return;
    onSubmit(value.trim());
    setValue("");
    if (textarea.current) textarea.current.style.height = "auto";
  };

  return (
    <div className={cn("w-full", centered ? "" : "pb-5 pt-2")}>
      <m.div
        layout
        transition={{ type: "spring", damping: 26, stiffness: 220 }}
        className="relative mx-auto w-full max-w-3xl rounded-[28px] border border-black/5 bg-card shadow-sm dark:border-white/10"
      >
        <div className="p-4 pb-14">
          <textarea
            suppressHydrationWarning
            ref={textarea}
            value={value}
            rows={1}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                submit();
              }
            }}
            onInput={(event) => {
              const target = event.target as HTMLTextAreaElement;
              target.style.height = "auto";
              target.style.height = `${Math.min(target.scrollHeight, 180)}px`;
            }}
            placeholder="Ask about rain, soil, wind - anywhere in India"
            className="max-h-[180px] min-h-[44px] w-full resize-none bg-transparent text-base outline-none placeholder:text-muted-foreground"
          />
        </div>

        <div className="absolute bottom-3 left-3 right-3 z-10 flex items-center justify-between">
          <div className="relative">
            <AIInputPillButton
              icon={Layers}
              isActive={open}
              showChevron
              chevronRotated={open}
              onClick={() => setOpen((current) => !current)}
              className="!py-1.5 text-xs"
            >
              {selected?.label ?? "model"}
            </AIInputPillButton>
            <AIInputDropdown
              isOpen={open}
              onClose={() => setOpen(false)}
              items={models}
              className="w-72"
              renderItem={(item: ModelItem) => (
                <button
                  key={item.id}
                  onClick={() => {
                    onModelChange(item.id);
                    setOpen(false);
                  }}
                  className={cn(
                    "flex w-full items-start gap-2 rounded-xl px-3 py-2 text-left transition-colors hover:bg-muted",
                    item.id === model && "bg-muted",
                  )}
                >
                  <Layers className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                  <span className="min-w-0">
                    <span className="block text-sm font-medium">{item.label}</span>
                    <span className="block text-[11px] leading-snug text-muted-foreground">
                      {item.description}
                    </span>
                  </span>
                </button>
              )}
            />
          </div>

          <div className="flex items-center gap-2">
            <span className="hidden items-center gap-1 text-[11px] text-muted-foreground sm:flex">
              <MapPin className="h-3 w-3" />
              asks for your location only when needed
            </span>
            <button
              onClick={submit}
              suppressHydrationWarning
              disabled={busy || !value.trim()}
              aria-label="Send"
              className={cn(
                "grid h-9 w-9 place-items-center rounded-full transition-all",
                value.trim() && !busy
                  ? "bg-primary text-primary-foreground hover:scale-105"
                  : "bg-muted text-muted-foreground",
              )}
            >
              {busy ? (
                <LoaderCircleIcon size={16} isAnimated />
              ) : (
                <ArrowUp className="h-4 w-4" />
              )}
            </button>
          </div>
        </div>
      </m.div>
      <p className="mx-auto mt-2 max-w-3xl px-1 text-[11px] text-muted-foreground">
        Typos are fine. Say &ldquo;my field&rdquo; and it asks for your location.
        <ChevronDown className="inline h-3 w-3 rotate-90 opacity-0" />
      </p>
    </div>
  );
}
