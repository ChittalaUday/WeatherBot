"use client";

import { ArrowUp, Cpu, Regex, Sparkles } from "lucide-react";
import { useRef, useState } from "react";
import { AIInputDropdown, AIInputPillButton } from "@/components/ai-input";
import { MicButton } from "@/components/mic-button";
import { LoaderCircleIcon } from "@/components/ui/loader-circle-icon";
import { cn } from "@/lib/utils";
import type { Parser } from "@/lib/v2/types";

/**
 * The input, plus the parser switch.
 *
 * The dropdown primitive is the app's own (`components/ai-input.tsx`); only the reachable
 * parsers are offered, so choosing Rasa while its container is down is not possible.
 */

const PARSERS = [
  {
    id: "rules" as Parser,
    label: "Rules cascade",
    description: "Deterministic: normalization, gazetteer patterns, entity rules. ~0.03 ms.",
    icon: Regex,
  },
  {
    id: "rasa" as Parser,
    label: "Rasa DIET",
    description: "Rasa 3.6 NLU. Intent + entities from a trained model. ~7 ms.",
    icon: Cpu,
  },
  {
    id: "llm" as Parser,
    label: "Hosted model",
    description:
      "The generation model reading the question as JSON. Broadest language coverage, " +
      "slowest, and it never resolves a place or a time itself.",
    icon: Sparkles,
  },
];

export function ComposerV2({
  parser,
  onParserChange,
  available,
  onSubmit,
  onCompare,
  busy,
  centered = false,
}: {
  parser: Parser;
  onParserChange: (parser: Parser) => void;
  available: Parser[];
  onSubmit: (text: string) => void;
  onCompare: (text: string) => void;
  busy: boolean;
  centered?: boolean;
}) {
  const [value, setValue] = useState("");
  const [open, setOpen] = useState(false);
  const textarea = useRef<HTMLTextAreaElement>(null);

  const items = PARSERS.filter((item) => available.includes(item.id));
  const selected = items.find((item) => item.id === parser) ?? items[0];

  const send = (compare: boolean) => {
    const text = value.trim();
    if (!text || busy) return;
    (compare ? onCompare : onSubmit)(text);
    setValue("");
    if (textarea.current) textarea.current.style.height = "auto";
  };

  return (
    <div className={cn("mx-auto w-full", centered ? "max-w-2xl" : "max-w-3xl", "pb-4")}>
      <div className="rounded-3xl border bg-card p-2 shadow-sm focus-within:border-primary/40">
        <textarea
          ref={textarea}
          value={value}
          rows={1}
          placeholder="Can I play cricket at 6 PM in Guntur?"
          onChange={(event) => {
            setValue(event.target.value);
            const node = event.target as HTMLTextAreaElement;
            node.style.height = "auto";
            node.style.height = `${Math.min(node.scrollHeight, 160)}px`;
          }}
          onKeyDown={(event) => {
            if (event.key === "Enter" && !event.shiftKey) {
              event.preventDefault();
              send(false);
            }
          }}
          className="max-h-40 w-full resize-none bg-transparent px-3 py-2 text-sm outline-none placeholder:text-muted-foreground"
        />
        <div className="flex items-center justify-between gap-2 px-1 pt-1">
          <div className="flex min-w-0 items-center gap-1.5">
            {items.length > 1 && selected ? (
              <div className="relative">
                <AIInputPillButton
                  icon={selected.icon}
                  isActive={open}
                  showChevron
                  chevronRotated={open}
                  onClick={() => setOpen((current) => !current)}
                  className="!py-1.5 text-xs"
                >
                  {selected.label}
                </AIInputPillButton>
                <AIInputDropdown
                  isOpen={open}
                  onClose={() => setOpen(false)}
                  items={items}
                  className="w-72"
                  renderItem={(item) => (
                    <button
                      key={item.id}
                      onClick={() => {
                        onParserChange(item.id);
                        setOpen(false);
                      }}
                      className={cn(
                        "flex w-full items-start gap-2 rounded-xl px-3 py-2 text-left transition-colors hover:bg-muted",
                        item.id === parser && "bg-muted",
                      )}
                    >
                      <item.icon className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
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
            ) : (
              <span className="px-2 text-[11px] text-muted-foreground">
                {selected ? selected.label : "no parser reachable"}
              </span>
            )}
            <button
              type="button"
              onClick={() => send(true)}
              disabled={!value.trim() || busy || available.length < 2}
              title={
                available.length < 2
                  ? "Start the Rasa container to compare: docker compose --profile rasa up -d"
                  : "Read the same sentence with both parsers, side by side"
              }
              className="rounded-lg border px-2 py-1 text-[11px] font-medium text-muted-foreground transition-colors hover:text-foreground disabled:opacity-40"
            >
              Compare both
            </button>
          </div>
          <div className="flex items-center gap-2">
          <MicButton value={value} onChange={setValue} disabled={busy} />
          <button
            type="button"
            onClick={() => send(false)}
            disabled={!value.trim() || busy}
            className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-primary text-primary-foreground transition-opacity disabled:opacity-40"
            aria-label="Send"
          >
            {busy ? <LoaderCircleIcon size={15} isAnimated /> : <ArrowUp className="h-4 w-4" />}
          </button>
          </div>
        </div>
      </div>
    </div>
  );
}
