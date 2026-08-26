"use client";

import { Brain, ChevronDown } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

/**
 * The model's reasoning, live.
 *
 * On its own channel and visibly separated from the answer, because it is not the answer: it is
 * a half-formed argument the model is having with itself, and presenting it as a conclusion
 * would be worse than not showing it at all.
 *
 * Open and scrolling while the model works, collapsed to one line the moment the answer starts.
 */
export function Thinking({ text, done }: { text: string; done: boolean }) {
  const [open, setOpen] = useState(true);
  const body = useRef<HTMLDivElement>(null);
  // Collapse on its own once the thinking is finished - it has served its purpose by then, and
  // it should not push the answer off the screen.
  const collapsed = useRef(false);

  useEffect(() => {
    if (done && !collapsed.current) {
      collapsed.current = true;
      setOpen(false);
    }
  }, [done]);

  useEffect(() => {
    if (open && body.current) body.current.scrollTop = body.current.scrollHeight;
  }, [text, open]);

  const words = text.trim().split(/\s+/).filter(Boolean).length;

  return (
    <div className="rounded-2xl border border-dashed bg-muted/30">
      <button
        type="button"
        onClick={() => setOpen((on) => !on)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-[11px] text-muted-foreground hover:text-foreground"
      >
        <Brain className={cn("h-3.5 w-3.5", !done && "animate-pulse text-primary")} />
        <span className="flex-1">
          {done ? `Thought for ${words} words` : "Thinking…"}
        </span>
        <ChevronDown className={cn("h-3 w-3 transition-transform", open && "rotate-180")} />
      </button>
      {open && (
        <div
          ref={body}
          className="max-h-44 overflow-y-auto px-3 pb-2.5 text-[11px] leading-relaxed text-muted-foreground"
        >
          <p className="whitespace-pre-wrap">{text}</p>
        </div>
      )}
    </div>
  );
}
