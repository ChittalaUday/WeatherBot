"use client";

import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * The answer, as the wording layer wrote it.
 *
 * A turn with several findings comes back laid out - a lead sentence and bullets - and one
 * with a single figure comes back as a sentence. The backend says which in `summary_format`,
 * because only what the model wrote can be markdown: the deterministic fallback sentence
 * never is, and parsing it as one would turn a stray "-" into a bullet.
 *
 * No raw HTML is enabled, so nothing the model writes can render as markup.
 */
export function AnswerText({ text, format, className = "" }: {
  text: string;
  format?: string;
  className?: string;
}) {
  if (format !== "markdown") {
    return <p className={`text-sm leading-relaxed ${className}`}>{text}</p>;
  }
  return (
    <div className={`space-y-2 text-sm leading-relaxed ${className}`}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="leading-relaxed">{children}</p>,
          ul: ({ children }) => <ul className="space-y-1 pl-1">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => (
            <li className="flex gap-2">
              <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-current opacity-40" />
              <span className="min-w-0">{children}</span>
            </li>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold tabular-nums">{children}</strong>
          ),
          // A heading is not something the prompt asks for; if one arrives, it reads as bold
          // text rather than punching a title into the middle of a chat bubble.
          h1: ({ children }) => <p className="font-semibold">{children}</p>,
          h2: ({ children }) => <p className="font-semibold">{children}</p>,
          h3: ({ children }) => <p className="font-semibold">{children}</p>,
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">{children}</table>
            </div>
          ),
          th: ({ children }) => <th className="pr-3 text-left font-medium">{children}</th>,
          td: ({ children }) => <td className="pr-3 tabular-nums">{children}</td>,
          a: ({ children }) => <span>{children}</span>,
        }}
      >
        {text}
      </Markdown>
    </div>
  );
}
