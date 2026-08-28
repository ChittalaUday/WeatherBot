"use client";

import { useCallback, useRef, useState, useSyncExternalStore } from "react";
import { readEvents, v2Url } from "@/lib/v2/api";
import type { Candidate, ChatMessage, Parser, ServerEvent } from "@/lib/v2/types";

/**
 * One conversation with Backend-v2 over plain HTTP: each turn is a POST whose response is a
 * stream of server-sent events, read to completion and closed.
 *
 * Its own chat id, under its own localStorage key, so the v1 and v2 chats never adopt each
 * other's conversation - the two backends have separate databases and a shared id would make
 * one of them replay history it does not have.
 *
 * There are no `delta`/`thinking` events to handle: v2's renderer is deterministic and returns
 * the whole answer at once, so there is no half-written state to keep on screen.
 */

const CHAT_KEY = "weathersnap.v2.chat_id";
const PARSER_KEY = "weathersnap.v2.parser";

let counter = 0;
const nextId = () => `v2-${++counter}`;

/** The transient messages: the status line and the half-written answer. Anything that ends a
 *  turn drops both, so neither can be left on screen underneath a finished answer. The thinking
 *  block is kept - it is a record of how the answer was reached. */
const live = (m: ChatMessage) => m.role !== "status" && m.role !== "streaming";
const TERMINAL = new Set(["result", "chat", "clarify", "need_location", "error", "compare_done"]);

function stored(key: string): string {
  if (typeof window === "undefined") return "";
  try {
    return window.localStorage.getItem(key) ?? "";
  } catch {
    return "";                          // private windows throw rather than return null
  }
}

function save(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // A remembered chat id is a convenience. Losing it is not worth failing a render over,
    // and in a private window the setter itself throws.
  }
}

/** localStorage never notifies this tab, so there is nothing to subscribe to. */
const noSubscribe = () => () => {};

/**
 * Read a persisted value without an effect.
 *
 * `useSyncExternalStore` is the right primitive: localStorage cannot be read during SSR, and a
 * lazy `useState` initializer would therefore disagree with the first client render. The server
 * snapshot is empty, the client snapshot is the stored value, and React reconciles the two
 * without a cascading render. Writes set the override, because nothing re-notifies this hook.
 */
function usePersisted(key: string): [string, (value: string) => void] {
  const persisted = useSyncExternalStore(noSubscribe, () => stored(key), () => "");
  const [override, setOverride] = useState<string | null>(null);
  const set = useCallback(
    (value: string) => {
      save(key, value);
      setOverride(value);
    },
    [key],
  );
  return [override ?? persisted, set];
}

export function useChatV2() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [chatId, setChatId] = usePersisted(CHAT_KEY);
  const [storedParser, setStoredParser] = usePersisted(PARSER_KEY);
  const parser: Parser = storedParser === "rasa" ? "rasa" : "rules";
  const inflight = useRef<AbortController | null>(null);

  const chooseParser = useCallback(
    (next: Parser) => setStoredParser(next),
    [setStoredParser],
  );

  const ensureChatId = useCallback(() => {
    const id = chatId || `chat-${Math.random().toString(36).slice(2, 12)}`;
    if (id !== chatId) setChatId(id);
    return id;
  }, [chatId, setChatId]);

  /** Fold one server event into the transcript. */
  const consume = useCallback((data: ServerEvent) => {
    switch (data.type) {
      case "status":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "status", stage: data.stage },
        ]);
        break;
      case "nlu":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "nlu", nlu: data },
          { id: nextId(), role: "status", stage: "locating" },
        ]);
        break;
      // The model's reasoning and its answer grow in place as they arrive. The status line goes
      // as soon as either starts: words on screen are better progress than a spinner claiming
      // there are none.
      case "thinking":
      case "delta": {
        const role = data.type === "thinking" ? "thinking" : "streaming";
        setMessages((current) => {
          const shown = current.filter((m) => m.role !== "status");
          const last = shown[shown.length - 1];
          if (last?.role === role) {
            return [...shown.slice(0, -1), { ...last, text: last.text + data.text }];
          }
          return [
            ...shown,
            role === "thinking"
              ? { id: nextId(), role, text: data.text, done: false }
              : { id: nextId(), role, text: data.text },
          ];
        });
        break;
      }
      case "thinking_done":
        setMessages((current) =>
          current.map((m) => (m.role === "thinking" ? { ...m, done: true } : m)),
        );
        break;
      case "delta_reset":
        // The wording failed verification. Drop it rather than leaving a rejected sentence on
        // screen next to the one that replaced it.
        setMessages((current) => current.filter((m) => m.role !== "streaming"));
        break;
      case "result":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "answer", result: data },
        ]);
        break;
      case "chat":
        setMessages((current) => [
          ...current.filter(live),
          {
            id: nextId(),
            role: "chat",
            turnId: data.turn_id,
            family: data.family,
            message: data.message,
          },
        ]);
        break;
      case "clarify":
        setMessages((current) => [
          ...current.filter(live),
          {
            id: nextId(),
            role: "clarify",
            message: data.message,
            text: data.text,
            reason: data.reason,
            candidates: data.candidates ?? [],
          },
        ]);
        break;
      case "need_location":
        setMessages((current) => [
          ...current.filter(live),
          {
            id: nextId(),
            role: "ask-location",
            text: data.text,
            message: data.message,
            answered: false,
          },
        ]);
        break;
      case "compare_start":
        setMessages((current) => [
          ...current.filter(live),
          {
            id: nextId(),
            role: "compare",
            text: data.text,
            columns: data.models.map((m) => ({ ...m, ok: false, latency_ms: 0 })),
            disagreements: [],
            agreed: false,
            pending: data.models.length,
            totalMs: 0,
          },
        ]);
        break;
      case "compare_result":
      case "compare_done":
        setMessages((current) => {
          const last = [...current].reverse().find((m) => m.role === "compare");
          if (!last) return current;
          return current.filter(live).map((m) => {
            if (m.id !== last.id || m.role !== "compare") return m;
            if (data.type === "compare_done") {
              return {
                ...m,
                disagreements: data.disagreements,
                agreed: data.agreed,
                totalMs: data.total_ms,
                pending: 0,
              };
            }
            return {
              ...m,
              pending: Math.max(m.pending - 1, 0),
              columns: m.columns.map((c) =>
                c.version === data.version ? { ...c, ...data } : c,
              ),
            };
          });
        });
        break;
      case "error":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "error", message: data.message },
        ]);
        break;
    }
  }, []);

  /** POST one turn and drain its stream. `busy` clears when the stream ends, however it ended. */
  const stream = useCallback(
    async (path: string, body: Record<string, unknown>) => {
      inflight.current?.abort();
      const controller = new AbortController();
      inflight.current = controller;
      setBusy(true);
      let sawTerminal = false;
      try {
        const response = await fetch(`${v2Url()}${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        for await (const event of readEvents<ServerEvent>(response)) {
          if (TERMINAL.has(event.type)) sawTerminal = true;
          consume(event);
        }
        if (!sawTerminal) {
          consume({ type: "error", message: "The answer stopped halfway. Try that again." });
        }
      } catch (error) {
        if ((error as Error).name === "AbortError") return;
        consume({
          type: "error",
          message: `Could not reach the v2 backend at ${v2Url()}. Check it is running: docker compose up -d`,
        });
      } finally {
        if (inflight.current === controller) {
          inflight.current = null;
          setBusy(false);
        }
      }
    },
    [consume],
  );

  const ask = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      setMessages((current) => [...current, { id: nextId(), role: "user", text }]);
      stream("/api/chat", { text, model: parser, chat_id: ensureChatId() });
    },
    [stream, parser, ensureChatId],
  );

  /** The same sentence through both parsers, with the shared engine behind each. */
  const compare = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      setMessages((current) => [...current, { id: nextId(), role: "user", text }]);
      stream("/api/compare", { text, chat_id: ensureChatId() });
    },
    [stream, ensureChatId],
  );

  /** Answer a need_location prompt with browser coordinates and rerun the pending question. */
  const sendLocation = useCallback(
    (text: string, lat: number, lon: number) => {
      setMessages((current) =>
        current.map((m) =>
          m.role === "ask-location" && m.text === text ? { ...m, answered: true } : m,
        ),
      );
      stream("/api/chat", { text, lat, lon, model: parser, chat_id: ensureChatId() });
    },
    [stream, parser, ensureChatId],
  );

  /**
   * Answer an ambiguous-place question by picking one of the offered places.
   *
   * The candidate's coordinates are sent, not a rewritten sentence. Re-asking as
   * "rainfall in Angara Ranchi" would send the answer back through name resolution - the exact
   * step that could not decide in the first place - and the reply shows which place was used,
   * so nothing is hidden by skipping it.
   */
  const pickPlace = useCallback(
    (originalText: string, candidate: Candidate) => {
      const where = [candidate.name, candidate.district].filter(Boolean).join(", ");
      setMessages((current) => [
        ...current,
        { id: nextId(), role: "user", text: `${originalText} — ${where}` },
      ]);
      stream("/api/chat", {
        text: originalText,
        lat: candidate.lat,
        lon: candidate.lon,
        model: parser,
        chat_id: ensureChatId(),
      });
    },
    [stream, parser, ensureChatId],
  );

  /** Drop the server-side slots and start a fresh conversation. */
  const newChat = useCallback(async () => {
    inflight.current?.abort();
    setMessages([]);
    const response = await fetch(`${v2Url()}/api/chat/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "", chat_id: stored(CHAT_KEY) }),
    }).catch(() => null);
    const created =
      (await response?.json().catch(() => null))?.chat_id ??
      `chat-${Math.random().toString(36).slice(2, 12)}`;
    setChatId(created);
  }, [setChatId]);

  /** Reopen a past conversation: answers are replayed as they were rendered, not re-queried. */
  const openChat = useCallback(async (id: string) => {
    setChatId(id);
    const response = await fetch(`${v2Url()}/api/chats/${id}`).catch(() => null);
    if (!response?.ok) return;
    const history = await response.json();
    const restored: ChatMessage[] = [];
    for (const turn of history.turns ?? []) {
      restored.push({ id: nextId(), role: "user", text: turn.text });
      if (turn.payload) {
        restored.push({ id: nextId(), role: "answer", result: turn.payload });
      } else if (turn.detail) {
        restored.push({
          id: nextId(),
          role: turn.outcome === "declined" ? "chat" : "clarify",
          ...(turn.outcome === "declined"
            ? { turnId: turn.turn_id, family: "declined" as const, message: turn.detail }
            : { message: turn.detail, text: turn.text, candidates: [] }),
        } as ChatMessage);
      }
    }
    setMessages(restored);
  }, [setChatId]);

  return {
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
    openChat,
  };
}
