"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage, ServerEvent } from "@/lib/types";
import { apiUrl } from "@/lib/utils";

/**
 * One conversation over plain HTTP. Each turn is a POST whose response is a stream of
 * server-sent events, read to completion and then closed.
 *
 * There is no socket to reconnect, no ping, and no state that outlives a turn: a request
 * either completes or fails, and a failure is one error bubble rather than a chat that
 * silently stops answering. The chat id is what carries the conversation, and it lives in
 * localStorage, so a reload resumes it server-side.
 */

const CHAT_KEY = "weathersnap.chat_id";

let counter = 0;
const nextId = () => `m${++counter}`;

/** The two transient messages - the thinking line and the half-written answer. Anything that
 *  ends a turn drops both, so neither can be left on screen under a finished answer. */
const live = (m: ChatMessage) => m.role !== "status" && m.role !== "streaming";

/** Everything that ends a turn. Exactly one of these arrives per request. */
const TERMINAL = new Set(["result", "chat", "clarify", "need_location", "confirm_location",
                          "error", "compare_done"]);

function storedChatId(): string {
  if (typeof window === "undefined") return "";
  return window.localStorage.getItem(CHAT_KEY) ?? "";
}

/** Read a `text/event-stream` body, yielding one parsed event per `data:` line. */
async function* readEvents(response: Response): AsyncGenerator<ServerEvent> {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // events are separated by a blank line; the tail is a partial event, kept for next read
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";
    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;
      try {
        yield JSON.parse(line.slice(5).trim()) as ServerEvent;
      } catch {
        // a truncated frame is not worth killing the turn over - the terminal event repeats
        // everything a client needs anyway
      }
    }
  }
}

export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  // Read at first render, not in an effect: `openChat` used to set it synchronously from the
  // mount effect, which is a cascading render and the one the lint rule is about.
  const [chatId, setChatId] = useState<string>(() => storedChatId() ?? "");
  const inflight = useRef<AbortController | null>(null);

  const push = useCallback((message: ChatMessage) => {
    setMessages((current) => [...current, message]);
  }, []);

  /** Reopen a past conversation: its answers are replayed from what was stored, not
   *  re-queried, so an old chat shows the forecast it actually gave.
   *
   *  The id is adopted before the history is fetched, so a backend that is briefly down
   *  costs an empty transcript rather than a new conversation the server does not know. */
  /** Pull one stored conversation back into the transcript. No state is set before the await,
   *  so this is safe to call from an effect. */
  const restoreChat = useCallback(async (id: string) => {
    const response = await fetch(`${apiUrl()}/api/chats/${id}`).catch(() => null);
    if (!response?.ok) return;
    const history = await response.json();

    const restored: ChatMessage[] = [];
    for (const turn of history.turns ?? []) {
      restored.push({ id: nextId(), role: "user", text: turn.text });
      if (turn.payload) {
        restored.push({ id: nextId(), role: "assistant", result: turn.payload });
      } else if (turn.outcome === "clarified" || turn.outcome === "need_location") {
        restored.push({ id: nextId(), role: "error", message: turn.detail || turn.outcome });
      } else if (turn.outcome === "error") {
        restored.push({ id: nextId(), role: "error", message: turn.detail || "failed" });
      }
    }
    setMessages(restored);
  }, []);

  const openChat = useCallback(
    (id: string) => {
      window.localStorage.setItem(CHAT_KEY, id);
      setChatId(id);
      return restoreChat(id);
    },
    [restoreChat],
  );

  // Replaying a stored conversation is a fetch, so it belongs in an effect - but only the
  // fetch does. The id itself is already in state above.
  useEffect(() => {
    const existing = storedChatId();
    // The compiler cannot see that every setState in restoreChat happens after an await, so
    // it reads a fetch-on-mount as a cascading render. It is not one.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (existing) void restoreChat(existing);
  }, [restoreChat]);

  /** Fold one server event into the transcript. Shared by chat and compare. */
  const consume = useCallback((data: ServerEvent) => {
    switch (data.type) {
      case "status":
        // one live status line per turn, replaced as the backend advances
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "status", stage: data.stage },
        ]);
        break;
      case "nlu":
        setMessages((current) => [...current, { id: nextId(), role: "nlu", nlu: data }]);
        break;
      // Both channels of the phrasing model grow one message in place: its reasoning, then
      // the answer under it. The status line goes as soon as either starts - words on screen
      // are better progress than a spinner claiming there are none.
      case "thinking":
      case "delta":
        setMessages((current) => {
          const role = data.type === "thinking" ? "thinking" : "streaming";
          const shown = current.filter((m) => m.role !== "status");
          const last = shown[shown.length - 1];
          return last?.role === role
            ? [...shown.slice(0, -1), { ...last, text: last.text + data.text }]
            : [...shown, { id: nextId(), role, text: data.text }];
        });
        break;
      case "confirm_location":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "pick-location", text: data.text, raw: data.raw,
            message: data.message, options: data.options, answered: false },
        ]);
        break;
      case "need_location":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "ask-location", text: data.text, message: data.message, answered: false },
        ]);
        break;
      case "result":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "assistant", result: data },
        ]);
        break;
      case "clarify":
        setMessages((current) => [
          ...current.filter(live),
          { id: nextId(), role: "clarify", message: data.message, text: data.text },
        ]);
        break;
      case "chat":
        // no table, no chart - a greeting, a session command, or a refusal with a reason
        setMessages((current) => [
          ...current.filter(live),
          {
            id: nextId(),
            role: "chat",
            turnId: data.turn_id,
            intent: data.intent,
            family: data.family,
            message: data.message,
            locations: data.locations ?? [],
            metrics: data.metrics,
          },
        ]);
        break;
      case "compare_start":
        // lay the columns out immediately; each fills in as its model finishes
        setMessages((current) => [
          ...current.filter(live),
          {
            id: nextId(),
            role: "compare",
            text: data.text,
            models: data.models.map((m) => ({ ...m, ok: false, latency_ms: 0 })),
            disagreements: [],
            totalMs: 0,
            pending: data.models.length,
          },
        ]);
        break;
      case "compare_result":
      case "compare_done":
        setMessages((current) => {
          const last = [...current].reverse().find((m) => m.role === "compare");
          if (!last) return current;
          return current.map((m) => {
            if (m.id !== last.id || m.role !== "compare") return m;
            if (data.type === "compare_done") {
              return { ...m, disagreements: data.disagreements, totalMs: data.total_ms, pending: 0 };
            }
            return {
              ...m,
              pending: Math.max(m.pending - 1, 0),
              totalMs: data.elapsed_ms,
              models: m.models.map((c) => (c.version === data.version ? { ...c, ...data } : c)),
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

  /** POST one turn and drain its stream. `busy` clears when the stream ends, whatever ended it. */
  const stream = useCallback(
    async (path: string, body: Record<string, unknown>) => {
      inflight.current?.abort();
      const controller = new AbortController();
      inflight.current = controller;
      setBusy(true);
      let sawTerminal = false;
      try {
        const response = await fetch(`${apiUrl()}${path}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        for await (const event of readEvents(response)) {
          if (TERMINAL.has(event.type)) sawTerminal = true;
          consume(event);
        }
        // a stream that ended without saying how is still a turn the user is waiting on
        if (!sawTerminal) {
          consume({ type: "error", message: "The answer stopped halfway. Try that again." });
        }
      } catch (error) {
        if ((error as Error).name === "AbortError") return;
        consume({
          type: "error",
          message: "I could not reach the server. Check it is running and try again.",
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

  const ensureChatId = useCallback(() => {
    const id = chatId || storedChatId() || `chat-${Math.random().toString(36).slice(2, 12)}`;
    if (id !== chatId) {
      window.localStorage.setItem(CHAT_KEY, id);
      setChatId(id);
    }
    return id;
  }, [chatId]);

  const ask = useCallback(
    (text: string, model?: string) => {
      if (!text.trim()) return;
      push({ id: nextId(), role: "user", text });
      stream("/api/chat", { text, model, chat_id: ensureChatId() });
    },
    [push, stream, ensureChatId],
  );

  /** Same sentence, every model at once - a comparison of understanding, not of answers. */
  const compare = useCallback(
    (text: string) => {
      if (!text.trim()) return;
      push({ id: nextId(), role: "user", text });
      stream("/api/compare", { text, chat_id: ensureChatId() });
    },
    [push, stream, ensureChatId],
  );

  /** Answer a need_location prompt with browser coordinates and rerun the pending question. */
  const sendLocation = useCallback(
    (text: string, lat: number, lon: number, model?: string) => {
      setMessages((current) =>
        current.map((m) =>
          m.role === "ask-location" && m.text === text ? { ...m, answered: true } : m,
        ),
      );
      stream("/api/chat", { text, lat, lon, model, chat_id: ensureChatId() });
    },
    [stream, ensureChatId],
  );

  /**
   * Answer a confirm_location prompt. The chosen place is qualified into the original
   * sentence - "rain in Angara" becomes "rain in Angara, Jharkhand" - because the resolver
   * already splits a name from its qualifier, so this needs no second endpoint and no place
   * id threaded through the turn.
   */
  const pickLocation = useCallback(
    (text: string, raw: string, option: { name: string; state: string }, model?: string) => {
      setMessages((current) =>
        current.map((m) =>
          m.role === "pick-location" && m.text === text ? { ...m, answered: true } : m,
        ),
      );
      const qualified = `${option.name}, ${option.state}`;
      const rewritten = text.replace(raw, qualified);
      push({ id: nextId(), role: "user", text: qualified });
      stream("/api/chat", { text: rewritten, model, chat_id: ensureChatId() });
    },
    [push, stream, ensureChatId],
  );

  /** Drop the server-side slots and start a fresh conversation. */
  const newChat = useCallback(async () => {
    inflight.current?.abort();
    setMessages([]);
    const response = await fetch(`${apiUrl()}/api/chat/reset`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "", chat_id: storedChatId() }),
    }).catch(() => null);
    const created =
      (await response?.json().catch(() => null))?.chat_id ??
      `chat-${Math.random().toString(36).slice(2, 12)}`;
    window.localStorage.setItem(CHAT_KEY, created);
    setChatId(created);
  }, []);

  return { busy, messages, chatId, ask, compare, sendLocation, pickLocation, newChat,
    openChat };
}
