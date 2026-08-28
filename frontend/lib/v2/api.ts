/**
 * Where the v2 backend is.
 *
 * Separate from `lib/utils.ts::apiUrl()` on purpose: that one points at the v1 backend on 8787
 * and the two run side by side, so this file never touches it. Nothing here reads
 * NEXT_PUBLIC_API_URL.
 *
 * Defaults to the page's own host on 8788, so opening the app from a phone on the same network
 * works with no configuration. Set NEXT_PUBLIC_V2_API_URL to override.
 */

export function v2Url(): string {
  const configured = process.env.NEXT_PUBLIC_V2_API_URL;
  if (typeof window === "undefined") return configured ?? "http://127.0.0.1:8788";
  const loopback = /127\.0\.0\.1|localhost/;
  // A configured loopback URL in a build being reached over the network is a developer
  // default, not an instruction: the page's own hostname is the answer.
  if (configured && !loopback.test(configured)) return configured;
  if (loopback.test(window.location.hostname)) return configured ?? "http://127.0.0.1:8788";
  return `${window.location.protocol}//${window.location.hostname}:8788`;
}

/** Read a `text/event-stream` body, yielding one parsed event per `data:` frame. */
export async function* readEvents<T>(response: Response): AsyncGenerator<T> {
  const reader = response.body?.getReader();
  if (!reader) return;
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";          // the tail is a partial frame, kept for the next read
    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;
      try {
        yield JSON.parse(line.slice(5).trim()) as T;
      } catch {
        // A truncated frame is not worth killing the turn over: the terminal event repeats
        // everything the client needs.
      }
    }
  }
}
