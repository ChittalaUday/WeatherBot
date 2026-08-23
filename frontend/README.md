# WeatherSnap frontend

Next.js chat UI. The backend is a sibling of this directory — see
[../README.md](../README.md) to run both together with `./scripts/run_app.sh`.

```bash
cp .env.example .env.local     # optional: only needed to point at a non-local backend
npm install
npm run dev                    # http://localhost:3001
```

## How a turn works

`lib/use-chat.ts` is the whole transport. A turn is one `POST /api/chat` whose response is a
stream of server-sent events, read to completion and then closed — there is no socket, no
reconnect loop and no state that outlives a turn. The chat id lives in `localStorage`, and it
is what carries the conversation, so a reload resumes it server-side.

```text
POST /api/chat            {"text": "...", "chat_id": "chat-1"}
  <- status               understanding -> locating -> fetching -> writing
  <- nlu                  what the model read (the debug strip)
  <- thinking / delta     the reply being written, streamed
  <- result | chat | clarify | need_location | error     exactly one, and it ends the turn
```

`busy` clears when the stream ends, whatever ended it. A stream that stops without a terminal
event is reported as an error rather than left spinning.

## Files worth knowing

| File | Role |
| :--- | :--- |
| `lib/use-chat.ts` | POST + SSE reader, the transcript reducer, the chat id |
| `lib/types.ts` | the wire format — the one place it is written down |
| `lib/utils.ts` | `apiUrl()`, the single definition of where the backend is |
| `components/messages.tsx` | the transcript, ratings, the correction entry point |
| `components/compare.tsx` | three models on one sentence, side by side |
| `components/composer.tsx` | input + model pill, fed by `/api/models` |

Do not inline `process.env.NEXT_PUBLIC_API_URL` in a component. Eight files used to, only one
of them had the LAN fallback, and the app broke the moment it was opened from a phone. Import
`apiUrl()`.
