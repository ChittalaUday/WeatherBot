"""Protocol contract the browser hook relies on. `python3 test_stream.py` (needs the container up).

frontend/lib/use-dictation.ts makes three assumptions that are not obvious from the socket
alone, and would each fail silently in the UI - a stuck spinner, or a transcript missing its
last words. This pins them.
"""
import asyncio
import json
import sys
import wave

import websockets

URL = "ws://127.0.0.1:2701/v1/realtime"
WAV = "../stt/sample16k_long.wav"
CHUNK = 4096  # 2048 int16 samples, one browser worklet message


async def main():
    audio = wave.open(WAV)
    assert (audio.getframerate(), audio.getnchannels()) == (16000, 1), "server wants 16k mono"
    pcm = audio.readframes(audio.getnframes())

    async with websockets.connect(URL, max_size=None) as ws:
        assert json.loads(await ws.recv())["type"] == "session.created", "server greets first"

        for i in range(0, len(pcm), CHUNK):
            await ws.send(pcm[i : i + CHUNK])
            await asyncio.sleep(0.064)
        await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))

        deltas, final = "", None
        while final is None:
            event = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
            if event["type"].endswith("transcription.delta"):
                deltas += event["delta"]
            elif event["type"].endswith("transcription.completed"):
                final = event["transcript"]

        # 1. Deltas are incremental fragments, not a cumulative transcript. The hook
        #    concatenates them; assigning each one would leave only the last token.
        assert final.strip().startswith(deltas.strip()), f"deltas {deltas.strip()!r} vs {final!r}"
        assert len(deltas.strip()) > len(final.strip()) / 2, "a cumulative partial would be shorter"

        # 2. The final is not just the deltas joined - it re-punctuates ("well" -> "well.").
        #    So on `completed` the hook replaces the accumulated partial with `transcript`
        #    rather than keeping what it built up.
        assert final.strip() != deltas.strip(), "final must differ, or preferring it is pointless"

        # 3. Words spoken just before the mic is released only arrive after the commit, so
        #    the hook must keep the socket open past the release.
        assert final.strip().endswith("well."), f"tail of the utterance was lost: {final!r}"

        # 4. The server does NOT close after committing - it waits for another utterance.
        #    The hook therefore has to tear the socket down itself; waiting on onclose to
        #    clear `pending` would leave the mic button spinning forever.
        try:
            await asyncio.wait_for(ws.recv(), timeout=3)
        except asyncio.TimeoutError:
            pass  # quiet but open, as expected
        except websockets.ConnectionClosed:
            raise AssertionError("server closed after commit; use-dictation's teardown is now wrong")
        assert ws.state is websockets.protocol.State.OPEN, "socket must still be open"

    print(f"protocol ok: {final.strip()!r}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (OSError, websockets.WebSocketException) as cause:
        sys.exit(f"cannot reach {URL} - is `docker compose up -d` running here? ({cause})")
