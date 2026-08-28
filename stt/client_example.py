"""Smoke test: stream a 16kHz mono WAV at real-ish time, print results as they arrive.

    python client_example.py sample.wav

Sending and receiving run separately: partials come back while the audio is still going out.
"""
import asyncio
import json
import sys
import time
import wave

import websockets

CHUNK = 4000  # 125ms @ 16kHz s16le


async def main(path):
    wf = wave.open(path, "rb")
    assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000), \
        "need 16kHz mono 16-bit WAV (docker compose exec stt ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav)"
    start = time.time()
    async with websockets.connect("ws://localhost:2700", max_size=None) as ws:

        async def send():
            while data := wf.readframes(CHUNK):
                await ws.send(data)
                await asyncio.sleep(CHUNK / 16000)  # pace it like a live mic
            await ws.send('{"eof":1}')

        async def receive():
            async for message in ws:
                reply = json.loads(message)
                kind = "FINAL  " if "text" in reply else "partial"
                print(f"[{time.time() - start:5.1f}s] {kind} {reply.get('text') or reply['partial']}")

        await asyncio.gather(send(), receive())


asyncio.run(main(sys.argv[1]))
