"""Streaming ASR over WebSocket, backed by Nemotron 3.5 ASR. Binary frames in = 16kHz mono PCM s16le.

Replies with JSON: {"partial": "..."} as words are recognised, then {"text": "..."} once the
client sends '{"eof":1}'. The partial always carries the whole transcript so far.

GET /health on the same port reports whether the model has finished loading, so a page can
check the exact origin it is about to dictate to.

This is a cache-aware streaming model: it takes fixed-size chunks and keeps its encoder state
between them, so there is no silence detector here and no waiting for a pause - text comes
back about every 600 ms while you are still talking.
"""
import asyncio
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from http import HTTPStatus

import numpy as np
import torch
import websockets
from transformers import AutoModelForRNNT, AutoProcessor, TextIteratorStreamer

MODEL_PATH = os.getenv("MODEL_PATH", "nvidia/nemotron-3.5-asr-streaming-0.6b")
LANGUAGE = os.getenv("LANGUAGE", "en-US")  # a locale, or "auto" to detect per utterance
DEVICE = os.getenv("DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu")
DTYPE = getattr(torch, os.getenv("DTYPE", "float32"))
# How far the model may look ahead before emitting, in tokens: higher is more accurate and
# slower to respond. 6 is the value NVIDIA's own streaming example uses (~560 ms).
LOOKAHEAD = int(os.getenv("LOOKAHEAD_TOKENS", "6"))

# ponytail: a thread per live session, so this is the concurrent-dictation cap, not a
# queue depth. Move to a batched server if more than a couple of people talk at once.
WORKERS = int(os.getenv("WORKERS", "2"))
pool = ThreadPoolExecutor(max_workers=WORKERS)

ready = False  # flipped once the weights are in memory; reported by /health


@lru_cache(maxsize=1)
def get_model():
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    processor.set_num_lookahead_tokens(LOOKAHEAD)
    model = AutoModelForRNNT.from_pretrained(MODEL_PATH, dtype=DTYPE).to(DEVICE).eval()
    return processor, model


class Stream:
    """One session's audio, growing as frames arrive.

    The model pulls fixed, slightly overlapping windows out of it and blocks when the
    speaker has not said enough yet - which is what makes this live rather than batch.
    Windows are indexed against the whole session, so the buffer is kept, not consumed.
    """

    def __init__(self):
        self.audio = np.zeros(0, dtype=np.float32)
        self.closed = False
        self.grew = threading.Condition()

    def feed(self, frame: bytes) -> None:
        chunk = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        with self.grew:
            self.audio = np.concatenate([self.audio, chunk])
            self.grew.notify_all()

    def close(self) -> None:
        with self.grew:
            self.closed = True
            self.grew.notify_all()

    def window(self, start: int, end: int):
        """Samples [start:end], or None if the speaker stopped before saying that much."""
        with self.grew:
            self.grew.wait_for(lambda: self.closed or self.audio.shape[0] >= end)
            return self.audio[start:end] if self.audio.shape[0] >= end else None


def transcribe(stream: Stream, emit) -> None:
    """Blocking: drive the model over `stream`, calling emit() with each piece of text."""
    processor, model = get_model()
    rate = processor.feature_extractor.sampling_rate
    hop = processor.feature_extractor.hop_length
    pad = processor.feature_extractor.n_fft // 2

    def encode(samples, first: bool):
        inputs = processor(samples, sampling_rate=rate, is_streaming=True,
                           is_first_audio_chunk=first, language=LANGUAGE, return_tensors="pt")
        return inputs.to(model.device, dtype=model.dtype)

    opening = stream.window(0, processor.num_samples_first_audio_chunk)
    if opening is None:
        return  # they stopped before the model had enough to start on
    first = encode(opening, True)

    def features():
        yield first.input_features[:, : processor.num_mel_frames_first_audio_chunk, :]
        # Windows are placed by mel frame and reach back half an FFT, so they overlap
        # slightly; stepping by samples instead would drift out of alignment.
        mel = processor.num_mel_frames_first_audio_chunk
        while True:
            start = mel * hop - pad
            samples = stream.window(start, start + processor.num_samples_per_audio_chunk)
            if samples is None:
                return
            yield encode(samples, False).input_features
            mel += processor.num_mel_frames_per_audio_chunk

    streamer = TextIteratorStreamer(processor.tokenizer, skip_special_tokens=True)
    worker = threading.Thread(
        target=model.generate,
        kwargs={**first, "input_features": features(), "streamer": streamer},
        daemon=True,
    )
    worker.start()
    for piece in streamer:
        emit(piece)
    worker.join()


async def handle(ws):
    loop = asyncio.get_running_loop()
    stream = Stream()
    pieces: asyncio.Queue = asyncio.Queue()

    def run():
        try:
            transcribe(stream, lambda piece: loop.call_soon_threadsafe(pieces.put_nowait, piece))
        finally:
            loop.call_soon_threadsafe(pieces.put_nowait, None)

    async def feed():
        # Always close the stream: without it the model thread waits on audio that a
        # disconnected client is never going to send.
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    if json.loads(msg).get("eof"):
                        return
                    continue
                stream.feed(msg)
        finally:
            stream.close()

    decoding = loop.run_in_executor(pool, run)
    feeding = asyncio.create_task(feed())
    text, sent = "", ""
    while (piece := await pieces.get()) is not None:
        text += piece
        # The streamer ticks per token, and most ticks add no visible text; sending those
        # too would be a few hundred identical frames per minute for the page to re-render.
        if text.strip() != sent:
            sent = text.strip()
            await ws.send(json.dumps({"partial": sent}))
    await feeding
    await decoding
    if sent:
        await ws.send(json.dumps({"text": sent}))


async def health(path, request_headers):
    """Serve GET /health beside the socket; anything else falls through to the handshake.

    Same port as the WebSocket on purpose: a page that can reach this can reach the socket,
    which a health check sitting on the backend could not honestly promise. The first run
    downloads a couple of GB, so "loading" has to be distinguishable from "down".
    """
    if path.split("?")[0] != "/health":
        return None
    body = json.dumps({
        "status": "ok" if ready else "loading",
        "model": MODEL_PATH,
        "device": DEVICE,
    }).encode()
    headers = [
        ("Content-Type", "application/json"),
        ("Access-Control-Allow-Origin", "*"),  # the page is served from another port
        ("Cache-Control", "no-store"),
    ]
    return (HTTPStatus.OK if ready else HTTPStatus.SERVICE_UNAVAILABLE), headers, body


async def main():
    global ready
    # Listen first and warm in the background, so /health can answer "loading" instead of
    # refusing the connection for as long as the first weight download takes.
    warm = asyncio.get_running_loop().run_in_executor(pool, get_model)
    async with websockets.serve(
        handle, "0.0.0.0", 2700, ping_interval=20, max_size=None, process_request=health
    ):
        await warm
        ready = True
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
