# Streaming STT (Nemotron 3.5 ASR, CPU/GPU)

`docker compose up -d --build` → WebSocket ASR on `ws://localhost:2700`.

Send binary frames of **16 kHz mono PCM s16le** (~125 ms each); get `{"partial": ...}` roughly
per word while you talk, then `{"text": ...}` after the text frame `{"eof":1}`. The partial
always carries the whole transcript so far.

- Model: [`nvidia/nemotron-3.5-asr-streaming-0.6b`](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
  (OpenMDW-1.1), pulled on first start into the `hf-cache` volume — ~2.4 GB, so the first `up`
  is slow. Cache-aware streaming FastConformer/RNNT: it keeps encoder state between fixed
  chunks, so there is no silence detector here and no waiting for a pause.
- **Runs at real time on CPU.** Measured on an M5 Pro: 15.1 s of speech, final result at 15.4 s,
  first words back ~1 s in, ~470 MB resident and about 2 cores while decoding. Image is 2.7 GB.
  On a CUDA host it picks the GPU up automatically (rebuild with
  `--build-arg TORCH_INDEX=https://pypi.org/simple` for the CUDA torch build).
- `GET /health` on the same port returns `{"status": "ok"|"loading", "model", "device"}`, and
  503s until the weights are in. It is served beside the socket on purpose: a page that can
  reach it can reach the socket, which a check on the backend could not honestly promise.
  The mic button polls it and stays disabled, with the reason on hover, until it says ok.
- Tuning (env vars): `LANGUAGE` (a locale like `en-US`, `hi-IN`, `te-IN`, or `auto` to detect
  per utterance — 40 locales), `LOOKAHEAD_TOKENS` (higher = more accurate, slower to respond;
  6 ≈ 560 ms), `WORKERS` (concurrent dictations), `MODEL_PATH`, `DEVICE`, `DTYPE`.
- Test: `docker compose exec stt python /app/test_stream.py` checks the session buffer.
  `python client_example.py sample.wav` streams real audio and prints partials as they land —
  make the WAV with the container's ffmpeg:
  `docker compose exec stt ffmpeg -i in.mp3 -ar 16000 -ac 1 sample.wav`.

## Frontend

The mic button in both composers (`components/mic-button.tsx` -> `lib/use-dictation.ts`)
streams to this service and draws a live waveform (`components/waveform.tsx`) while recording.
Point it somewhere else with `NEXT_PUBLIC_STT_URL`; unset it and the page uses
`ws://<its own hostname>:2700`. A page served over HTTPS needs `wss://` (and the mic needs a
secure origin or localhost).
