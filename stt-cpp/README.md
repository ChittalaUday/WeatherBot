# STT via NeMo-Speech.cpp (native C++, CPU)

The same Nemotron 3.5 ASR model as [`../stt`](../stt), but run through NVIDIA's
native C++ runtime instead of PyTorch/transformers. Lives beside `../stt` rather
than replacing it: this one serves **:2701**, the Python one still serves :2700.

```bash
./setup.sh              # clone the upstream repo + fetch and verify the GGUF
docker compose up -d --build
curl -X POST http://127.0.0.1:2701/v1/audio/transcriptions -F file=@sample16k.wav
```

## Measured on this machine (M-series, Docker arm64, 4 CPU / 4 GB limit)

`python3 ../scripts/stt_bench.py` drives both services with N simulated microphones each
streaming at 1x and reports what a user feels. Numbers below are from a 1/4/8/16/32 sweep.

| | `../stt` (PyTorch) | this (`nemo-speech`) |
|---|---|---|
| image | 2.72 GB | **28.3 MB** |
| cold start to healthy | ~20 min first run (weights download) | **6 s** |
| concurrent live mics under a 2 s tail | **1** | **8** |
| throughput ceiling | 2.1x real time | **5.9x real time** |
| first word, 1 mic | 1.28 s | **0.81 s** |
| wait after you stop, 8 mics | 9.5 s (p50) / 16.2 s (p95) | **1.8 s** |
| CPU at saturation | 420% | 305% |
| resident | ~570 MB | ~1.1 GB (includes the mmap'd GGUF) |

Same weights, so accuracy should match; only the runtime differs. Per 4-core container,
budget **~8 concurrent dictations**; past that the tail grows roughly linearly (7 s at 16,
14 s at 32) because the CPU, not the queue, is the limit.

Both services queue rather than fail under overload, and each hides it somewhere different -
worth knowing before reading a graph and concluding everything is fine:

- `../stt` accepts every connection instantly, then queues in a
  `ThreadPoolExecutor(max_workers=WORKERS)` that defaults to **2**. Overload shows up as
  time-to-first-word, not as connection errors.
- `nemo-speech serve --threads` defaults to **4**, and a realtime WebSocket holds one for
  the whole dictation, so the 5th mic waits at accept. That looked like a *perfect* service
  on every latency metric - because those are timed from the first audio frame, which had
  not happened yet. This compose file sets `--threads 32`; without it, 16 clients saw 15 s
  connect waits while reporting a 0.3 s tail.

## What is in this folder

- `setup.sh` — clones [NVIDIA/NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp)
  into `NeMo-Speech.cpp/` and downloads `models/*.gguf`. Both are gitignored, so
  this script is the only thing that reproduces them. It is idempotent and
  re-verifies the model SHA-256 on every run.
- `docker-compose.yml` — builds upstream's own `docker/Dockerfile` (`runtime`
  target). Nothing upstream is patched or vendored; every choice is a build arg.

`setup.sh` disables the git-lfs filters when cloning. The only LFS files in the
repo are the Mandarin TTS tokenizer tables, which this build does not compile,
and without the overrides `git checkout` aborts outright on a machine that has
no `git-lfs` installed.

## Build arguments, and why

CPU-only: `ENABLE_CUDA=OFF`, and with it `ENABLE_GGML_PATCHES=OFF` — the patch
series in `ggml-patches/` fuses CUDA kernels, so a CPU build wants stock ggml.
Off as unused: `ENABLE_GRPC` (Riva), `ENABLE_NORM` (Sparrowhawk/OpenFST — the
single slowest thing in the builder), `ENABLE_NMT`, `ENABLE_TTS_JA/ZH`.

`ENABLE_FLASHLIGHT=ON` is the one that is not about features. Nothing here uses
LM-fused decoding, but the Dockerfile runs `build_sentencepiece_static.sh` only
inside that branch, and `src/asr/CMakeLists.txt:101` does an unconditional
`find_library(SENTENCEPIECE_LIB)`. With it OFF the CMake configure fails
outright. Leave it ON.

`JOBS=6`, not `nproc`: Docker Desktop has 8 GB here and the ggml/llama.cpp
translation units peak near 1 GB each.

## Known ceilings

- **The builder base is `nvcr.io/nvidia/cuda:13.0-devel`, ~3.7 GB, for a build
  with CUDA off.** It is only supplying gcc/cmake/ninja. A one-line change to
  `docker/Dockerfile`'s first `FROM` (to `ubuntu:24.04`) removes the pull, at the
  cost of diverging from upstream. Not worth it while the pull is cached; do it
  if this ever builds in CI.
- **Generic armv8-a codegen.** `GGML_NATIVE=OFF` is hardcoded in the Dockerfile,
  and the configure log shows `HAVE_DOTPROD`, `HAVE_MATMUL_INT8` and
  `HAVE_FP16_VECTOR_ARITHMETIC` all failing. A q8_0 model leaves real throughput
  on the table without dotprod/i8mm. It is already ~6x real time, so this only
  matters if concurrency becomes the constraint; fixing it needs a
  `GGML_CPU_ARM_ARCH` argument threaded through the upstream Dockerfile.
- **In-tree libraries are staged twice** (`/opt/nemo-speech/lib` and
  `/opt/nemo-speech-scratch-rootfs/...`), ~5 MB of the 28 MB image. Upstream
  `copy_dep` quirk, harmless.
- `--cors-origin "*"` in the compose command, because the browser reaches this
  from the Next.js origin. Narrow it to the real origin before this is exposed
  anywhere untrusted — and add `--api-key` and TLS at the same time; the server
  logs a warning about exactly this on every start.

## The chatbot uses this

`frontend/lib/use-dictation.ts` speaks this protocol and `lib/utils.ts::sttUrl()`
defaults to `ws://<page hostname>:2701/v1/realtime`. `../stt` is no longer wired
to anything; it still runs on :2700 if you want to compare.

Three differences from the old `../stt` socket caught the port out, each of which
fails silently in the UI rather than loudly:

- `delta` is **incremental** (`" hum"`, `"id"`, `"ity"`), where the old `partial`
  was cumulative — so deltas are concatenated, not assigned.
- `completed` re-punctuates the whole utterance (`"well"` → `"well."`), so it
  replaces the accumulated deltas instead of being appended to them.
- **The server never closes the socket.** The old hook cleared its "transcribing"
  spinner in `onclose`; here nothing would ever fire it, so the hook tears the
  socket down itself once the committed utterance completes.

## API

Upstream's, unchanged: `docs/api.md` in the checkout.

- `POST /v1/audio/transcriptions` — multipart WAV, OpenAI-compatible subset.
  `response_format` takes `json`, `verbose_json` (word timings), `text`, `srt`,
  `vtt`.
- `WS /v1/realtime` — live PCM16, and what the chatbot's mic button now uses.
  An event protocol: optional `session.update`, then binary PCM16 frames, then
  `input_audio_buffer.commit`; partials arrive as
  `conversation.item.input_audio_transcription.delta`, finals as `.completed`.
  `test_stream.py` pins the four behaviours `frontend/lib/use-dictation.ts`
  depends on — run it with the container up.
- `GET /health`, `/ready`, `/version`, `/v1/models`, and a playground UI at `/`.
- `nemo-speech health --quiet` is the container healthcheck — the image is
  `FROM scratch`, so there is no shell or curl to probe with.
