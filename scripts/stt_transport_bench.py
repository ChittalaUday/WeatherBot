#!/usr/bin/env python3
"""HTTP vs gRPC for one-shot transcription, the way an internal service would call it.

    .venv/bin/python scripts/stt_transport_bench.py            # 1,2,4,8 concurrent callers
    .venv/bin/python scripts/stt_transport_bench.py -c 16 -n 40

Not the same question as scripts/stt_bench.py. That one simulates live microphones and asks
how many people can dictate at once. This one asks whether a backend calling the ASR over
gRPC gets an answer faster than one POSTing a WAV, which is what matters when the caller is
another service and there is no human waiting on a partial.

Both sides do the same work - upload ~4.5 s of audio, block, get a transcript - and both run
as blocking calls in a thread pool, so the comparison is transport and serialisation only:

  HTTP  POST /v1/audio/transcriptions   multipart WAV  ->  {"text": ...}   (nemo-speech serve)
  gRPC  RivaSpeechRecognition.Recognize raw PCM bytes  ->  transcript      (riva_server)

They are separate processes that each load their own copy of the model, so this also shows
whether one of them is simply better fed.
"""
import argparse
import io
import json
import statistics
import subprocess
import sys
import time
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Anchored to the repo, not the shell: these services get moved between folders and a
# cwd-relative default silently breaks the run rather than the path.
ROOT = Path(__file__).resolve().parent.parent
HTTP = "http://127.0.0.1:2701/v1/audio/transcriptions"
GRPC = "127.0.0.1:50051"
PROTO_DIR = ROOT / "stt-cpp/NeMo-Speech.cpp/proto/riva-common"
STUBS = ROOT / ".venv/lib/riva-stubs"  # generated, not committed


def build_stubs():
    """protoc the Riva protos once into a cache dir, so nothing generated lives in the repo."""
    marker = STUBS / "riva" / "proto" / "riva_asr_pb2_grpc.py"
    if not marker.exists():
        from grpc_tools import protoc

        STUBS.mkdir(parents=True, exist_ok=True)
        sources = [str(p) for p in (PROTO_DIR / "riva" / "proto").glob("*.proto")]
        code = protoc.main(["protoc", f"-I{PROTO_DIR}", f"--python_out={STUBS}",
                            f"--grpc_python_out={STUBS}", *sources])
        if code != 0:
            raise SystemExit("protoc failed generating the Riva stubs")
    sys.path.insert(0, str(STUBS))


def multipart(wav: bytes) -> tuple:
    """A file upload without pulling in requests - it is nine lines of bytes."""
    boundary = "----sttbench"
    body = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n"
        f"Content-Type: audio/wav\r\n\r\n".encode() + wav +
        f"\r\n--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def http_once(wav: bytes) -> tuple:
    body, content_type = multipart(wav)
    request = urllib.request.Request(HTTP, data=body, headers={"Content-Type": content_type})
    began = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        text = json.loads(response.read())["text"]
    return time.perf_counter() - began, text, len(body)


def grpc_once(stub, asr, audio, pcm: bytes) -> tuple:
    config = asr.RecognitionConfig(
        encoding=audio.LINEAR_PCM, sample_rate_hertz=16000,
        language_code="en-US", max_alternatives=1,
        # HTTP's automatic_punctuation defaults to true and Riva's does not, so without
        # this gRPC returns "will it rain in mumbi" and looks like a different, worse
        # model. Same engine, different default.
        enable_automatic_punctuation=True,
    )
    request = asr.RecognizeRequest(config=config, audio=pcm)
    began = time.perf_counter()
    response = stub.Recognize(request, timeout=120)
    elapsed = time.perf_counter() - began
    results = response.results
    text = results[0].alternatives[0].transcript if results and results[0].alternatives else ""
    return elapsed, text, request.ByteSize()


def pct(values, q):
    clean = sorted(values)
    return clean[min(len(clean) - 1, int(q * len(clean)))] if clean else float("nan")


def run(label, call, concurrency, repeat):
    call()  # warm the path; the first call pays for connection setup and any lazy init
    began = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        out = list(pool.map(lambda _: call(), range(repeat)))
    wall = time.perf_counter() - began
    times = [t for t, _, _ in out]
    return {
        "transport": label, "concurrency": concurrency,
        "p50": pct(times, 0.5), "p95": pct(times, 0.95), "mean": statistics.fmean(times),
        "rps": repeat / wall, "bytes": out[0][2], "text": out[0][1].strip(),
    }


def up(url):
    try:
        urllib.request.urlopen(url, timeout=5).read()
        return True
    except Exception:
        return False


def grpc_up():
    try:
        subprocess.run(["docker", "inspect", "stt-cpp-stt-cpp-grpc-1"], capture_output=True,
                       check=True, timeout=10)
        return True
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--concurrency", default="1,2,4,8")
    ap.add_argument("-n", "--repeat", type=int, default=20, help="requests per level")
    ap.add_argument("-w", "--wav", default=str(ROOT / "stt-cpp/sample16k_long.wav"))
    args = ap.parse_args()

    raw = Path(args.wav).read_bytes()
    with wave.open(io.BytesIO(raw)) as audio_file:
        if (audio_file.getframerate(), audio_file.getnchannels()) != (16000, 1):
            return f"{args.wav} must be 16 kHz mono"
        seconds = audio_file.getnframes() / 16000
        audio_file.rewind()
        pcm = audio_file.readframes(audio_file.getnframes())

    # `build_stubs()` generates riva.proto into the working directory a line above, so there
    # is nothing here for a static checker to resolve - the package is real only after it runs.
    build_stubs()
    import grpc
    from riva.proto import riva_asr_pb2 as asr  # pyright: ignore[reportMissingImports]
    from riva.proto import riva_asr_pb2_grpc as asr_grpc  # pyright: ignore[reportMissingImports]
    from riva.proto import riva_audio_pb2 as audio  # pyright: ignore[reportMissingImports]

    if not up("http://127.0.0.1:2701/health"):
        return "HTTP service down - docker compose up -d in stt-cpp/"
    channel = grpc.insecure_channel(GRPC)
    try:
        grpc.channel_ready_future(channel).result(timeout=20)
    except grpc.FutureTimeoutError:
        return f"gRPC service not answering on {GRPC} - docker compose up -d in stt-cpp/"
    stub = asr_grpc.RivaSpeechRecognitionStub(channel)

    print(f"audio {args.wav}  {seconds:.2f}s   {args.repeat} requests per level\n")
    header = f"{'transport':10}{'conc':>5}{'p50':>9}{'p95':>9}{'mean':>9}{'req/s':>9}{'payload KB':>12}"
    print(header)
    print("-" * len(header))
    rows = []
    for level in (int(n) for n in args.concurrency.split(",")):
        for row in (run("http", lambda: http_once(raw), level, args.repeat),
                    run("grpc", lambda: grpc_once(stub, asr, audio, pcm), level, args.repeat)):
            rows.append(row)
            print(f"{row['transport']:10}{row['concurrency']:>5}{row['p50']:>9.3f}"
                  f"{row['p95']:>9.3f}{row['mean']:>9.3f}{row['rps']:>9.1f}"
                  f"{row['bytes'] / 1024:>12.1f}")

    print("\nseconds per request; req/s is what the level actually sustained.")
    for transport in ("http", "grpc"):
        sample = next((r["text"] for r in rows if r["transport"] == transport), "")
        print(f"  {transport}: {sample!r}")
    best = {t: max(r["rps"] for r in rows if r["transport"] == t) for t in ("http", "grpc")}
    faster = max(best, key=lambda transport: best[transport])
    print(f"\n  peak throughput: http {best['http']:.1f} req/s, grpc {best['grpc']:.1f} req/s"
          f"  ->  {faster} by {best[faster] / min(best.values()):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
