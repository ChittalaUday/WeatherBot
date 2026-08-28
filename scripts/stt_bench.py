#!/usr/bin/env python3
"""Load-compare the two speech services to decide which one goes to production.

    python3 scripts/stt_bench.py                      # both, 1/2/4/8 concurrent mics
    python3 scripts/stt_bench.py -c 1,4,16 -s stt-cpp
    python3 scripts/stt_bench.py --pace fast          # batch throughput instead

The default pace is `realtime`: every simulated client streams at 1x, the speed a person
actually talks, because the production question is not "how fast can it chew a file" but
"how many people can dictate at once before it stops keeping up". A service keeps up while
the tail latency - the wait after you stop talking - stays short.

`--pace fast` pushes audio as fast as the socket accepts it, which is the right question
only for batch transcription of stored files.
"""
import argparse
import asyncio
import json
import statistics
import subprocess
import sys
import time
import urllib.request
import wave
from dataclasses import dataclass, field

import websockets

CHUNK = 4096  # 2048 int16 samples = 128 ms, one browser worklet message
CHUNK_SECONDS = CHUNK / 2 / 16000

# `protocol` picks the wire format, not the service: stt and faster-whisper happen to speak
# the same one (cumulative {"partial"}, {"eof":1} to finish, {"text"} to close the utterance).
SERVICES = {
    "stt": {
        "socket": "ws://127.0.0.1:2700",
        "health": "http://127.0.0.1:2700/health",
        "container": "stt-stt-1",
        "protocol": "partial-eof",
        "what": "Nemotron 0.6B, PyTorch (stt/)",
    },
    "stt-cpp": {
        "socket": "ws://127.0.0.1:2701/v1/realtime",
        "health": "http://127.0.0.1:2701/health",
        "container": "stt-cpp-stt-cpp-1",
        "protocol": "realtime",
        "what": "Nemotron 0.6B, NeMo-Speech.cpp (stt-cpp/)",
    },
    "whisper": {
        # localhost, not 127.0.0.1, and it matters: Docker publishes this one on IPv6 (*:8000)
        # and a stray uvicorn was found holding IPv4 127.0.0.1:8000, which answers 404 to
        # everything. `localhost` resolves to ::1 first and reaches the container.
        "socket": "ws://localhost:8000/ws",
        "health": "http://localhost:8000/health",
        "container": "faster-whisper-server",
        "protocol": "partial-eof",
        "what": "faster-whisper (../faster-whisper)",
    },
}


@dataclass
class Result:
    ok: bool = False
    # Time to get a usable socket. Counted separately because everything else here is
    # timed from the first audio frame, so a server that queues connections looks
    # instantaneous on every other metric while users wait in line.
    connect: float = float("nan")
    first_partial: float = float("nan")  # audio start -> first text back
    # How far behind real time the *sending* fell. A backed-up service stops draining its
    # socket, so our own ws.send() blocks and the audio goes out slower than it was spoken.
    # Without this, `tail` flatters an overloaded server: it is measured from a "end of
    # speech" that backpressure already pushed into the future.
    lag: float = float("nan")
    tail: float = float("nan")  # end of speech -> final transcript
    total: float = float("nan")
    text: str = ""
    error: str = ""


@dataclass
class Sample:
    cpu: list = field(default_factory=list)
    mem: list = field(default_factory=list)


async def stream_once(name: str, pcm: bytes, pace: str) -> Result:
    """One simulated microphone, start to final transcript."""
    url = SERVICES[name]["socket"]
    legacy = SERVICES[name]["protocol"] == "partial-eof"
    r = Result()
    started = time.perf_counter()
    try:
        async with websockets.connect(url, max_size=None, open_timeout=30) as ws:
            first = asyncio.get_running_loop().create_future()
            final = asyncio.get_running_loop().create_future()

            async def read():
                deltas = ""
                async for raw in ws:
                    event = json.loads(raw)
                    if legacy:
                        if event.get("partial") and not first.done():
                            first.set_result(time.perf_counter())
                        if event.get("text") is not None and not final.done():
                            final.set_result((time.perf_counter(), event["text"]))
                            return
                    else:
                        kind = event.get("type", "")
                        if kind.endswith("transcription.delta"):
                            deltas += event.get("delta", "")
                            if deltas.strip() and not first.done():
                                first.set_result(time.perf_counter())
                        elif kind.endswith("transcription.completed"):
                            final.set_result((time.perf_counter(), event["transcript"]))
                            return
                        elif kind == "error":
                            raise RuntimeError(event.get("error", {}).get("message", "server error"))

            reader = asyncio.create_task(read())
            audio_started = time.perf_counter()
            r.connect = audio_started - started
            for i in range(0, len(pcm), CHUNK):
                await ws.send(pcm[i : i + CHUNK])
                if pace == "realtime":
                    await asyncio.sleep(CHUNK_SECONDS)
            spoke_until = time.perf_counter()
            await ws.send('{"eof":1}' if legacy else '{"type":"input_audio_buffer.commit"}')

            done, text = await asyncio.wait_for(final, timeout=300)
            reader.cancel()
            r.ok = True
            r.text = text.strip()
            r.lag = (spoke_until - audio_started) - len(pcm) / 2 / 16000
            r.tail = done - spoke_until
            r.total = done - started
            r.first_partial = (first.result() - audio_started) if first.done() else float("nan")
    except Exception as cause:  # a failure under load is a result, not a crash
        r.error = f"{type(cause).__name__}: {cause}"
    return r


def container_stats(container: str) -> tuple:
    """CPU% and MiB right now, or (nan, nan) if docker cannot say."""
    try:
        out = subprocess.run(
            ["docker", "stats", "--no-stream", "--format", "{{.CPUPerc}}\t{{.MemUsage}}", container],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        cpu, mem = out.split("\t")
        value, unit = mem.split(" / ")[0].rstrip("B"), mem.split(" / ")[0][-3:]
        mib = float(value.rstrip("GiMK")) * (1024 if "G" in unit else 1)
        return float(cpu.rstrip("%")), mib
    except Exception:
        return float("nan"), float("nan")


async def sample_while(container: str, stop: asyncio.Event, into: Sample):
    while not stop.is_set():
        cpu, mem = await asyncio.to_thread(container_stats, container)
        if cpu == cpu:  # not nan
            into.cpu.append(cpu)
            into.mem.append(mem)
        try:
            await asyncio.wait_for(stop.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            pass


def pct(values: list, q: float) -> float:
    clean = sorted(v for v in values if v == v)
    if not clean:
        return float("nan")
    return clean[min(len(clean) - 1, int(q * len(clean)))]


async def measure(name: str, pcm: bytes, audio_seconds: float, concurrency: int, pace: str, stats: bool):
    stop = asyncio.Event()
    sample = Sample()
    watcher = asyncio.create_task(sample_while(SERVICES[name]["container"], stop, sample)) if stats else None

    began = time.perf_counter()
    results = await asyncio.gather(*(stream_once(name, pcm, pace) for _ in range(concurrency)))
    wall = time.perf_counter() - began

    if watcher:
        stop.set()
        await watcher

    ok = [r for r in results if r.ok]
    return {
        "service": name, "concurrency": concurrency, "wall": wall,
        "ok": len(ok), "failed": len(results) - len(ok),
        "conn_p95": pct([r.connect for r in ok], 0.95),
        "first_p50": pct([r.first_partial for r in ok], 0.5),
        "lag_p95": pct([r.lag for r in ok], 0.95),
        "tail_p50": pct([r.tail for r in ok], 0.5),
        "tail_p95": pct([r.tail for r in ok], 0.95),
        # Audio handled per second of wall clock. Below `concurrency`, it is behind real time.
        "throughput": (audio_seconds * len(ok) / wall) if wall else float("nan"),
        "cpu": max(sample.cpu) if sample.cpu else float("nan"),
        "mem": max(sample.mem) if sample.mem else float("nan"),
        "texts": {r.text for r in ok},
        "sample": ok[0].text if ok else "",
        "errors": [r.error for r in results if not r.ok][:2],
    }


def check_up(name: str) -> str:
    try:
        with urllib.request.urlopen(SERVICES[name]["health"], timeout=5) as response:
            body = json.loads(response.read())
        extra = " ".join(str(body[k]) for k in ("model", "compute_type", "device") if k in body)
        return f"{body.get('status', '?')}{(' - ' + extra) if extra else ''}"
    except Exception as cause:
        return f"DOWN ({type(cause).__name__})"


def limits(container: str) -> str:
    """CPU/memory caps, because an uncapped container is not comparable with a capped one."""
    try:
        out = subprocess.run(
            ["docker", "inspect", container, "--format",
             "{{.HostConfig.NanoCpus}} {{.HostConfig.Memory}}"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.split()
        cpus, mem = int(out[0]) / 1e9, int(out[1]) / 1024**3
        return f"{cpus:g} cpu / {mem:g} GB" if cpus or mem else "UNCAPPED (all host cores)"
    except Exception:
        return "?"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-s", "--service", default="all", choices=[*SERVICES, "all"],
                    help="default: all reachable services")
    ap.add_argument("-c", "--concurrency", default="1,2,4,8", help="comma-separated (default 1,2,4,8)")
    ap.add_argument("-w", "--wav", default="stt-cpp/sample16k_long.wav", help="16 kHz mono WAV")
    ap.add_argument("--pace", default="realtime", choices=["realtime", "fast"])
    ap.add_argument("--no-stats", action="store_true", help="skip docker stats sampling")
    args = ap.parse_args()

    audio = wave.open(args.wav)
    if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
        return f"{args.wav} must be 16 kHz mono 16-bit"
    pcm = audio.readframes(audio.getnframes())
    seconds = audio.getnframes() / 16000

    names = list(SERVICES) if args.service == "all" else [args.service]
    levels = [int(n) for n in args.concurrency.split(",")]

    print(f"audio {args.wav}  {seconds:.2f}s  pace={args.pace}  concurrency={levels}\n")
    caps = set()
    for name in names:
        cap = limits(SERVICES[name]["container"])
        caps.add(cap)
        print(f"  {name:8} {SERVICES[name]['what']:38} {cap:24} {check_up(name)}")
    if len(caps) > 1:
        print("\n  ! containers have different resource limits - these numbers are not a fair")
        print("    comparison until they match. Cap them the same way before deciding anything.")
    live = [n for n in names if not check_up(n).startswith("DOWN")]
    if not live:
        return "no service reachable - docker compose up -d in stt/, stt-cpp/ and faster-whisper/"
    print()

    for name in live:  # a cold model would libel the first level, so spend one stream warming
        await stream_once(name, pcm, "fast")

    header = f"{'service':9}{'conc':>5}{'ok':>5}{'fail':>5}{'connect':>9}{'1st word':>10}{'lag p95':>9}{'tail p50':>10}{'tail p95':>10}{'x-realtime':>12}{'cpu%':>8}{'mem MiB':>9}"
    print(header)
    print("-" * len(header))
    rows = []
    for level in levels:
        for name in live:
            row = await measure(name, pcm, seconds, level, args.pace, not args.no_stats)
            rows.append(row)
            print(
                f"{row['service']:9}{row['concurrency']:>5}{row['ok']:>5}{row['failed']:>5}"
                f"{row['conn_p95']:>9.2f}{row['first_p50']:>10.2f}{row['lag_p95']:>9.2f}{row['tail_p50']:>10.2f}{row['tail_p95']:>10.2f}"
                f"{row['throughput']:>11.1f}x{row['cpu']:>8.0f}{row['mem']:>9.0f}"
            )
            if row["errors"]:
                print(f"{'':9}  ! {row['errors'][0][:100]}")
            if len(row["texts"]) > 1:
                print(f"{'':9}  ! transcripts diverged under load: {len(row['texts'])} distinct")

    print("\nseconds. 'tail' is the wait after you stop speaking - the number a user feels.")
    print("'lag' is how far the stream itself fell behind real time; above ~0 it is saturated.")
    print("'x-realtime' is audio handled per wall second; below the concurrency it is behind.\n")
    print("what each one heard (same audio - accuracy is not what this script measures,")
    print("but a service that is fast and wrong is not a faster service):")
    for name in live:
        sample = next((r["sample"] for r in rows if r["service"] == name and r["sample"]), "")
        print(f"  {name:9} {sample!r}")
    print()

    for name in live:
        mine = [r for r in rows if r["service"] == name and not r["failed"]]
        # Both gates matter: a saturated server can show a short tail purely because
        # backpressure stretched the utterance it is being measured against.
        good = [r for r in mine if r["tail_p95"] < 2.0 and r["lag_p95"] < 1.0 and r["conn_p95"] < 1.0]
        limit = max((r["concurrency"] for r in good), default=0)
        print(f"  {name:8} sustains {limit} concurrent real-time mics within a 2 s tail")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
