import asyncio
import json
import time
import wave
import statistics
import websockets

CHUNK = 4000  # 4000 samples = 250ms @ 16kHz s16le

async def run_single_stream(audio_path, real_time=True):
    wf = wave.open(audio_path, "rb")
    assert (wf.getnchannels(), wf.getsampwidth(), wf.getframerate()) == (1, 2, 16000), \
        "Audio must be 16kHz mono 16-bit WAV"
    
    total_frames = wf.getnframes()
    audio_duration = total_frames / 16000.0
    
    start_time = time.perf_counter()
    first_partial_time = None
    eof_time = None
    final_time = None
    partials = []
    final_text = ""
    
    async with websockets.connect("ws://localhost:2700", max_size=None) as ws:
        async def send_audio():
            nonlocal eof_time
            while data := wf.readframes(CHUNK):
                await ws.send(data)
                if real_time:
                    await asyncio.sleep(CHUNK / 16000.0)
            eof_time = time.perf_counter()
            await ws.send('{"eof":1}')

        async def receive_transcripts():
            nonlocal first_partial_time, final_time, final_text
            async for message in ws:
                now = time.perf_counter()
                reply = json.loads(message)
                if "partial" in reply:
                    if first_partial_time is None:
                        first_partial_time = now
                    partials.append((now - start_time, reply["partial"]))
                if "text" in reply:
                    final_time = now
                    final_text = reply["text"]

        await asyncio.gather(send_audio(), receive_transcripts())
        
    end_time = final_time or time.perf_counter()
    ttft = (first_partial_time - start_time) * 1000.0 if first_partial_time else None
    post_eof_latency = (end_time - eof_time) * 1000.0 if eof_time else None
    total_duration = end_time - start_time
    # Processing latency is total time spent decoding minus silence/audio pacing
    rtf = total_duration / audio_duration if audio_duration > 0 else 0

    return {
        "audio_duration_sec": audio_duration,
        "total_duration_sec": total_duration,
        "ttft_ms": ttft,
        "post_eof_latency_ms": post_eof_latency,
        "rtf": rtf,
        "partial_count": len(partials),
        "final_text": final_text,
        "partials": partials
    }

async def run_stress_test(audio_path, concurrency_levels=[1, 2, 4, 8], real_time=False):
    print("\n" + "="*70)
    print(f"  STRESS TEST RUNNING (Real-time Pacing: {real_time})")
    print("="*70)
    
    results_by_concurrency = {}
    
    for c in concurrency_levels:
        print(f"\n[+] Testing Concurrency Level: {c} simultaneous stream(s)...")
        tasks = []
        t0 = time.perf_counter()
        for _ in range(c):
            tasks.append(run_single_stream(audio_path, real_time=real_time))
        
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        t1 = time.perf_counter()
        
        successful = [o for o in outcomes if isinstance(o, dict) and "final_text" in o]
        failures = len(outcomes) - len(successful)
        
        if successful:
            audio_durs = [r["audio_duration_sec"] for r in successful]
            ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
            post_eofs = [r["post_eof_latency_ms"] for r in successful if r["post_eof_latency_ms"] is not None]
            rtfs = [r["rtf"] for r in successful]
            
            total_audio_processed = sum(audio_durs)
            wall_time = t1 - t0
            throughput = total_audio_processed / wall_time if wall_time > 0 else 0
            
            summary = {
                "concurrency": c,
                "total_streams": c,
                "successful": len(successful),
                "failed": failures,
                "wall_time_sec": wall_time,
                "throughput_audio_sec_per_sec": throughput,
                "avg_ttft_ms": statistics.mean(ttfts) if ttfts else 0,
                "p50_ttft_ms": statistics.median(ttfts) if ttfts else 0,
                "avg_post_eof_ms": statistics.mean(post_eofs) if post_eofs else 0,
                "p50_post_eof_ms": statistics.median(post_eofs) if post_eofs else 0,
                "p95_post_eof_ms": sorted(post_eofs)[int(len(post_eofs)*0.95)] if post_eofs else 0,
                "avg_rtf": statistics.mean(rtfs) if rtfs else 0
            }
        else:
            summary = {
                "concurrency": c,
                "total_streams": c,
                "successful": 0,
                "failed": failures,
                "wall_time_sec": t1 - t0,
                "throughput_audio_sec_per_sec": 0,
                "avg_ttft_ms": 0,
                "p50_ttft_ms": 0,
                "avg_post_eof_ms": 0,
                "p50_post_eof_ms": 0,
                "p95_post_eof_ms": 0,
                "avg_rtf": 0
            }
            
        results_by_concurrency[c] = summary
        
        print(f"    Streams Completed: {summary['successful']}/{summary['total_streams']}")
        print(f"    Wall Time:        {summary['wall_time_sec']:.2f}s")
        print(f"    Throughput:       {summary['throughput_audio_sec_per_sec']:.2f}x real-time audio/sec")
        print(f"    Avg TTFT:         {summary['avg_ttft_ms']:.1f} ms")
        print(f"    Avg Post-EOF Lat: {summary['avg_post_eof_ms']:.1f} ms")
        print(f"    P95 Post-EOF Lat: {summary['p95_post_eof_ms']:.1f} ms")
        print(f"    Avg RTF:          {summary['avg_rtf']:.3f}")
        
    return results_by_concurrency

async def main():
    import os
    sample_short = "stt/sample16k.wav" if os.path.exists("stt/sample16k.wav") else "/tmp/sample16k.wav"
    sample_long = "stt/sample16k_long.wav" if os.path.exists("stt/sample16k_long.wav") else "/tmp/sample16k_long.wav"
    
    print("="*70)
    print("  STT ENDPOINT EVALUATION & BENCHMARK")
    print("  Endpoint: ws://localhost:2700")
    print("  Model:    nvidia/nemotron-3.5-asr-streaming-0.6b")
    print("="*70)
    
    print("\n--- [1] Single Stream Real-Time Evaluation (Short Sample) ---")
    res1 = await run_single_stream(sample_short, real_time=True)
    print(f"Audio Duration:        {res1['audio_duration_sec']:.2f} s")
    print(f"Total Stream Time:     {res1['total_duration_sec']:.2f} s")
    print(f"Time to 1st Partial:   {res1['ttft_ms']:.1f} ms")
    print(f"Post-EOF Latency:      {res1['post_eof_latency_ms']:.1f} ms")
    print(f"Real-Time Factor (RTF):{res1['rtf']:.3f}")
    print(f"Partial Updates Count: {res1['partial_count']}")
    print(f"Recognized Text:       \"{res1['final_text']}\"")
    
    print("\n--- [2] Single Stream Real-Time Evaluation (Long Sample) ---")
    res2 = await run_single_stream(sample_long, real_time=True)
    print(f"Audio Duration:        {res2['audio_duration_sec']:.2f} s")
    print(f"Total Stream Time:     {res2['total_duration_sec']:.2f} s")
    print(f"Time to 1st Partial:   {res2['ttft_ms']:.1f} ms")
    print(f"Post-EOF Latency:      {res2['post_eof_latency_ms']:.1f} ms")
    print(f"Real-Time Factor (RTF):{res2['rtf']:.3f}")
    print(f"Partial Updates Count: {res2['partial_count']}")
    print(f"Recognized Text:       \"{res2['final_text']}\"")
    
    print("\n--- Partial Updates Stream Timeline (Long Sample) ---")
    for t, p in res2['partials']:
        print(f"  [+{t:5.2f}s] {p}")
        
    print("\n" + "="*70)
    print("  STARTING CONCURRENCY STRESS TESTS")
    print("="*70)
    
    # Stress test at full batch speed (no sleeping between audio frames)
    stress_fast = await run_stress_test(sample_long, concurrency_levels=[1, 2, 4, 8], real_time=False)
    
    # Save results as JSON for summary report
    with open("/tmp/stt_benchmark_results.json", "w") as f:
        json.dump({
            "single_short": res1,
            "single_long": res2,
            "stress_fast": stress_fast
        }, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main())
