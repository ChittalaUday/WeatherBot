"""
The checks that need the network: the weather APIs, Solr, ollama and the hosted parser.
Run: python tests/test_live_stack.py

Kept apart from the unit checks because these fail when a third party is having a bad day, and
a suite that goes red for that reason stops being read. Two of them assert offline first and
only then reach out, so running this file exercises both halves.

These were `demo()` functions inside backend/api/*.py, backend/pipeline/__init__.py and
backend/nlu/llm.py. Nothing about them changed except where they live.
"""

import asyncio

from _root import ROOT  # noqa: F401 - puts the repo root on sys.path


async def check_pipeline_live():
    """Self-check: the column rule offline, then one full run if the APIs answer."""
    from backend.nlu import Registry
    from backend.pipeline.__init__ import run, served_fields, sources

    # the archive's own shape - six measurements and their normals, nothing else. Asking it
    # for sunshine must lose the column, not the answer.
    archive = [{"Date_time": "2019-08-07T00:00:00", "Rainfall": 6.6279, "Tmax": 28.209,
                "Tmin": 22.8525, "RH": 79.5186, "Wind_Speed": 4.2744, "DayLength": 3.0498}]
    assert served_fields(["Rainfall", "Tmax", "SunSD"], [archive]) == \
        (["Rainfall", "Tmax"], ["SunSD"])
    # hourly: no daily max/min in the feed, but humidity itself is there
    hourly = [{"Date_time": "2026-08-14T16:00:00", "RH": 58.83, "RH_max": None, "SunSD": 1}]
    assert served_fields(["RH", "RH_max", "SunSD"], [hourly]) == (["RH", "SunSD"], ["RH_max"])
    # a dead fetch drops nothing: quality must stay free to report that nothing came back
    assert served_fields(["RH", "SunSD"], [[{"Date_time": "x"}]]) == (["RH", "SunSD"], [])
    assert served_fields(["RH"], [[]]) == (["RH"], [])
    print("  columns: kept what the feed sent, dropped what it never did")

    registry = Registry()
    async with sources.client() as http:
        for text in ("should i spray pesticide on the cotton in Guntur tomorrow",
                     "rain and temperature in Guntur this week", "hey there"):
            got = await run(http, registry.understand(text))
            print(f"\n  {text}")
            print(f"    ok={got.ok} answered={got.answered} {got.total_ms}ms  "
                  f"stages: {list(got.stages)}")
            for name, stage in got.stages.items():
                head = (stage.get("summary") or stage.get("headline") or stage.get("status")
                        or stage.get("verdict") or stage.get("served_by")
                        or stage.get("note") or "")
                print(f"      {name:14s} {str(head)[:64]}")
            if got.answered:
                # the payload a client renders must be built from the same object
                body = got.payload(registry.understand(text))
                assert body["summary"] == got.summary and body["table"] == got.table

async def check_chat_stream_live():
    """Self-check: one turn's events, in order, with every stage timed."""
    from backend.api.chat import turn
    sent = [event async for event in turn("will it rain in Guntur tomorrow",
                                          chat_id="demo-chat", model="v4")]
    kinds = [e["type"] for e in sent]
    streamed = {"delta", "thinking"}
    print("  " + " -> ".join(f"{e['type']}:{e.get('stage', '')}".rstrip(":") for e in sent
                             if e["type"] not in streamed)
          + f"  ({kinds.count('thinking')} thinking + {kinds.count('delta')} answer pieces)")

    assert kinds[-1] in {"result", "chat", "clarify", "need_location", "error"}, kinds[-1]
    result = next((e for e in sent if e["type"] == "result"), None)
    if not result:
        print(f"  no answer ({sent[-1].get('message', kinds[-1])}) - network or model down")
        return
    assert [e["stage"] for e in sent if e["type"] == "status"] == \
        ["understanding", "locating", "fetching", "writing"], kinds
    if "delta" in kinds:
        # words must arrive while it is writing, and add up to the answer that follows
        assert kinds.index("status") < kinds.index("delta") < kinds.index("result"), kinds
        said = "".join(e["text"] for e in sent if e["type"] == "delta")
        assert result["summary"].startswith(said), (said, result["summary"])
        if "thinking" in kinds:
            # the reasoning is its own channel, and it comes before the answer it reasons about
            assert kinds.index("thinking") < kinds.index("delta"), kinds
            thought = "".join(e["text"] for e in sent if e["type"] == "thinking")
            assert thought not in said, "reasoning leaked into the answer"
            print(f"  thought: {thought[:70]}...")
    else:
        print("  no deltas - the local model is offline, answered with the rule-built sentence")

    metrics = result["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {"nlu_ms", "solr_ms", "api_ms", "llm_ms", "db_ms",
                            "total_ms"}, metrics
    assert metrics["total_ms"] >= metrics["llm_ms"]
    print("  " + "  ".join(f"{k}={v}" for k, v in metrics.items()))

    # a greeting never reaches the location resolver
    greeting = [e async for e in turn("hey there", chat_id="demo-chat", model="v4")]
    assert [e["type"] for e in greeting] == ["status", "chat"], greeting
    print(f"  greeting -> {greeting[-1]['message']!r}")
    print("chat stream check OK")

async def check_compare_live(text: str = "should i spray fertilizer on the cotton in Guntur tomorrow"):
    """Self-check: the disagreement rule offline, then a live comparison if the APIs answer."""
    from backend.api.compare import columns, disagreements
    same = [{"ok": True, "intent": "ADVICE", "locations": ["Guntur", "Vizag"]},
            {"ok": True, "intent": "ADVICE", "locations": ["vizag", "guntur"]}]
    assert disagreements(same) == [], disagreements(same)          # order does not matter
    differ = [{"ok": True, "intent": "ADVICE"}, {"ok": True, "intent": "INFORMATION"}]
    assert disagreements(differ) == ["intent"], disagreements(differ)
    assert disagreements([differ[0]]) == [], "one column cannot disagree with anything"
    assert disagreements([{"ok": False}, differ[0]]) == [], "a dead column is not a dissent"
    print("  disagreements: order-insensitive on lists, exact on scalars")

    print(f"\n  {text}")
    async for event in columns(text):
        if event["type"] == "compare_result":
            head = (f"{event.get('intent', '-'):12s} {event.get('activity', '-'):10s} "
                    f"loc={event.get('locations')}" if event.get("ok")
                    else f"FAILED {event.get('error', '')[:50]}")
            print(f"    {event['version']:4s} {event['latency_ms']:>6}ms  {head}")
        elif event["type"] == "compare_done":
            print(f"    disagree on: {event['disagreements'] or 'nothing'} "
                  f"({event['total_ms']}ms)")

async def check_hosted_parser_live():
    from backend.nlu.llm import asyncio, understand
    for text in ("should i spray fertilizer on my cotton field in Guntur tomorrow",
                 "what is the rainfall fro whole day",
                 "hey there",
                 "what is the air quality in delhi"):
        got = await understand(text)
        await asyncio.sleep(4)                      # the demo would otherwise rate-limit itself
        if not got["ok"]:
            print(f"  {text[:46]:48s} FAILED {got['error']}")
            continue
        print(f"  {text[:46]:48s} {got['intent']:12s} {got['weather_intent']:10s} "
              f"{got['activity']:16s} loc={got['locations']} time={got['times']} "
              f"[{got['latency_ms']}ms]")

def main():
    """Every live check. A third party being down is reported, not raised - that is not a
    regression in this code, and the unit suites cover what is."""
    failures = 0
    for check in (check_pipeline_live, check_chat_stream_live, check_compare_live,
                  check_hosted_parser_live):
        print(f"{check.__name__}:")
        try:
            asyncio.run(check())
        except AssertionError:
            failures += 1
            raise
        except Exception as exc:                       # noqa: BLE001
            print(f"  skipped - {type(exc).__name__}: {str(exc)[:90]}")
    print(f"\nlive checks done{' with ' + str(failures) + ' failure(s)' if failures else ''}")


if __name__ == "__main__":
    main()
