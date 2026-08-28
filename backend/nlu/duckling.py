"""Time expressions, resolved by Duckling instead of by a hand-written table.

    python -m backend.nlu.duckling            # the checks for this module (needs the service)
    docker run -d --name duckling-service -p 8008:8000 rasa/duckling:latest

`src.tagger.normalize_time` holds a few hundred hand-written entries and every one of them was
added after somebody's question came back wrong. Duckling is the same job done by a grammar
that already knows the tail: measured against this deployment's own wording, it places "last
summer", "last few days", "so far today", "between 6pm and 9pm" and "tonight at 6" - five
expressions that the tables miss and that cost either a wrong window or a call to a local model.

**It returns a canonical string, not a window.** That is deliberate, and it is what keeps this
module a proposer rather than a second authority on dates:

    "last summer"   ->  "2025-06-21 to 2025-09-24"   ->  timewindow.resolve  ->  Window
    "at 6"          ->  "18:00"                      ->  timewindow.resolve  ->  Window
    "tomorrow"      ->  "tomorrow"                   ->  timewindow.resolve  ->  Window

`resolve` already reads a two-date range through `parse_dates`, so any window Duckling can
describe is expressible in a form the existing resolver understands. Nothing downstream changes,
`resolve` stays the only thing that decides what a date means, and `understood=False` still
closes the silent-wrong-window class. Handing back a `Window` directly would have made this the
second place in the codebase that owns a calendar, which is the bug the timewindow docstring was
written about.

What Duckling does NOT decide, and why each stays here or upstream:

    granularity     hourly or daily drives which weather API is called. Duckling's `grain` is a
                    good signal and it is used, but the profile layer still has the last word.
    the domain read A June that has not happened is past the ten-day forecast, so "june" means
                    the most recent June. Duckling answers with the next one - correct English,
                    wrong product. Overridden below.
    the tail it     "prior days", "the other day", "couple of days back", "whole day", "the last
    misses          decade", and Hinglish ("barish hogi kya kal") all come back empty. The
                    existing rules-then-model path still runs for those.

Failure is a return of "", never an exception: this is a container, containers stop, and a turn
that cannot reach Duckling must fall through to the tables rather than fail.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import httpx

from backend.config import (
    DUCKLING_ENABLED,
    DUCKLING_LOCALE,
    DUCKLING_TIMEOUT,
    DUCKLING_TZ,
    DUCKLING_URL,
)
from src.dates import MONTHS

# Duckling's grain -> whether the answer is about hours or about days. "minute" and "second"
# are hourly too: the feed has no finer resolution than an hour, so a 6:30 question is an
# hourly question.
HOURLY_GRAINS = {"second", "minute", "hour"}

# Grain -> how long a bare point covers. Duckling answers "this month" with the first of the
# month at day zero and `grain: month`, so the grain is what says the answer is a month rather
# than a midnight.
_POINT_SPAN = {"second": timedelta(seconds=1), "minute": timedelta(minutes=1),
               "hour": timedelta(hours=1), "day": timedelta(days=1),
               "week": timedelta(days=7)}


def _iso(when: datetime) -> str:
    return when.strftime("%Y-%m-%d")


def _parse_stamp(value: str) -> datetime | None:
    """Duckling's "2026-08-29T00:00:00.000+05:30" -> a naive local datetime.

    Naive on purpose. Every other datetime in this pipeline is naive local time, and mixing the
    two is how a comparison raises `can't compare offset-naive and offset-aware`. The offset is
    dropped rather than converted because `DUCKLING_TZ` already asked for local time.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value[:26].split("+")[0].split(".")[0])
    except ValueError:
        return None


async def _ask(text: str, now: datetime, client: httpx.AsyncClient | None = None) -> list:
    """Duckling's time entities for this sentence. [] on anything going wrong."""
    if not DUCKLING_ENABLED or not (text or "").strip():
        return []
    owns = client is None
    client = client or httpx.AsyncClient(timeout=DUCKLING_TIMEOUT)
    try:
        # `reftime` rather than the container's clock: a fixed reference is what makes this
        # testable and what stops two calls a second apart disagreeing across midnight.
        response = await client.post(
            f"{DUCKLING_URL}/parse", timeout=DUCKLING_TIMEOUT,
            data={"locale": DUCKLING_LOCALE, "text": text, "tz": DUCKLING_TZ,
                  "reftime": int(now.timestamp() * 1000), "dims": json.dumps(["time"])})
        response.raise_for_status()
        found = response.json()
    except (httpx.HTTPError, ValueError, OSError):
        return []                      # not running, slow, or nonsense - the tables take over
    return [e for e in found if isinstance(e, dict) and e.get("dim") == "time"]


def _best(entities: list) -> dict | None:
    """The entity most likely to be the period asked about.

    Non-latent first: Duckling marks a reading it had to reach for as `latent`, and a latent
    hit on a bare number is how "rain over 5 acres" would become five o'clock. Then the longest
    body, because "this time last year" beats the "last year" inside it.
    """
    real = [e for e in entities if not e.get("latent")] or entities
    return max(real, key=lambda e: len(e.get("body") or ""), default=None)


def _bare_future_month(entity: dict, when: datetime, now: datetime) -> str:
    """A month named with no year resolves forward in English and backward in this product.

    Duckling answers "june", asked in August 2026, with June 2027 - the next one, which is what
    the words mean. It is also past the ten-day forecast, so it is not answerable, and the only
    reading worth having is the most recent June. `timewindow._bare_month` already holds this
    rule; this hands the expression back to it rather than restating it.
    """
    body = (entity.get("body") or "").lower().strip()
    named_month = body in MONTHS or any(w in MONTHS for w in body.split())
    if named_month and when > now:
        return body                    # let timewindow._bare_month read it
    return ""


def _canonical(entity: dict, now: datetime) -> str:
    """One time entity -> a canonical string `timewindow.resolve` understands."""
    value = entity.get("value") or {}
    kind = value.get("type")

    if kind == "interval":
        start = _parse_stamp((value.get("from") or {}).get("value", ""))
        end = _parse_stamp((value.get("to") or {}).get("value", ""))
        grain = ((value.get("from") or value.get("to") or {}).get("grain")) or "day"
        if start is None and end is None:
            return ""
        # "since 2018" and "from 2010 to" come back open-ended. Open at the top means "up to
        # now"; open at the bottom is not a period this product can answer, so it is declined.
        if end is None:
            end = now
        if start is None:
            return ""
        # Duckling's `to` is exclusive - "last 3 days" ends at the 28th at 00:00 meaning the
        # 27th is the last full day. Half-open arithmetic left every range a day long.
        end = end - timedelta(seconds=1)
        if grain in HOURLY_GRAINS and start.date() == end.date():
            return f"{start:%H:%M}-{end:%H:%M}"
        if _iso(start) == _iso(end):
            return _iso(start)
        return f"{_iso(start)} to {_iso(end)}"

    start = _parse_stamp(value.get("value", ""))
    if start is None:
        return ""
    grain = value.get("grain") or "day"

    if (handed_back := _bare_future_month(entity, start, now)):
        return handed_back

    if grain in HOURLY_GRAINS:
        # A clock time on today is just the clock time - `resolve` already rolls it forward to
        # the next occurrence if it has passed, which is the behaviour asked for.
        if start.date() == now.date():
            return f"{start:%H:%M}"
        return f"{_iso(start)} to {_iso(start)}"
    if grain == "month":
        return f"{start:%B %Y}".lower()          # "march 2022" - resolve reads a whole month
    if grain == "year":
        return f"{start:%Y}"                     # a bare year - resolve reads the whole of it
    span = _POINT_SPAN.get(grain, timedelta(days=1))
    end = start + span - timedelta(seconds=1)
    if _iso(start) == _iso(end):
        return _iso(start)
    return f"{_iso(start)} to {_iso(end)}"


async def canonical(text: str, now: datetime | None = None,
                    client: httpx.AsyncClient | None = None) -> str:
    """The period this sentence asks about, as a form `timewindow.resolve` understands.

    "" when Duckling found nothing, is switched off, or could not be reached - all three mean
    the same thing to the caller, which is "keep going with the tables".
    """
    now = now or datetime.now()
    entity = _best(await _ask(text, now, client))
    return _canonical(entity, now) if entity else ""


async def probe() -> dict:
    """Is the service actually answering, and with the locale that reads dates day-first?

    Startup-only, and worth a round trip for the same reason `generation.probe` is: every
    failure inside a turn is deliberately silent - it falls back to the tables, which is right
    mid-turn and wrong for a deployment. A stopped container and a slow one look identical from
    inside `_ask`, and so does the far worse case of a container answering with the wrong
    locale: en_US reads "11/06/2026" as 6 November and nothing downstream can tell.
    """
    if not DUCKLING_ENABLED:
        return {"ok": False, "note": "DUCKLING_ENABLED is off - time expressions fall back to "
                                     "the tables and the local model"}
    now = datetime(2026, 6, 1, 12, 0)
    try:
        async with httpx.AsyncClient(timeout=DUCKLING_TIMEOUT) as client:
            said = await canonical("rain on 11/06/2026", now, client)
    except Exception as exc:                                        # noqa: BLE001
        return {"ok": False, "url": DUCKLING_URL,
                "note": f"no duckling at {DUCKLING_URL} ({type(exc).__name__}) - time "
                        f"expressions fall back to the tables and the local model"}
    if not said:
        return {"ok": False, "url": DUCKLING_URL,
                "note": f"duckling at {DUCKLING_URL} answered nothing for a plain date - is it "
                        f"up? Time expressions fall back to the tables and the local model. "
                        f"docker run -d --name duckling-service -p 8008:8000 rasa/duckling"}
    if said != "2026-06-11":
        return {"ok": False, "url": DUCKLING_URL, "locale": DUCKLING_LOCALE, "read": said,
                "note": f"DUCKLING_LOCALE={DUCKLING_LOCALE!r} read 11/06/2026 as {said} - it "
                        f"must be 2026-06-11. en_US reads dates month-first; use en_IN."}
    return {"ok": True, "url": DUCKLING_URL, "locale": DUCKLING_LOCALE, "tz": DUCKLING_TZ}


async def demo():
    """Self-check against the running service. Skips itself when it is not up."""
    from backend.pipeline.timewindow import resolve

    now = datetime(2026, 8, 28, 15, 30)          # a Friday, 3:30pm
    if not await canonical("tomorrow", now):
        print(f"SKIP: no duckling at {DUCKLING_URL} - "
              f"docker run -d --name duckling-service -p 8008:8000 rasa/duckling:latest")
        return

    async with httpx.AsyncClient(timeout=DUCKLING_TIMEOUT) as client:
        async def form(text):
            return await canonical(text, now, client)

        # the five the hand-written tables miss - the reason this module exists
        assert await form("rainfall last summer") == "2025-06-21 to 2025-09-23", \
            await form("rainfall last summer")
        assert await form("rain in the last few days") == "2026-08-25 to 2026-08-27"
        assert await form("rain so far today") == "2026-08-28"
        assert await form("rain between 6pm and 9pm") == "18:00-21:59"
        assert await form("tonight at 6") == "18:00"

        # the everyday wording still comes back in the form the tables already used
        assert await form("will it rain tomorrow") == "2026-08-29"
        assert await form("rain yesterday") == "2026-08-27"
        assert await form("rain at 6pm") == "18:00"
        # 6am has passed at 15:30, and resolve rolls a passed clock time to tomorrow
        assert await form("rain at 6") == "18:00", await form("rain at 6")

        # day-first, because DUCKLING_LOCALE is en_IN. en_US answers 6 November.
        assert await form("rain on 11/06/2026") == "2026-06-11", await form("rain on 11/06/2026")
        assert await form("rain on 2026-06-11") == "2026-06-11"
        assert await form("rain in march 2022") == "march 2022"
        assert await form("rain in 2017") == "2017"

        # a bare future month is handed back for timewindow._bare_month to read as the most
        # recent one - Duckling says June 2027, which is true English and unanswerable weather
        # the body Duckling matched is what is handed back, preposition and all - "in june" is
        # a form `_bare_month` reads, and its `_NEEDS_A_PREPOSITION` rule wants that preposition
        assert await form("rainfall in june") == "in june", await form("rainfall in june")
        assert resolve("in june", now).start.year == 2026, "the most recent June, not 2027"

        # the tail Duckling does not place: "" so the caller keeps going
        for missed in ("prior days", "the other day", "couple of days back",
                       "rainfall for whole day", "barish hogi kya kal", "hello there", ""):
            assert await form(missed) == "", f"{missed!r} -> {await form(missed)!r}"

        # and every form it does produce has to survive the resolver, or it is worthless
        for text in ("rainfall last summer", "rain in the last few days", "rain so far today",
                     "rain between 6pm and 9pm", "tonight at 6", "will it rain tomorrow",
                     "rain on 11/06/2026", "rain in march 2022", "rainfall in june"):
            said = await form(text)
            window = resolve(said, now)
            assert window.understood, f"{text!r} -> {said!r} is not a form resolve knows"
    print("duckling demo OK - the tail it places, the tail it does not, and every form resolves")


if __name__ == "__main__":
    import asyncio

    asyncio.run(demo())
