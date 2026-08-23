"""
The last layer: the deterministic conclusion, said in plain words by a local model.

    python -m backend.generation.llm        # self-check
    ollama pull qwen3:1.7b                  # OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TIMEOUT in .env

Everything upstream is unchanged - the numbers, the advice verdict, the caveats are still
decided by rules. This only re-says what they produced, from the sections
`backend.generation.context` retrieved, so it cannot state a figure the pipeline did not
already stand behind.

Any failure - Ollama not running, a timeout, an empty reply, a reply that grew instead of
shrank - returns the original sentence untouched. A chat that answers slightly less well is a
worse day; a chat that answers nothing is a broken product.
"""

from __future__ import annotations

import asyncio
import json
import re

import httpx

from backend.config import OLLAMA_MODEL, OLLAMA_TIMEOUT, OLLAMA_URL
from backend.generation import prompts

# One line per way a turn can fail, in the reader's terms. The model re-says these; if it is
# not running, or says something it should not, they go out exactly as written - a failure
# message must never depend on another thing working.
TROUBLE_LINES = {
    "location": "I could not find that place. Try a nearby town or district, or share your "
                "location.",   # ADVICE below is its second half, reused after a suggestion
    "data": "The weather service is not answering just now. Try again in a moment.",
    "time": "I could not work out which dates you meant. Try naming them directly, like "
            "\"12 June\" or \"next 3 days\".",
    "archive": "I do not keep records that far back for that place. Try a more recent date.",
    "unknown": "Something went wrong at my end and I could not finish that. Try again in a "
               "moment.",
}
ADVICE = "Try a nearby town or district, or share your location."

# The pipeline's `failed_at` stage names, mapped onto the lines above
TROUBLE_KINDS = {"locations": "location", "fetch": "data", "model": "unknown",
                 "pipeline": "unknown"}

# Weather a no-data reply has no business mentioning. Checked against what it was given, so
# "the weather service is not answering" may still say "weather" - anything it added is
# invented, because there was nothing to invent it from.
WEATHER_WORDS = ("rain", "sun", "temperature", "degree", "°", "mm", "hot", "cold", "warm",
                 "wind", "humid", "storm", "cloud", "snow", "shower", "forecast", "likely")

# Scaffolding the prompt forbids and a small model writes anyway. Cheap to check, and each one
# is a phrase no person answering a weather question would say out loud.
SCAFFOLDING = ("the data show", "the data indicate", "according to", "based on the reading",
               "based on the data", "as an ai", "i cannot provide", "here is the",
               "here's the", "in summary", "to summarize", "the provided", "the system",
               "the conclusion", "retrieved", "the readings show", "the figures show",
               "the evidence")

# A reply that reverses the verdict it was handed. The advice path is the one where this is
# not a style problem: asked whether clothes would dry, a 1b model was given "No - 2.4mm
# expected from 06:00" and answered "making it ideal for drying clothes". Someone acts on that.
#
# ponytail: a word list, not an entailment model. It catches the confident reversals, which are
# the dangerous ones; a subtly hedged reversal gets through. Upgrade the model before upgrading
# this - the failure is capability, not detection.
REVERSALS = {
    "NO": ("ideal", "good time", "good window", "go ahead", "safe to", "you can ", "perfect",
           "fine to", "no problem", "should be fine", "suitable"),
    "YES": ("not a good", "avoid", "hold off", "do not", "don't ", "wait until", "unsuitable",
            "not ideal", "not safe", "postpone"),
}

# A CAUTION cannot be reversed, only flattened - and flattening is the same harm by a quieter
# route. Handed "It keeps breaking up - the longest clear spell is 2 days and you need 3", a
# 1b model opened "You should harvest now". The condition is the entire content of a caution,
# so a reply carrying none of these words has dropped it.
HEDGES = ("but", "though", "however", "only", "watch", "keep an eye", "if ", "tight",
          "not enough", "might", "may ", "risk", "short", "careful", "unless")

MAX_REPLY_CHARS = 700      # four sentences with figures in - the prompt asks for 2-4
NUM_PREDICT = 700          # covers the reasoning AND the answer, hence the room
# Not 0. At temperature 0 the likeliest continuation of "rewrite this sentence" is that
# sentence, and a small model duly echoed the conclusion back verbatim - correct, and no
# better than not calling it. Low enough that the figures are still copied exactly.
TEMPERATURE = 0.4


def trouble_line(kind: str) -> str:
    """The line for a pipeline stage name, or for a line's own name. Never raises."""
    return TROUBLE_LINES.get(TROUBLE_KINDS.get(kind or "", kind or ""),
                             TROUBLE_LINES["unknown"])


def clean(said: str) -> str:
    """Strip the thinking block and fences, and reject a reply that is not a rewrite.

    qwen3 emits <think>...</think> even with think=false on some builds, and a small model
    asked to be brief sometimes answers *about* the sentence instead of rewriting it. A reply
    longer than the original is that failure, so it is dropped rather than shown.
    """
    # ponytail: length is the only check - a 0.6b will sometimes drop a trailing caveat. If
    # that matters, append the caveats after say() instead of feeding them in.
    said = re.sub(r"<think>.*?</think>", "", said or "", flags=re.S)
    said = re.sub(r"^```\w*|```$", "", said.strip(), flags=re.M).strip().strip('"')
    said = " ".join(said.split())
    return "" if len(said) > MAX_REPLY_CHARS else said


def first_sentence(said: str) -> str:
    """A suggestion is one question, not a paragraph.

    Everything after it is where a small model puts what it made up - offered "Ondia" for
    "nodia", it went on to add "if you're in Madhya Pradesh, that's a nearby town", which it
    had no way of knowing and which happens to be false.
    """
    match = re.search(r"^.*?[?.!]", (said or "").strip())
    return match.group(0).strip() if match else (said or "").strip()


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def figures_in(text: str) -> set:
    """Every number in a piece of text, as written. "12.5mm" -> {"12.5"}."""
    return set(_NUMBER.findall(text or ""))


def _rounds_to(written: str, given: float) -> bool:
    """Is `written` what `given` looks like, rounded to the precision `written` was written at?

    The precision of the reply is what sets the tolerance, which is what makes this both
    permissive and safe. "2" is written to the nearest whole number, so it may stand for
    anything within 0.5 - and 1.93 qualifies. "2.4" is written to a tenth, so it may only
    stand for something within 0.05 - and 1.93 does not. A model may therefore say "about
    2mm" for 1.93mm, and may not say "2.4mm".
    """
    try:
        value = float(written)
    except ValueError:
        return False
    decimals = len(written.split(".")[1]) if "." in written else 0
    return abs(value - given) <= 0.5 * (10 ** -decimals)


def grounded(said: str, given: str) -> bool:
    """True when every number in the reply is one it was given, or that number rounded.

    The whole architecture rests on the model wording facts rather than producing them, and
    until this existed nothing enforced that on the answering path - only on the no-data one.
    A 1b model handed "Visakhapatnam Rural 12.5mm against Guntur 9.9mm" answered "about 3mm of
    rain is expected in Visakhapatnam Rural", which is a fabricated forecast delivered in a
    confident sentence under a correct table.

    Rounding is allowed because the prompt asks for it: "1.93mm" is not how anyone speaks, and
    a guard that rejected "about 2mm" would have made the wording layer pointless in the name
    of protecting it. What is caught is a number that is not any given figure at any sensible
    precision - which is what an invented one looks like.

    Numbers that are part of the wording rather than a reading ("2 of 7 readings") pass for
    free, because they came from the conclusion too. The cost of a false positive is the
    deterministic sentence, which is what the reader would have got anyway.
    """
    allowed = figures_in(given)
    if not (numbers := figures_in(said) - allowed):
        return True
    values = []
    for figure in allowed:
        try:
            values.append(float(figure))
        except ValueError:
            continue
    return all(any(_rounds_to(n, v) for v in values) for n in numbers)


def invented(said: str, given: str) -> bool:
    """True when a no-data reply contains weather, or any figure, that its input did not.

    Biased hard towards rejection: a false positive costs the fixed wording, a false negative
    puts a made-up forecast in front of someone deciding whether to take a coat.
    """
    low, given = said.lower(), given.lower()
    return (bool(re.search(r"\d", said)) and not re.search(r"\d", given)) or \
        any(word in low and word not in given for word in WEATHER_WORDS)


async def probe() -> dict:
    """Is the local model actually reachable, and is the configured name one it has?

    Worth a network call at startup because every failure in `stream` below is deliberately
    silent - it falls back to the deterministic sentence, which is correct but blunt. That is
    the right behaviour mid-turn and the wrong behaviour for a deployment: a one-character
    typo in OLLAMA_MODEL (`qwen3:1.7bJ`) degraded every single answer for as long as nobody
    happened to compare the wording to what the rules produce. Nothing said anything, because
    from inside a turn a missing model and a slow model look identical.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{OLLAMA_URL}/api/tags")
            response.raise_for_status()
            served = response.json().get("models", [])
            names = [m.get("name", "") for m in served]
            capabilities = {m.get("name", ""): m.get("capabilities") or [] for m in served}
    except (httpx.HTTPError, ValueError) as exc:
        return {"ok": False, "model": OLLAMA_MODEL,
                "note": f"no local model at {OLLAMA_URL} ({type(exc).__name__}) - answers will "
                        f"be the rule-built sentences"}
    if OLLAMA_MODEL not in names:
        return {"ok": False, "model": OLLAMA_MODEL, "available": names,
                "note": f"OLLAMA_MODEL={OLLAMA_MODEL!r} is not installed. Available: "
                        f"{', '.join(names) or 'none'}. Answers will be the rule-built "
                        f"sentences until this matches."}
    global _THINKS
    if _THINKS is None:
        _THINKS = "thinking" in capabilities.get(OLLAMA_MODEL, [])
    return {"ok": True, "model": OLLAMA_MODEL, "thinking": bool(_THINKS)}


# Whether the configured model accepts `think`. None until something finds out. Ollama rejects
# the field outright on a model that has no reasoning mode ("gemma3:1b does not support
# thinking", HTTP 400), and hard-coding think=True meant swapping the model in .env silently
# turned the whole wording layer off - the 400 was caught, nothing was streamed, and the
# rule-built sentence went out looking exactly like a working answer.
_THINKS: bool | None = None


def _body(summary: str, question: str, context: str, answering: bool, think: bool) -> dict:
    body = {"model": OLLAMA_MODEL, "stream": True,
            "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
            "messages": [
                {"role": "system",
                 "content": prompts.system(answering=answering, grounded=bool(context))},
                {"role": "user",
                 "content": prompts.user(summary, question, context, answering=answering)}]}
    if think:
        # the reasoning is shown, not hidden, so the wait has something in it
        body["think"] = True
    return body


async def stream(summary: str, question: str = "", client: httpx.AsyncClient | None = None,
                 context: str = "", answering: bool = True):
    """Yield `(kind, text)` pieces as the local model writes them.

    `kind` is "thinking" for the model's reasoning and "answer" for the reply it settles on.
    Ollama streams the two on separate fields, so the reasoning is shown as reasoning rather
    than filtered out of the answer or leaked into it.

    Cleaning is applied to the answer so far and only the *growth* is yielded, so a pass that
    shortened the text cannot un-say what the reader already read. A failed call yields
    nothing at all and the caller falls back to `summary` - half a sentence is worse than the
    deterministic one.
    """
    global _THINKS

    if not summary or not summary.strip():
        return
    owns = client is None
    client = client or httpx.AsyncClient(timeout=OLLAMA_TIMEOUT)
    seen = sent = ""
    try:
        # Ask for reasoning unless we already learned this model cannot do it, and downgrade
        # once on the 400 that says so. One retry, then it is remembered for the process.
        for think in ([True, False] if _THINKS is None else [bool(_THINKS)]):
            try:
                async with client.stream(
                    "POST", f"{OLLAMA_URL}/api/chat", timeout=OLLAMA_TIMEOUT,
                    json=_body(summary, question, context, answering, think),
                ) as response:
                    response.raise_for_status()
                    _THINKS = think
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        try:
                            message = json.loads(line).get("message", {})
                        except json.JSONDecodeError:
                            continue
                        # reasoning goes out raw and unjudged - it is shown as thinking, and
                        # cleaning it would only make it look like an answer
                        if (thought := message.get("thinking")):
                            yield "thinking", thought
                        if not (piece := message.get("content", "")):
                            continue
                        seen += piece
                        cleaned = clean(seen)
                        # only ever append: a cleaning pass that shortened the text (a fence
                        # closing, the length guard tripping) must not un-say what was read
                        if cleaned.startswith(sent) and len(cleaned) > len(sent):
                            yield "answer", cleaned[len(sent):]
                            sent = cleaned
                return
            except httpx.HTTPStatusError as exc:
                if not (think and exc.response.status_code == 400):
                    raise
                _THINKS = False                  # this model has no reasoning mode; carry on
    except (httpx.HTTPError, KeyError, ValueError):
        return
    finally:
        if owns:
            await client.aclose()


async def say(summary: str, question: str = "", client: httpx.AsyncClient | None = None,
              context: str = "", answering: bool = True) -> str:
    """`summary` in plain words, or `summary` itself if the local model cannot answer."""
    said = "".join([text async for kind, text in stream(summary, question, client, context,
                                                        answering) if kind == "answer"])
    return usable(said, summary, summary + " " + context) or summary


def echoed(said: str, summary: str) -> bool:
    """True when the reply is the input back again rather than a rewrite of it.

    Discarded rather than shown: an echo is the deterministic sentence with a round trip to a
    model in front of it, so returning the deterministic sentence loses nothing and keeps the
    two paths producing one recognisable voice instead of two.
    """
    strip = lambda s: " ".join((s or "").lower().split()).strip(" .!?")
    return bool(said) and strip(said) == strip(summary)


def scaffolded(said: str) -> bool:
    """True when the reply talks about where the answer came from instead of giving it."""
    low = (said or "").lower()
    return any(phrase in low for phrase in SCAFFOLDING)


def reverses(said: str, verdict: str) -> bool:
    """True when the reply says the opposite of the verdict it was handed.

    For YES and NO that is a reversal; for CAUTION it is a reply with no condition in it at
    all, which turns "you can, but only just" into "you can".
    """
    low = (said or "").lower()
    if verdict == "CAUTION":
        return not any(hedge in low for hedge in HEDGES)
    return any(word in low for word in REVERSALS.get(verdict or "", ()))


def usable(said: str, summary: str, given: str, verdict: str = "") -> str:
    """The reply, or "" when it must not be shown. One gate, so every caller checks the same
    four things: it exists, it says something new, it invents no figure, it says what was
    decided, and it does not narrate the machinery."""
    if not said or echoed(said, summary) or scaffolded(said):
        return ""
    if not grounded(said, given) or reverses(said, verdict):
        return ""
    return said


async def explain(kind: str, question: str = "", near: list | None = None,
                  client: httpx.AsyncClient | None = None) -> str:
    """A failure, said as something a person can act on instead of a stack trace.

    Goes through the model under the no-data half of the prompt, and is checked on the way out
    because this is the one reply that must not contain weather. The question is deliberately
    withheld: given it, the model answers it.
    """
    line = trouble_line(kind)
    if not near:
        # Nothing was retrieved, so there is nothing for generation to add - and measured, a
        # 0.6b asked to re-say a bare line only takes away: "the weather service is not
        # answering" came back as "I couldn't complete the task", losing the one specific
        # thing the reader wanted to know. Augment or don't call it.
        return line
    listed = "\n".join(f"- {name}" for name in near)
    said = first_sentence(await say(line, "", client, context=listed, answering=False))
    # The offer has to be one of the names retrieved. Given "Belamguda (Rayagarha, Odisha)" a
    # 0.6b offered "did you mean Rayagarha?" - the district out of the brackets, which is not
    # a place anyone typed or meant.
    names = [entry.split(" (")[0].strip().lower() for entry in near]
    if not said or invented(said, line + " " + listed) or \
            not any(name and name in said.lower() for name in names):
        return line
    return f"{said} If not, {ADVICE[0].lower() + ADVICE[1:]}"


def demo():
    """Self-check: the cleaning rules offline, then one live call if Ollama is up."""
    assert clean("<think>hmm</think>\nRain tomorrow.") == "Rain tomorrow."
    assert clean('```\n"Rain tomorrow."\n```') == "Rain tomorrow."
    assert clean("x" * (MAX_REPLY_CHARS + 1)) == ""          # a lecture, not a reply
    assert clean("x" * MAX_REPLY_CHARS) != ""                # ...but four sentences fit
    assert clean("") == ""
    assert first_sentence("Did you mean Vedurumudi? It is in East Godavari.") == \
        "Did you mean Vedurumudi?"

    # the no-data guard: weather or a figure the note did not carry is invented
    assert invented("It will rain tomorrow.", "I could not find that place.")
    assert invented("Try again in 5 minutes.", "I could not find that place.")
    assert not invented("I could not find that place.", "I could not find that place.")

    # The fact guard. A figure it was given, or that figure rounded the way people speak, is
    # fine; anything else is a fabricated forecast in a confident sentence.
    given = "Visakhapatnam Rural has the higher rainfall for this week: 12.5mm against Guntur 9.9mm."
    assert grounded("Vizag is wetter this week, 12.5mm to Guntur's 9.9mm.", given)
    assert grounded("Vizag is the wetter of the two this week.", given)      # no figures at all
    assert grounded("Vizag gets about 13mm this week, Guntur about 10mm.", given)   # rounded
    assert not grounded("About 3mm of rain is expected in Visakhapatnam Rural.", given)
    assert not grounded("Vizag gets 14mm this week.", given)                 # too far to be 12.5
    assert not grounded("Rain starts at 2pm.", given)                        # nowhere in scope

    # precision sets the tolerance: a whole number may stand for anything within half of one,
    # a tenth may only stand for something within a twentieth
    assert _rounds_to("2", 1.93) and _rounds_to("1.9", 1.93)
    assert not _rounds_to("2.4", 1.93)
    assert grounded("About 2mm tomorrow.", "Guntur, tomorrow: 1.93mm.")
    assert not grounded("About 2.4mm tomorrow.", "Guntur, tomorrow: 1.93mm.")
    assert grounded("In the low twenties, about 26 degrees.", "25.3-27.2°C (avg 26.3°C)")

    assert figures_in("25.3-27.2°C (avg 26.3°C)") == {"25.3", "27.2", "26.3"}

    # The reversal gate. This is the one that is not about style: asked whether clothes would
    # dry, a 1b handed "No - 2.4mm expected from 06:00" answered "making it ideal for drying
    # clothes", and somebody hangs washing out on that.
    verdict_line = "No - 2.4mm expected from 06:00. Guntur, today: rain on 1 of 23 readings."
    context = verdict_line + " Decision: No - 2.4mm expected from 06:00"
    assert reverses("...making it ideal for drying clothes.", "NO")
    assert reverses("Hold off on spraying today.", "YES")
    assert not reverses("You will want to wait - rain from six.", "NO")
    # a caution is flattened, not reversed: dropping the condition is dropping the answer
    assert reverses("You should harvest now.", "CAUTION")
    assert not reverses("You can harvest, but the dry spell is only two days.", "CAUTION")
    assert not reverses("Tight - watch the rain on Thursday.", "CAUTION")
    assert not reverses("anything at all", ""), "not an advice turn, nothing to reverse"

    # ...and the one gate every caller goes through
    assert usable("...making it ideal for drying clothes.", verdict_line, context, "NO") == ""
    assert usable(verdict_line, verdict_line, context) == "", "an echo is not a rewrite"
    assert usable("The data shows 2.4mm of rain.", verdict_line, context) == ""
    assert usable("", verdict_line, context) == ""
    kept = "You will want to wait - about 2.4mm is coming from six, so nothing dries today."
    assert usable(kept, verdict_line, context + " 2.4", "NO") == kept

    assert scaffolded("According to the forecast, rain.") and not scaffolded("Rain tomorrow.")
    # the model narrating its own brief is the same failure wearing different words
    assert scaffolded("The conclusion is correct based on the retrieved data.")
    assert scaffolded("The evidence shows 2mm.")

    assert trouble_line("locations") == TROUBLE_LINES["location"]
    assert trouble_line("fetch") == TROUBLE_LINES["data"]
    assert trouble_line("nonsense") == TROUBLE_LINES["unknown"]
    # a bare failure never reaches the model - there is nothing to augment it with
    assert asyncio.run(explain("fetch")) == TROUBLE_LINES["data"]

    assert asyncio.run(say("")) == ""                         # empty in, empty out, no call
    line = ("Safe to spray. Wind speed in Guntur for tomorrow: 1.2-3.4m/s (avg 2.1m/s). "
            "(No exact match for \"guntur\" - showing the closest, Guntur, Andhra Pradesh.)")
    said = asyncio.run(say(line))
    print(f"  in : {line}\n  out: {said}")
    assert echoed(line, line) and not echoed("Rain tomorrow.", line)
    if said != line and echoed(said, line):
        print("  WARNING: the model echoed its input instead of rewriting it - check the "
              "prompt and OLLAMA_MODEL")

    # the retrieved sections must reach the answer: the conclusion names one place, the reply
    # has to be able to name the other
    both = asyncio.run(say(
        "Hyderabad has a higher minimum temperature of 21.3C compared to Vijawada's 20.5C.",
        "compare between hyderabad and vijawada",
        context="Places: Hyderabad (Telangana), Vijawada (Andhra Pradesh)\n"
                "Period: next 7 days (daily readings)\n"
                "Comparison: Hyderabad leads Vijawada by 0.8°C on min temp\n"
                "Range:\n- Hyderabad: average min temp 21.3°C, range 20.6-21.9°C\n"
                "- Vijawada: average min temp 20.5°C, range 19.7-20.9°C"))
    print(f"  both: {both}")
    if said != line:                                          # ollama answered - a real test
        assert "vijawada" in both.lower(), both
    print("generation demo OK" + (" (ollama offline - passed through)" if said == line else ""))


if __name__ == "__main__":
    demo()
