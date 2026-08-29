"""
The last layer: the deterministic conclusion, said in plain words by a local model.

    python tests/test_generation_units.py          # the checks for this module
    ollama pull qwen3:1.7b                  # OLLAMA_MODEL, OLLAMA_URL, OLLAMA_TIMEOUT in .env

The numbers, the verdict and the caveats are still decided by rules. This only re-says what
they produced, from the sections `backend.generation.context` retrieved, so it cannot state a
figure the pipeline did not already stand behind.

Any failure returns the original sentence untouched: a chat that answers slightly less well is
a worse day; a chat that answers nothing is a broken product.
"""

from __future__ import annotations

import json
import re

import httpx

from backend.config import OLLAMA_MODEL, OLLAMA_THINK, OLLAMA_TIMEOUT, OLLAMA_URL
from backend.generation import prompts

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

# Weather a no-data reply has no business mentioning, checked against what it was given -
# anything it added is invented, because there was nothing to invent it from.
WEATHER_WORDS = ("rain", "sun", "temperature", "degree", "°", "mm", "hot", "cold", "warm",
                 "wind", "humid", "storm", "cloud", "snow", "shower", "forecast", "likely")

# Scaffolding the prompt forbids and a small model writes anyway.
SCAFFOLDING = ("the data show", "the data indicate", "according to", "based on the reading",
               "based on the data", "as an ai", "i cannot provide", "here is the",
               "here's the", "in summary", "to summarize", "the provided", "the system",
               "the conclusion", "retrieved", "the readings show", "the figures show",
               "the evidence")

REVERSALS = {
    "NO": ("ideal", "good time", "good window", "go ahead", "safe to", "you can ", "perfect",
           "fine to", "no problem", "should be fine", "suitable"),
    "YES": ("not a good", "avoid", "hold off", "do not", "don't ", "wait until", "unsuitable",
            "not ideal", "not safe", "postpone"),
}

HEDGES = ("but", "though", "however", "only", "watch", "keep an eye", "if ", "tight",
          "not enough", "might", "may ", "risk", "short", "careful", "unless")

MAX_REPLY_CHARS = 900      # the prompt asks for length to follow the question, not a cap
NUM_PREDICT = 700          # covers the reasoning AND the answer, hence the room
# Not 0: at temperature 0 the likeliest continuation of "rewrite this sentence" is that
# sentence, and a small model echoed the conclusion back verbatim.
TEMPERATURE = 0.4


def trouble_line(kind: str) -> str:
    """The line for a pipeline stage name, or for a line's own name. Never raises."""
    return TROUBLE_LINES.get(TROUBLE_KINDS.get(kind or "", kind or ""),
                             TROUBLE_LINES["unknown"])


def clean(said: str, structured: bool = False) -> str:
    """Strip the thinking block and fences, and reject a reply that is not a rewrite.

    qwen3 emits <think>...</think> even with think=false on some builds, and a reply longer
    than the original is a model answering *about* the sentence instead of rewriting it.

    `structured` keeps the line breaks. Collapsing whitespace is right for a paragraph and
    destroys a bullet list - it is what turns six findings back into one run-on sentence.
    """
    said = re.sub(r"<think>.*?</think>", "", said or "", flags=re.S)
    said = re.sub(r"^```\w*|```$", "", said.strip(), flags=re.M).strip().strip('"')
    if structured:
        # blank lines collapse to one, trailing spaces go, the newlines stay
        said = re.sub(r"\n{3,}", "\n\n", "\n".join(line.rstrip() for line in said.split("\n")))
    else:
        said = " ".join(said.split())
    cap = MAX_REPLY_CHARS * 2 if structured else MAX_REPLY_CHARS
    return "" if len(said) > cap else said


def first_sentence(said: str) -> str:
    """A suggestion is one question, not a paragraph - everything after it is where a small
    model puts what it made up, and what it makes up here is a place that does not exist."""
    match = re.search(r"^.*?[?.!]", (said or "").strip())
    return match.group(0).strip() if match else (said or "").strip()


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def figures_in(text: str) -> set:
    """Every number in a piece of text, as written. "12.5mm" -> {"12.5"}."""
    return set(_NUMBER.findall(text or ""))


def _rounds_to(written: str, given: float) -> bool:
    """Is `written` what `given` looks like, rounded to the precision `written` was written at?

    The reply's own precision sets the tolerance: "2" may stand for anything within 0.5, so
    1.93 qualifies; "2.4" may only stand for something within 0.05, so it does not.
    """
    try:
        value = float(written)
    except ValueError:
        return False
    decimals = len(written.split(".")[1]) if "." in written else 0
    return abs(value - given) <= 0.5 * (10 ** -decimals)


def grounded(said: str, given: str) -> bool:
    """True when every number in the reply is one it was given, or that number rounded.

    Handed "Visakhapatnam Rural 12.5mm against Guntur 9.9mm", a 1b model answered "about 3mm
    of rain is expected in Visakhapatnam Rural" - a fabricated forecast under a correct table.

    Rounding is allowed because the prompt asks for it; what is caught is a number that is not
    any given figure at any sensible precision. A false positive costs the deterministic
    sentence, which is what the reader would have got anyway.
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

    Biased hard towards rejection: a false negative puts a made-up forecast in front of
    someone deciding whether to take a coat.
    """
    low, given = said.lower(), given.lower()
    return (bool(re.search(r"\d", said)) and not re.search(r"\d", given)) or \
        any(word in low and word not in given for word in WEATHER_WORDS)


async def probe() -> dict:
    """Is the local model actually reachable, and is the configured name one it has?

    Every failure in `stream` is deliberately silent, which is right mid-turn and wrong for a
    deployment: a typo in OLLAMA_MODEL (`qwen3:1.7bJ`) degraded every answer and nothing said
    so, because from inside a turn a missing model and a slow model look identical.
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
    if not OLLAMA_THINK:
        _THINKS = False
    return {"ok": True, "model": OLLAMA_MODEL, "thinking": bool(_THINKS)}


_THINKS: bool | None = None if OLLAMA_THINK else False


def _body(summary: str, question: str, context: str, answering: bool, think: bool,
          deciding: bool = False, history: list | None = None, headings=(),
          structured: bool = False) -> dict:
    """The request for one turn. `think` decides the reasoning block as well as the field:
    telling a model with no reasoning mode what to reason about is tokens it cannot spend."""
    body = {"model": OLLAMA_MODEL, "stream": True,
            "options": {"temperature": TEMPERATURE, "num_predict": NUM_PREDICT},
            "messages": [
                {"role": "system",
                 "content": prompts.system(answering=answering, grounded=bool(context),
                                           deciding=deciding, thinking=think,
                                           structured=structured, headings=headings)},
                {"role": "user",
                 "content": prompts.user(summary, question, context, answering=answering,
                                         history=history)}]}
    # Always sent: omitted, a reasoning model reasons anyway and the "cannot think" fallback
    # costs exactly as much as the path it was falling back from.
    body["think"] = think
    return body


async def stream(summary: str, question: str = "", client: httpx.AsyncClient | None = None,
                 context: str = "", answering: bool = True, deciding: bool = False,
                 history: list | None = None, headings=(), structured: bool = False):
    """Yield `(kind, text)` pieces as the local model writes them.

    `kind` is "thinking" for the reasoning and "answer" for the reply - Ollama streams the two
    on separate fields. Only the *growth* of the cleaned answer is yielded, so a pass that
    shortened it cannot un-say what the reader already read. A failed call yields nothing.
    """
    global _THINKS

    if not summary or not summary.strip():
        return
    owns = client is None
    client = client or httpx.AsyncClient(timeout=OLLAMA_TIMEOUT)
    seen = sent = ""
    try:
        for think in ([True, False] if _THINKS is None else [_THINKS]):
            try:
                async with client.stream(
                    "POST", f"{OLLAMA_URL}/api/chat", timeout=OLLAMA_TIMEOUT,
                    json=_body(summary, question, context, answering, think, deciding,
                               history, headings, structured),
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
                        if (thought := message.get("thinking")):
                            yield "thinking", thought
                        if not (piece := message.get("content", "")):
                            continue
                        seen += piece
                        cleaned = clean(seen, structured)
                        if cleaned.startswith(sent) and len(cleaned) > len(sent):
                            yield "answer", cleaned[len(sent):]
                            sent = cleaned
                return
            except httpx.HTTPStatusError as exc:
                if not (think and exc.response.status_code == 400):
                    raise
                _THINKS = False                  
    except (httpx.HTTPError, KeyError, ValueError):
        return
    finally:
        if owns:
            await client.aclose()


async def say(summary: str, question: str = "", client: httpx.AsyncClient | None = None,
              context: str = "", answering: bool = True, deciding: bool = False,
              history: list | None = None, headings=(), structured: bool = False) -> str:
    """`summary` in plain words, or `summary` itself if the local model cannot answer."""
    said = "".join([text async for kind, text in stream(summary, question, client, context,
                                                        answering, deciding, history,
                                                        headings, structured)
                    if kind == "answer"])
    return said if said else summary


async def say_conversational(question: str, intent: str = "", family: str = "",
                               history: list | None = None, fallback: str = "",
                               client: httpx.AsyncClient | None = None) -> str:
    """A greeting or a control turn, worded by the local model from the chat so far."""
    if not question or not question.strip():
        return fallback

    owns = client is None
    client = client or httpx.AsyncClient(timeout=OLLAMA_TIMEOUT)
    try:
        body = {
            "model": OLLAMA_MODEL,
            "stream": False,
            "options": {"temperature": 0.7, "num_predict": 300},
            "messages": [
                {"role": "system", "content": prompts.conversational_system()},
                {"role": "user", "content": prompts.conversational_user(question, intent=intent, family=family, history=history)},
            ],
            "think": False,
        }
        response = await client.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=OLLAMA_TIMEOUT)
        response.raise_for_status()
        res_data = response.json()
        said = (res_data.get("message", {}).get("content") or "").strip()
        cleaned_said = clean(said)
        return cleaned_said if cleaned_said else fallback
    except Exception as exc:
        print(f"say_conversational fallback: {type(exc).__name__}: {exc}", flush=True)
        return fallback
    finally:
        if owns:
            await client.aclose()



ECHO_RUN = 8


def echoed(said: str, summary: str) -> bool:
    """True when the reply is the input back again rather than an answer built from it.

    Exact equality caught almost nothing - a small model returns the sentence with the dashes
    turned into commas, a different string and the same non-answer - so run length is what is
    measured. An echo is discarded, not shown: it is the deterministic sentence with a round
    trip in front of it.
    """
    words = lambda s: re.findall(r"[a-z0-9.]+", (s or "").lower())
    said_words, given = words(said), words(summary)
    if not said_words:
        return False
    if said_words == given:
        return True
    if len(said_words) < ECHO_RUN:
        return False
    runs = {tuple(given[i:i + ECHO_RUN]) for i in range(len(given) - ECHO_RUN + 1)}
    return any(tuple(said_words[i:i + ECHO_RUN]) in runs
               for i in range(len(said_words) - ECHO_RUN + 1))


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


# A bullet at the start of a line, or a bold run. Enough to tell a laid-out reply from a
# paragraph, and nothing more - the renderer only needs to know which of the two it has.
_MARKDOWN = re.compile(r"^\s*[-*+]\s+\S|\*\*\S", re.M)


def is_markdown(said: str) -> bool:
    """Did the reply actually come back laid out?

    Asked for a layout, a model that finds one sentence is the honest answer writes one
    sentence - which is right, and which makes the layout flag a lie if it is set from the
    request. So it is read off the reply instead.
    """
    return bool(_MARKDOWN.search(said or ""))


def usable(said: str, summary: str, given: str, verdict: str = "") -> str:
    """The reply, or "" when it must not be shown. One gate: it exists, it says something new,
    it invents no figure, it keeps the verdict, and it does not narrate the machinery."""
    if not said or echoed(said, summary) or scaffolded(said):
        return ""
    if not grounded(said, given) or reverses(said, verdict):
        return ""
    return said


async def explain(kind: str, question: str = "", near: list | None = None,
                  client: httpx.AsyncClient | None = None) -> str:
    """A failure, said as something a person can act on instead of a stack trace.

    Checked on the way out because this is the one reply that must not contain weather. The
    question is deliberately withheld: given it, the model answers it.
    """
    line = trouble_line(kind)
    if not near:
        return line
    listed = "\n".join(f"- {name}" for name in near)
    said = first_sentence(await say(line, "", client, context=listed, answering=False))
    names = [entry.split(" (")[0].strip().lower() for entry in near]
    if not said or invented(said, line + " " + listed) or \
            not any(name and name in said.lower() for name in names):
        return line
    return f"{said} If not, {ADVICE[0].lower() + ADVICE[1:]}"
