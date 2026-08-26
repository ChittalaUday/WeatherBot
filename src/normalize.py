"""
Text normalizer - the cheapest layer, and the first one.

Runs before any model: SMS shorthand, repeated characters, punctuation and the common
misspellings of weather words are folded away so the classifier sees one spelling instead of
twenty. Every rewrite is recorded, so a wrong answer can always be traced back to the words
the user actually typed.

    "What's da wthr in KKD tmrw??"
      -> "what is the weather in KKD tomorrow?"   [[da, the], [wthr, weather], [tmrw, tomorrow]]

Two things are deliberately NOT normalized here:

  - place names. "KKD" stays "KKD": the model reports what the user said (Rule 4.1) and
    backend/locations.py resolves it. Rewriting it here would hide the raw span from the
    span tagger and from the audit trail.
  - anything ambiguous. "mon" could be Monday or a name; leave it for the model.
"""

from __future__ import annotations

import re

from src.schema import Normalized

# word -> canonical word. Only unambiguous, weather-domain rewrites belong here.
LEXICON = {
    # chat shorthand
    "u": "you", "ur": "your", "r": "are", "pls": "please", "plz": "please", "thx": "thanks",
    "da": "the", "dis": "this", "n": "and", "abt": "about", "bcz": "because", "coz": "because",
    "wud": "would", "shud": "should", "cud": "could", "wat": "what", "wats": "whats",
    "hw": "how", "y": "why", "k": "ok",
    # weather vocabulary
    "wthr": "weather", "wether": "weather", "weathr": "weather", "wheather": "weather",
    "temp": "temperature", "tempature": "temperature", "temprature": "temperature",
    "tempreture": "temperature", "temparature": "temperature",
    "humidty": "humidity", "humdity": "humidity", "humididty": "humidity",
    "forcast": "forecast", "forecaste": "forecast", "forcaste": "forecast",
    "rian": "rain", "rainfal": "rainfall", "precipitaion": "precipitation",
    "sunshien": "sunshine", "sunlite": "sunlight", "cloudness": "cloudiness",
    "wnd": "wind", "windspeed": "wind speed", "dewpoit": "dew point",
    "moistur": "moisture", "moisure": "moisture", "mositure": "moisture",
    # time shorthand (the resolver still gets the canonical form from tagger.normalize_time)
    "tmrw": "tomorrow", "tmr": "tomorrow", "tomm": "tomorrow", "tomo": "tomorrow",
    "2moro": "tomorrow", "2morrow": "tomorrow", "tomorow": "tomorrow",
    "tommorow": "tomorrow", "tommorrow": "tomorrow", "tomarrow": "tomorrow",
    "2day": "today", "2nite": "tonight", "tonite": "tonight", "rn": "right now",
    "nxt": "next", "yest": "yesterday", "wk": "week", "mrng": "morning", "evng": "evening",
    "aftrnoon": "afternoon", "nite": "night",
}
CONTRACTIONS = {
    "what's": "what is", "whats": "what is", "whatis": "what is", "hows": "how is",
    "how's": "how is", "it's": "it is", "isn't": "is not", "won't": "will not",
    "don't": "do not", "doesn't": "does not", "can't": "can not", "i'm": "i am",
    "there's": "there is", "let's": "let us", "we're": "we are", "you're": "you are",
    "gonna": "going to", "wanna": "want to", "gimme": "give me",
}

# Curly quotes and friends: "What’s" must fold to the same token as "What's".
UNICODE_FIXES = {"’": "'", "‘": "'", "“": '"', "”": '"',
                 "–": "-", "—": "-", " ": " ", "°": " degrees "}

_TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?::\d+)?|[^\sA-Za-z\d]")
_REPEAT = re.compile(r"([a-z])\1{2,}")          # "sooooo" -> "soo", never "ss" -> "s"


def _squash_repeats(word: str) -> str:
    return _REPEAT.sub(r"\1\1", word)


def normalize(text: str) -> Normalized:
    """Fold a message to canonical wording, keeping a record of every substitution."""
    original = text
    for bad, good in UNICODE_FIXES.items():
        text = text.replace(bad, good)
    text = re.sub(r"\s+", " ", text).strip()

    replacements: list[list[str]] = []
    out: list[str] = []
    for token in _TOKEN.findall(text):
        lowered = token.lower()

        if lowered in CONTRACTIONS:
            replacements.append([token, CONTRACTIONS[lowered]])
            out.append(CONTRACTIONS[lowered])
            continue

        squashed = _squash_repeats(lowered)
        if squashed != lowered and squashed in LEXICON:
            lowered = squashed
        elif squashed != lowered:
            replacements.append([token, squashed])
            out.append(squashed)
            continue

        if lowered in LEXICON:
            # keep the original casing style: ALL CAPS stays shouting, Title stays Title
            replacement = LEXICON[lowered]
            if token.isupper() and len(token) > 1:
                replacement = replacement.upper()
            replacements.append([token, replacement])
            out.append(replacement)
            continue

        out.append(token)

    # rebuild with sane spacing: no space before , . ? ! : ' or after an opening bracket
    rebuilt = ""
    for token in out:
        if not rebuilt:
            rebuilt = token
        elif token in ",.?!:;" or token.startswith("'"):
            rebuilt += token
        else:
            rebuilt += " " + token

    rebuilt = re.sub(r"\s+", " ", rebuilt).strip()
    return Normalized(original=original, normalized=rebuilt, replacements=replacements)
