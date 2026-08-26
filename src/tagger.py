"""
BIO span tagger for the LOCATION / TIME entities of MODEL_RULES.md Rules 4.1 / 4.2.

A gazetteer can only return spans it memorised, which is why it finds 13% of the locations
in the hand-written evaluation set. This tagger labels every *token* instead, from context
and word shape - "the capitalised token after `in`, not a metric noun, suffix -pur" - so it
recognises villages it has never seen, in any casing, misspelt or not. The gazetteer stays
on as a feature rather than as the decider.

Deliberately sklearn-only (DictVectorizer + LogisticRegression): a CRF would model label
transitions properly, but B-/I- consistency is nearly free here because entity spans are
short and the surrounding words are highly predictive. Upgrade to sklearn-crfsuite only if
the measured span F1 stops improving.
"""

import difflib
import re
from collections import Counter

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

# "5:30", "10cm", words, and standalone punctuation ("," carries I-LOC inside addresses)
TOKEN_RE = re.compile(r"\d+:\d+|\d+[a-zA-Z]+|[A-Za-z]+|\d+|[^\w\s]")
PREPOSITIONS = {"in", "at", "for", "near", "from", "to", "by", "around", "of", "on", "over"}
FIELDS = ("location", "time")

# Deterministic wall-clock / duration expressions. Unseen clock values need no training
# data - "4:15 pm" is a time whether or not that exact string was ever generated.
CLOCK_RE = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:am|pm)(?:\s*to\s*\d{1,2}(?::\d{2})?\s*(?:am|pm))?\b", re.I)
DURATION_RE = re.compile(
    r"\b(?:next\s+)?\d+\s*(?:days?|hours?|hrs?|minutes?|mins?|weeks?|months?)\b", re.I)


def tokenize(text):
    """[(start, end, token)] - offsets kept so predicted spans are verbatim slices."""
    return [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(text)]


# --- time normalisation ------------------------------------------------------
# The raw span stays verbatim (Rule 4.2); this is the queryable twin of it, so the
# deterministic Time Parser downstream never has to deal with "tommorrow" or "2nite".
# Clock times become 24h HH:MM, ranges HH:MM-HH:MM, durations "next N <unit>". This
# canonicalises the *surface form* only - resolving to an actual datetime stays downstream.

# "whole day", "all week", "the entire month" are span quantifiers: they name a window, not a
# place. A user reported "what is the rainfall fro whole day" resolving "whole" as a village,
# and the root cause was that no training text ever used these - so the tagger read a rare
# word in a location-shaped slot and did the only thing it knew.
SPAN_QUANTIFIERS = {
    f"{qualifier}{quantifier} {unit}": canonical
    for qualifier in ("", "the ")
    for quantifier, in (("whole",), ("entire",), ("full",), ("complete",))
    for unit, canonical in (("day", "today"), ("week", "this week"), ("month", "this month"),
                            ("weekend", "this weekend"), ("night", "tonight"),
                            ("morning", "this morning"), ("afternoon", "this afternoon"),
                            ("evening", "this evening"), ("year", "this month"))
}
# "history", "historical data", "past records" name no period at all but are unambiguously
# about the past. Without these the model saw no time span and defaulted to the forecast.
HISTORY_WORDS = {w: "last 7 days" for w in (
    "history", "historical", "historic", "history data", "historical data",
    "past data", "past records", "previous records", "past readings", "old data",
    "records", "past history", "history of", "in history")}

SPAN_QUANTIFIERS.update(HISTORY_WORDS)
SPAN_QUANTIFIERS.update({
    "all day": "today", "all day long": "today", "all week": "this week",
    "all night": "tonight", "all month": "this month", "the rest of the day": "today",
    "rest of the day": "today", "rest of the week": "this week",
    "remainder of the day": "today", "throughout the day": "today",
    "through the day": "today", "over the day": "today", "for the day": "today",
    "the day": "today", "the week": "this week", "the night": "tonight",
})

TIME_ALIASES = {
    **SPAN_QUANTIFIERS,
    "rn": "now", "right now": "now", "atm": "now", "immediately": "now",
    "2day": "today", "todya": "today", "later today": "today",
    "2nite": "tonight", "tonite": "tonight",
    "tmrw": "tomorrow", "tmr": "tomorrow", "2moro": "tomorrow", "tomorow": "tomorrow",
    "tommorow": "tomorrow", "tommorrow": "tomorrow", "tomarrow": "tomorrow",
    "nxt week": "next week", "coming week": "next week",
    "coming weekend": "next weekend", "the weekend": "this weekend", "weekend": "this weekend",
    "day after tomorow": "day after tomorrow", "day after": "day after tomorrow",
    "overnight": "tonight", "mid night": "midnight",
    # unambiguous code-mixed words (diagnostic only - English is the V1 target)
    "abhi": "now", "aaj": "today", "ee roju": "today", "repu": "tomorrow", "raat": "tonight",
}
CANONICAL_TIMES = {
    "now", "today", "tonight", "tomorrow", "yesterday", "day after tomorrow",
    "this morning", "this afternoon", "this evening", "early morning", "midnight",
    "tomorrow morning", "tomorrow afternoon", "tomorrow evening", "tomorrow night",
    "this week", "next week", "last week", "this weekend", "next weekend",
    "this month", "next month", "monday", "tuesday", "wednesday", "thursday",
    "friday", "saturday", "sunday",
}
_TIME_VOCAB = sorted(CANONICAL_TIMES | set(TIME_ALIASES))
_CLOCK_ONE = re.compile(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", re.I)
_RANGE = re.compile(r"^(.+?)\s+to\s+(.+)$", re.I)
_DURATION = re.compile(r"^(?:next\s+|in\s+)?(\d+)\s*(day|hour|hr|minute|min|week|month)s?$", re.I)
_UNITS = {"hr": "hours", "min": "minutes"}


def _clock24(label):
    """'6:45 pm' -> '18:45'. None when the label is not a wall-clock time."""
    match = _CLOCK_ONE.match(label.strip())
    if not match:
        return None
    hour, minute, meridiem = int(match.group(1)), int(match.group(2) or 0), match.group(3)
    if meridiem:
        hour = hour % 12 + (12 if meridiem.lower() == "pm" else 0)
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def normalize_time(span: str) -> str:
    """One canonical shape per temporal expression, whatever the user typed.

    Unknown expressions pass through lowercased rather than being dropped - a wrong guess
    downstream is worse than an unrecognised string the Time Parser can still inspect.
    """
    text = " ".join(span.lower().split()).strip(" ,.?!")
    # a leading preposition belongs to the sentence, not the time ("on friday" -> "friday")
    # `\s+|$` not just `\s+`: a span of exactly "for" is a preposition the tagger swept up,
    # and it has to normalise to nothing so the caller can drop it. Left as "for" it fuzzy
    # matched a real time word and moved the window.
    text = re.sub(r"^(on|at|in|by|during|for|from|around|till|until)(\s+|$)", "", text)
    if not text:
        return text

    if (range_match := _RANGE.match(text)):
        start, end = (_clock24(part) for part in range_match.groups())
        if start and end:
            return f"{start}-{end}"
    if (clock := _clock24(text)):
        return clock
    if (duration := _DURATION.match(text)):
        count, unit = duration.groups()
        unit = _UNITS.get(unit.lower(), unit.lower() + "s")
        return f"next {count} {unit}"

    if text in TIME_ALIASES:
        return TIME_ALIASES[text]
    if text in CANONICAL_TIMES:
        return text
    # unseen misspelling: nearest canonical form, or pass the raw text through
    close = difflib.get_close_matches(text, _TIME_VOCAB, n=1, cutoff=0.82)
    return TIME_ALIASES.get(close[0], close[0]) if close else text


# Words that sit inside a location span without being a place name ("my field", "near me").
GENERIC_PLACE_WORDS = {"and", "my", "me", "by", "this", "our", "here", "i", "near", "nearby",
                       "area", "place", "field", "farm", "village", "plot", "location",
                       "feild", "vilage", "ka", "ki", "point", "bad",
                       # span quantifiers - these describe how much of a window, never where
                       "whole", "entire", "full", "complete", "all", "rest", "remainder",
                       "throughout", "day", "week", "month", "year", "night"}


def choose_min_word_freq(texts, location_spans, candidates=(20, 40, 80, 120, 160, 200, 300, 500)):
    """Smallest frequency cut at which no real place name survives in the word vocabulary.

    Derived from the training split only - never from the evaluation set. Below this cut the
    tagger can still satisfy training by memorising village names, and it then fails on the
    first village it has not met. Above it, the only signal left is context and word shape,
    which is what actually transfers.
    """
    counts = Counter(w.lower() for text in texts for _, _, w in tokenize(text))
    names = {w.lower() for spans in location_spans for span in spans
             for _, _, w in tokenize(span)}
    names = {w for w in names if w.isalpha()} - GENERIC_PLACE_WORDS
    for cut in candidates:
        if not ({w for w, n in counts.items() if n >= cut} & names):
            return cut
    return candidates[-1]


def _shape(word):
    if word.isupper():
        return "UPPER"
    if word.istitle():
        return "Title"
    if word.islower():
        return "lower"
    if word.isdigit():
        return "digit"
    return "mixed"


def _token_features(tokens, i, gazetteer, metrics, vocab):
    start, end, word = tokens[i]
    low = word.lower()
    feats = {
        # Rare words are hidden behind <rare> so the model cannot solve training by
        # memorising village names: at inference every unseen name looks exactly like a
        # rare training token, and only context and word shape are left to decide.
        "w": low if low in vocab else "<rare>",
        "known_word": low in vocab,
        "shape": _shape(word),
        "suf3": low[-3:], "suf4": low[-4:], "pre3": low[:3],
        # character shingles: "hyderbad" shares hyd/yde/der with "hyderabad", so a
        # misspelt village still looks like a village. Fixed affixes alone miss that -
        # "abad" is a known place ending, "rbad" is not.
        **{f"ng_{low[i:i + 3]}": True for i in range(max(len(low) - 2, 0))},
        "len": str(min(len(word), 10)),
        "has_digit": any(c.isdigit() for c in word),
        "has_colon": ":" in word,
        "is_metric": low in metrics,
        # NOTE: gazetteer membership is deliberately absent - see SpanTagger.fit
        "bos": i == 0,
        "eos": i == len(tokens) - 1,
        "from_end": str(min(len(tokens) - i - 1, 6)),
    }
    for offset in (-2, -1, 1, 2):
        j = i + offset
        key = f"{offset:+d}"
        if 0 <= j < len(tokens):
            neighbour = tokens[j][2].lower()
            feats[f"w{key}"] = neighbour if neighbour in vocab else "<rare>"
            feats[f"shape{key}"] = _shape(tokens[j][2])
            feats[f"metric{key}"] = neighbour in metrics
        else:
            feats[f"w{key}"] = "<pad>"
    feats["after_prep"] = i > 0 and tokens[i - 1][2].lower() in PREPOSITIONS
    return feats


def _bio_labels(text, tokens, spans_by_field):
    """Gold BIO label per token, from the annotated spans."""
    labels = ["O"] * len(tokens)
    for field, spans in spans_by_field.items():
        tag = "LOC" if field == "location" else "TIME"
        cursor = 0
        for span in spans:
            start = text.find(span, cursor)
            if start < 0:                      # annotation and text disagree - skip the span
                continue
            cursor, end = start + 1, start + len(span)
            inside = [k for k, (s, e, _) in enumerate(tokens) if s < end and start < e]
            for position, k in enumerate(inside):
                labels[k] = f"{'B' if position == 0 else 'I'}-{tag}"
    return labels


def _decode(text, tokens, labels):
    """BIO labels -> [(start, end, verbatim slice)] per field."""
    spans = {field: [] for field in FIELDS}
    current_tag, start, end = None, None, None

    def flush():
        if current_tag:
            field = "location" if current_tag == "LOC" else "time"
            trimmed = text[start:end].strip(" ,")
            if trimmed:
                offset = text.find(trimmed, start)
                spans[field].append((offset, offset + len(trimmed), trimmed))

    for (token_start, token_end, _), label in zip(tokens, labels):
        tag = label[2:] if label != "O" else None
        if label.startswith("B-") or (tag and tag != current_tag):
            flush()
            current_tag, start, end = tag, token_start, token_end
        elif tag == current_tag and tag is not None:
            end = token_end                    # I- continuation, extend to here
        else:
            flush()
            current_tag = None
    flush()
    return spans


def _merge_times(text, tagged):
    """Clock/duration regex matches beat the tagger on overlap - the tagger tends to clip
    "4:15 pm" down to "pm", and a deterministic match is never worse than a partial one."""
    matched = []
    for pattern in (CLOCK_RE, DURATION_RE):
        for m in pattern.finditer(text):
            if not any(m.start() < e and s < m.end() for s, e, _ in matched):
                matched.append((m.start(), m.end(), m.group().strip()))
    kept = [span for span in tagged
            if not any(span[0] < e and s < span[1] for s, e, _ in matched)]
    return sorted(matched + kept)


class SpanTagger:
    """Fit on annotated prompts, predict {"location": [...], "time": [...]}."""

    def __init__(self, metric_nouns=(), C=4.0, min_word_freq=15):
        self.metrics = {w for noun in metric_nouns for w in noun.lower().split()}
        self.gazetteer = {field: set() for field in FIELDS}
        self.min_word_freq = min_word_freq
        self.vocab = set()
        self.vectorizer = DictVectorizer()
        self.model = LogisticRegression(max_iter=2000, C=C, n_jobs=-1)

    def fit(self, texts, spans_per_text):
        counts = Counter(w.lower() for text in texts for _, _, w in tokenize(text))
        self.vocab = {w for w, n in counts.items() if n >= self.min_word_freq}

        # The gazetteer is built for callers that want it, but is NOT a feature: during
        # fitting every training place name is in it by construction, so the model would
        # learn "in the gazetteer -> LOCATION" and never learn context. At inference the
        # first unseen village then reads as an ordinary word. Context and character
        # shingles have to carry the decision.
        for spans in spans_per_text:
            for field in FIELDS:
                for span in spans[field]:
                    self.gazetteer[field].update(w.lower() for _, _, w in tokenize(span))

        rows, labels = [], []
        for text, spans in zip(texts, spans_per_text):
            tokens = tokenize(text)
            if not tokens:
                continue
            rows.extend(_token_features(tokens, i, self.gazetteer, self.metrics, self.vocab)
                        for i in range(len(tokens)))
            labels.extend(_bio_labels(text, tokens, spans))
        self.model.fit(self.vectorizer.fit_transform(rows), labels)
        return self

    def predict(self, text):
        tokens = tokenize(text)
        if not tokens:
            return {field: [] for field in FIELDS}
        features = [_token_features(tokens, i, self.gazetteer, self.metrics, self.vocab)
                    for i in range(len(tokens))]
        spans = _decode(text, tokens, self.model.predict(self.vectorizer.transform(features)))
        return {
            "location": [span for _, _, span in spans["location"]],
            "time": [span for _, _, span in _merge_times(text, spans["time"])],
        }
