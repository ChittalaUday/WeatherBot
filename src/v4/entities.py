"""
Entity extraction - the specific thing an activity involves.

    python -m src.v4.entities "should i spray pesticide on my cotton field tomorrow"
    {"material": ["pesticide"], "crop": ["cotton"]}

A gazetteer, not a classifier, and deliberately so. Sports, crops, transport and farm inputs
are closed vocabularies of a few dozen entries each; a lookup over a closed list is right
100% of the time on the terms it holds, where a trained tagger would be right about 90% and
be wrong unpredictably. Location is the opposite case - 600k village names, no closed list -
which is why that one *is* tagged (src/tagger.py).

The cost is that an unlisted term is invisible: a crop nobody added is simply not found. That
is a vocabulary problem with a vocabulary fix (add the word to ENTITY_VOCAB), which is a far
better failure mode than a model quietly guessing "cotton" for a word it never saw.

Matching is longest-first with overlap suppression, so "white clothes" wins over "clothes"
and "bengal gram" over "gram". Spans come back verbatim from the original text, cased as the
user typed them, exactly like location and time spans (Rule 4.1).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.v4.schema import ENTITY_VOCAB, EntityType

# term -> type, longest first so the specific phrase is tried before the word inside it
_TERMS = sorted(
    ((term, kind) for kind, terms in ENTITY_VOCAB.items() for term in terms),
    key=lambda pair: -len(pair[0]),
)
_PATTERNS = [(kind, term, re.compile(rf"\b{re.escape(term)}\b", re.I)) for term, kind in _TERMS]


def extract(text: str) -> dict[str, list[str]]:
    """Every known entity in the text, as {type: [verbatim spans]}."""
    found: dict[str, list[str]] = {}
    taken: list[tuple[int, int]] = []
    for kind, _term, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            if any(match.start() < end and start < match.end() for start, end in taken):
                continue                       # already covered by a longer term
            taken.append((match.start(), match.end()))
            span = text[match.start():match.end()]
            bucket = found.setdefault(kind.value, [])
            if span not in bucket:
                bucket.append(span)
    return found


def vocabulary_report() -> dict:
    terms = [term for terms in ENTITY_VOCAB.values() for term in terms]
    return {
        "types": len(ENTITY_VOCAB),
        "terms": len(terms),
        "duplicates across types": [t for t in set(terms) if terms.count(t) > 1],
        "per type": {kind.value: len(terms) for kind, terms in ENTITY_VOCAB.items()},
    }


def demo():
    """Self-check: longest match wins, spans stay verbatim, unknown words stay unknown."""
    assert extract("should i spray pesticide on my cotton field tomorrow") == {
        "material": ["pesticide"], "crop": ["cotton"]}
    assert extract("can we play cricket in Guntur tomorrow") == {"sport": ["cricket"]}
    assert extract("can i wash my white clothes today")["clothing"] == ["white clothes"], \
        "longest match must beat the word inside it"
    assert extract("bengal gram sowing")["crop"] == ["bengal gram"]
    assert extract("Should I take the Motorcycle") == {"transport": ["Motorcycle"]}, \
        "spans come back cased as typed"
    assert extract("will it rain tomorrow") == {}, "no entity is a normal answer"
    assert extract("scattered showers") == {}, "substring must not match inside a word"

    report = vocabulary_report()
    assert not report["duplicates across types"], report["duplicates across types"]
    print(f"entities demo OK: {report['terms']} terms over {report['types']} types "
          f"{report['per type']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        import json
        print(json.dumps(extract(" ".join(sys.argv[1:])), indent=2))
    else:
        demo()
