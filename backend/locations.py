"""
Location resolution - the layer between "what the user said" and "where that is".

The NLU model reports raw spans and nothing else (Rule 4.1): "KKD" stays "KKD". Turning that
into a place is this module's job, in three steps that fail independently:

    raw span  ->  alias table  ->  Solr lookup  ->  candidates
    "KKD"         "Kakinada"      village/district match     [Kakinada, Andhra Pradesh]

Aliases live in data/location_aliases.json, not in code and never in the model: teaching the
system that BZA means Vijayawada is a one-line edit to a data file, not a retraining run.

A resolved place carries both halves, so the answer can show what was asked and what was used:

    {"raw": "KKD", "normalized": "Kakinada", "type": "village",
     "lat": .., "lon": .., "district": .., "state": .., "matches": [...]}

When several places match equally well ("Angara" exists in Jharkhand and Andhra Pradesh) the
resolver returns them all and lets the caller ask, rather than picking one silently.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

ALIAS_FILE = Path(__file__).resolve().parent.parent / "data" / "location_aliases.json"

# Ordinary English the tagger sometimes hands over as a place. Solr is fuzzy enough to match
# most of them to *some* village, which is how a question about Pithoragarh once came back
# with a forecast for "skies".
NOT_PLACES = {
    "expect", "i expect", "skies", "plan", "might be", "outdoor plan", "trip", "weather",
    "forecast", "clouds", "rain", "sun", "wind", "temperature", "humidity", "today",
    "tomorrow", "week", "weekend", "morning", "evening", "night", "chance", "idea",
    "question", "time", "bit", "thing", "way", "one", "some", "any", "please", "thanks",
}
LEADING_JUNK = {"than", "or", "and", "the", "a", "an", "i", "we", "you", "should", "is", "of"}

# Words that point at the previous turn rather than naming a place. Without this, "and
# there?" gets fuzzy-matched to a village called Thera. backend/state.py owns the meaning;
# here they only need to be kept away from Solr.
REFERENCE_WORDS = {
    "there", "that place", "same place", "that city", "that town", "same location",
    "that village", "the same", "over there", "same spot", "that area", "then",
    "same day", "that day", "same date", "that time", "it", "this",
}

# Relative locations (Rule 4.1) carry no coordinates - the browser has to supply them.
RELATIVE_LOCATIONS = {
    "near me", "nearby", "near by", "here", "my location", "this area", "my area",
    "my field", "feild", "my feild", "my farm", "my village", "my vilage", "our village",
    "my plot", "my place", "this village",
}

LEVELS = ("village", "sub_district", "district", "state")


def _load_aliases() -> tuple[dict, dict]:
    try:
        data = json.loads(ALIAS_FILE.read_text())
        return data.get("aliases", {}), data.get("states", {})
    except (OSError, ValueError):
        return {}, {}


ALIASES, STATE_ALIASES = _load_aliases()
SELF_NAMED_STATES = {
    "telangana", "karnataka", "kerala", "odisha", "maharashtra", "gujarat", "rajasthan",
    "uttarakhand", "jharkhand", "bihar", "punjab", "haryana", "goa", "assam", "tripura",
    "manipur", "meghalaya", "mizoram", "nagaland", "sikkim", "delhi", "ladakh", "puducherry",
    "chandigarh", "andaman", "lakshadweep", "andhra pradesh", "tamil nadu", "west bengal",
    "madhya pradesh", "himachal pradesh", "arunachal pradesh",
}


def is_relative(name: str) -> bool:
    return " ".join(name.lower().split()).strip(" ,.?!") in RELATIVE_LOCATIONS


def is_probably_not_a_place(name: str) -> bool:
    """True for ordinary English that only looks like a place to a fuzzy index."""
    cleaned = " ".join(name.lower().split()).strip(" ,.?!")
    if cleaned in NOT_PLACES or cleaned in REFERENCE_WORDS or len(cleaned) < 3:
        return True
    return cleaned.split()[0] in LEADING_JUNK


def normalize(text: str) -> str:
    """'KKD' -> 'Kakinada'. Unknown text passes through unchanged."""
    key = " ".join(text.lower().split()).strip(" ,.?!")
    return ALIASES.get(key, text.strip(" ,.?!"))


def canonical_state(text: str) -> str | None:
    """'andhrapradesh' / 'AP' -> 'Andhra Pradesh'. None when it is not a state."""
    key = " ".join(text.lower().split()).strip(" ,.?!")
    if key in STATE_ALIASES:
        return STATE_ALIASES[key]
    if key.replace(" ", "") in STATE_ALIASES:
        return STATE_ALIASES[key.replace(" ", "")]
    return key.title() if key in SELF_NAMED_STATES else None


def _first(doc: dict, key: str):
    """Solr multi-valued fields come back as single-element lists."""
    value = doc.get(key)
    return value[0] if isinstance(value, list) and value else value


def _candidate(doc: dict, wanted: str, raw: str) -> dict | None:
    """Most specific level in a Solr doc whose label resembles what was asked for."""
    head = wanted.lower()[:4]
    for level in LEVELS:
        label = _first(doc, level)
        lat, lon = _first(doc, f"{level}_latitude"), _first(doc, f"{level}_longitude")
        if label and lat and lon and label.lower().startswith(head):
            return {
                "raw": raw, "normalized": label, "type": level,
                "name": label,                       # kept for the existing table/chart payloads
                "lat": float(lat), "lon": float(lon),
                "district": _first(doc, "district"), "state": _first(doc, "state"),
            }
    return None


LEVEL_RANK = {"district": 0, "sub_district": 1, "village": 2, "state": 3, "point": 4}


def _is_seat(candidate: dict) -> bool:
    """The town that names its own district - "Guntur, Guntur" is the city everyone means."""
    return bool(candidate["district"]) and \
        candidate["district"].lower().startswith(candidate["normalized"].lower()[:5])


def _rank(candidates: list[dict], wanted: str) -> list[dict]:
    """Exact name, then district seat, then the broader administrative level.

    "Guntur" means the city that names the district, not a same-named hamlet in Maharashtra,
    so this ordering settles most queries without troubling the user.
    """
    target = wanted.lower().strip()
    return sorted(candidates, key=lambda c: (c["normalized"].lower() != target,
                                             not _is_seat(c),
                                             LEVEL_RANK.get(c["type"], 5)))


def _is_ambiguous(candidates: list[dict], wanted: str) -> bool:
    """True only when the top two are equally good answers in different states.

    A district seat is never ambiguous against a hamlet of the same name: asking "did you
    mean Guntur or Guntur?" is noise, while "Angara: Jharkhand or Andhra Pradesh?" is a real
    question the user has to settle.
    """
    if len(candidates) < 2:
        return False
    first, second = candidates[0], candidates[1]
    if _is_seat(first) and not _is_seat(second):
        return False
    target = wanted.lower().strip()
    equally_exact = (first["normalized"].lower() == target) == (second["normalized"].lower() == target)
    return (equally_exact
            and LEVEL_RANK.get(first["type"], 5) == LEVEL_RANK.get(second["type"], 5)
            and (first["state"] or "") != (second["state"] or ""))


def _dedupe(candidates: list[dict]) -> list[dict]:
    seen, out = set(), []
    for candidate in candidates:
        key = (candidate["normalized"].lower(), (candidate["state"] or "").lower())
        if key not in seen:
            seen.add(key)
            out.append(candidate)
    return out


def split_span(raw: str) -> list[str]:
    """"Ziro, Kalimpong" -> two places; "Taloda, Nandurbar" -> one address.

    The tail decides: a state or district name qualifies the head, anything else is a list.
    """
    if "," not in raw:
        return [raw]
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) < 2:
        return [raw]
    if canonical_state(parts[-1]) or len(parts[-1].split()) > 1:
        return [raw]                       # "Kochi, Kerala" / "Amarpatan, Satna" - an address
    return parts if all(len(p.split()) <= 2 for p in parts) else [raw]


async def resolve(solr, raw: str) -> dict | None:
    """Resolve one raw span. `solr` is an async callable(query, rows) -> docs.

    Returns the best candidate with every other candidate under "matches", so an ambiguous
    name can be handed back to the user instead of guessed at.
    """
    if not raw or is_probably_not_a_place(raw):
        return None
    cleaned = normalize(raw)
    if not cleaned:
        return None

    # Split "angara andhrapradesh" / "angara, andhra pradesh" into place + qualifier. Every
    # split point is tried, longest head first, so multi-word village names still work.
    attempts: list[tuple[str, str | None]] = [(cleaned, None)]
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        attempts.insert(0, (normalize(parts[0]), ", ".join(parts[1:])))
    words = cleaned.split()
    for cut in range(len(words) - 1, 0, -1):
        attempts.append((normalize(" ".join(words[:cut])), " ".join(words[cut:])))

    for head, qualifier in attempts:
        if not head:
            continue
        exact = (f'(village:"{head}"^4 OR sub_district:"{head}"^3 OR '
                 f'district:"{head}"^2 OR state:"{head}")')
        query = exact
        if qualifier:
            state = canonical_state(qualifier)
            query = f'{exact} AND ' + (f'state:"{state}"' if state
                                       else f'(district:"{qualifier}" OR state:"{qualifier}")')
        candidates = _rank(_dedupe([c for c in (_candidate(doc, head, raw)
                                                for doc in await solr(query, 8)) if c]), head)
        if candidates:
            best = dict(candidates[0])
            best["matches"] = candidates[:5]
            best["ambiguous"] = _is_ambiguous(candidates, head)
            return best

    # Nothing exact: prefix, then fuzzy for misspellings ("hyderbad" -> Hyderabad). Only for
    # single-word spans - fuzzy-matching "Ziro, Kalimpong" once produced "Ziri, Maharashtra".
    if len(words) > 1 or "," in cleaned:
        return None
    head = normalize(words[0]) if words else cleaned
    safe = re.sub(r"[^\w ]", "", head)
    for query in (f"(village:{safe}* OR district:{safe}* OR state:{safe}*)",
                  f"(village:{safe}~1 OR district:{safe}~1 OR state:{safe}~1)"):
        loose = []
        for doc in await solr(query, 8):
            for level in LEVELS:
                label = _first(doc, level)
                lat, lon = _first(doc, f"{level}_latitude"), _first(doc, f"{level}_longitude")
                if label and lat and lon and label.lower()[:3] == safe.lower()[:3]:
                    loose.append({"raw": raw, "normalized": label, "type": level, "name": label,
                                  "lat": float(lat), "lon": float(lon),
                                  "fuzzy": label.lower() != safe.lower(),
                                  "district": _first(doc, "district"), "state": _first(doc, "state")})
                    break
        if loose:
            candidates = _rank(_dedupe(loose), safe)
            best = dict(candidates[0])
            best["matches"] = candidates[:5]
            best["ambiguous"] = _is_ambiguous(candidates, safe)
            return best
    return None


def demo():
    """Self-check for the parts that need no network."""
    assert normalize("KKD") == "Kakinada"
    assert normalize("bza") == "Vijayawada"
    assert normalize("Kakinada") == "Kakinada"          # unknown text passes through
    assert canonical_state("AP") == "Andhra Pradesh"
    assert canonical_state("andhrapradesh") == "Andhra Pradesh"
    assert canonical_state("Kakinada") is None
    assert is_probably_not_a_place("I expect") and is_probably_not_a_place("skies")
    assert is_probably_not_a_place("there") and is_probably_not_a_place("that place")
    assert not is_probably_not_a_place("Kakinada")
    assert is_relative("my field") and not is_relative("Guntur")
    print(f"locations demo OK: {len(ALIASES)} aliases, {len(STATE_ALIASES)} state aliases")


if __name__ == "__main__":
    demo()
