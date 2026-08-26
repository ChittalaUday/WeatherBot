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

from backend.config import DATA_DIR

ALIAS_FILE = DATA_DIR / "location_aliases.json"

# Ordinary English the tagger sometimes hands over as a place. Solr is fuzzy enough to match
# most of them to *some* village, which is how a question about Pithoragarh once came back
# with a forecast for "skies".
NOT_PLACES = {
    "expect", "i expect", "skies", "plan", "might be", "outdoor plan", "trip", "weather",
    "forecast", "clouds", "rain", "sun", "wind", "temperature", "humidity", "today",
    "tomorrow", "week", "weekend", "morning", "evening", "night", "chance", "idea",
    "question", "time", "bit", "thing", "way", "one", "some", "any", "please", "thanks",
    # span quantifiers: "rainfall for whole day" resolved "whole" against the location index
    "whole", "entire", "full", "complete", "all", "rest", "remainder", "day", "hour",
    "whole day", "entire day", "full day", "all day", "whole week", "entire week",
    "full week", "rest of the day", "the day", "the week", "month", "year",
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
    # "soil moisture in my field" reached the resolver as a place called "my field"; it is a
    # relative location, so the browser supplies the coordinates (Rule 4.1).
    "my field", "my farm", "my plot", "my village", "my land", "our field", "our farm",
    "the field", "the farm", "my place", "my side", "my town", "my city",
    "near me", "nearby", "near by", "here", "my location", "this area", "my area",
    "feild", "my feild", "my vilage", "our village",
    "this village",
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


MONTH_WORDS = {"january", "february", "march", "april", "may", "june", "july", "august",
               "september", "october", "november", "december", "jan", "feb", "mar", "apr",
               "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


def looks_like_a_date(text: str) -> bool:
    """True for '11 jan', '2026', '12/06/2021' - fragments a split date range leaves behind."""
    cleaned = " ".join(text.lower().split()).strip(" ,.?!")
    if not cleaned:
        return False
    if re.fullmatch(r"[\d/\-.:\s]+", cleaned):          # all digits and separators
        return True
    words = set(re.findall(r"[a-z]+", cleaned))
    return bool(words) and words <= MONTH_WORDS


def is_probably_not_a_place(name: str) -> bool:
    """True for ordinary English that only looks like a place to a fuzzy index."""
    cleaned = " ".join(name.lower().split()).strip(" ,.?!")
    if cleaned in NOT_PLACES or cleaned in REFERENCE_WORDS or len(cleaned) < 3:
        return True
    # "11 jan" is what a mis-split date range leaves behind, and the index will happily
    # fuzzy-match it to a village
    if looks_like_a_date(cleaned):
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


def _squeeze(text: str) -> str:
    """Collapse runs of the same letter: "Beeramguda" and "Beramguda" are one name, twice.

    Transliterated names double their vowels and consonants inconsistently - Beeramguda /
    Beramguda, Kompally / Kompaly - so a prefix guard on the raw spelling throws away the
    exact class of misspelling the fuzzy pass exists to catch. It still separates real
    neighbours: "beramguda" and "Belamguda" squeeze to "ber" and "bel".
    """
    return re.sub(r"(.)\1+", r"\1", (text or "").lower())


def _first(doc: dict, key: str) -> str | None:
    """Solr multi-valued fields come back as single-element lists."""
    value = doc.get(key)
    if isinstance(value, list) and value:
        val = value[0]
        return str(val) if val is not None else None
    return str(value) if value is not None else None


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
                if label and lat and lon and _squeeze(label)[:3] == _squeeze(safe)[:3]:
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


async def suggest(solr, raw: str, limit: int = 3) -> list[str]:
    """The nearest names the index *does* hold, for a name it does not.

    The same fuzzy queries `resolve` runs, without the guard that makes a hit safe to answer
    with. That guard is right for answering - Belamguda in Odisha is not Beramguda near
    Hyderabad and must never be silently served as it - but too far to act on is still close
    enough to ask about, and "did you mean Vedurumudi?" is the whole difference between a dead
    end and a resolved turn.

    Retrieval only: what to do with these is the caller's business, and the model that words
    the reply is told to drop them if none looks right.
    """
    head = normalize((raw or "").split(",")[0].split()[0] if raw.strip() else "")
    safe = re.sub(r"[^\w ]", "", head)
    if len(safe) < 4:                      # too short to be a near miss of anything
        return []
    seen, out = set(), []
    for query in (f"(village:{safe}~1 OR sub_district:{safe}~1 OR district:{safe}~1)",
                  f"(village:{safe}* OR district:{safe}*)"):
        for doc in await solr(query, 8):
            for level in LEVELS:
                label = _first(doc, level)
                if not label:
                    continue
                where = ", ".join(p for p in (_first(doc, "district"), _first(doc, "state"))
                                  if p and p != label)
                line = f"{label} ({where})" if where else label
                if line.lower() not in seen:
                    seen.add(line.lower())
                    out.append(line)
                break
        if len(out) >= limit:
            break
    return out[:limit]


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
    # doubled letters are a spelling, not a different place - but a changed letter is
    assert _squeeze("Beeramguda")[:3] == _squeeze("beramguda")[:3]
    assert _squeeze("Kompally")[:3] == _squeeze("kompaly")[:3]
    assert _squeeze("Belamguda")[:3] != _squeeze("beramguda")[:3]
    assert not is_probably_not_a_place("Kakinada")
    assert is_relative("my field") and not is_relative("Guntur")
    print(f"locations demo OK: {len(ALIASES)} aliases, {len(STATE_ALIASES)} state aliases")


if __name__ == "__main__":
    demo()
