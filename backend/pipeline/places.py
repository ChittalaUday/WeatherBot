"""
Location resolution - the layer between "what the user said" and "where that is".

The NLU model reports raw spans and nothing else (Rule 4.1): "KKD" stays "KKD". Turning that
into a place is this module's job, in three steps that fail independently:

    raw span  ->  alias table  ->  Solr lookup  ->  candidates
    "KKD"         "Kakinada"      village/district match     [Kakinada, Andhra Pradesh]

Every name-based lookup lives in data/location_aliases.json, not in code and never in the
model - aliases, state spellings, the states that are already canonical, and the relative
phrases only the browser can resolve. Teaching the system that BZA means Vijayawada, or that
"my plot" is a relative place, is a one-line edit to a data file, not a retraining run.

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
from src.tagger import GENERIC_PLACE_WORDS

ALIAS_FILE = DATA_DIR / "location_aliases.json"

LEVELS = ("village", "sub_district", "district", "state")


def _load_lookups() -> tuple[dict, dict, set, set]:
    """Everything the resolver matches by name, out of ALIAS_FILE.

    Four lookups, one file, loaded once:

        aliases             "kkd" -> "Kakinada"
        states              a spelling -> its canonical name ("ap", "telengana", "orissa")
        self_named_states   already canonical, so they map to themselves ("telangana")
        relative_locations  a place only the browser can resolve ("my field", "near me")

    An unreadable file degrades rather than crashes - the resolver still answers on names the
    index holds verbatim. It does mean a relative phrase would be sent to Solr as if it were a
    place name, so the file being present is worth checking in a deployment, not assumed.
    """
    try:
        data = json.loads(ALIAS_FILE.read_text())
    except (OSError, ValueError):
        return {}, {}, set(), set()
    lower = lambda names: {" ".join(str(n).lower().split()) for n in names or ()}
    return (data.get("aliases", {}), data.get("states", {}),
            lower(data.get("self_named_states")), lower(data.get("relative_locations")))


ALIASES, STATE_ALIASES, SELF_NAMED_STATES, RELATIVE_LOCATIONS = _load_lookups()


def is_relative(name: str) -> bool:
    return " ".join(name.lower().split()).strip(" ,.?!") in RELATIVE_LOCATIONS


def relative_in(text: str) -> list[str]:
    """The relative phrase in a sentence, for a tagger that does not tag them (Rule 4.1).

    v4 tags only names from the location index, so "near me" and "my field" reach the resolver
    as no location at all - and a turn with no location asks "which place should I check?",
    which is the one question someone who said "near me" has already answered. Longest match
    first, so "my village" is not read as "my".
    """
    low = " ".join((text or "").lower().split())
    for phrase in sorted(RELATIVE_LOCATIONS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            return [phrase]
    return []


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
    if not raw or is_relative(raw):
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


# Words that are never a place, whatever they look like: the weather vocabulary itself, the
# question words around it, and the span quantifiers. Anything else in a sentence with no
# tagged location is worth one lookup before the turn gives up.
NOT_A_PLACE = GENERIC_PLACE_WORDS | {
    "weather", "rain", "rainfall", "temperature", "temp", "humidity", "wind", "cloud",
    "clouds", "sunshine", "sun", "soil", "moisture", "forecast", "conditions", "climate",
    "report", "summary", "summarize", "summarise", "update", "outlook", "rundown", "snapshot",
    "showers", "precipitation", "dew", "uv", "gusts", "breeze", "overview", "data",
    "what", "whats", "how", "hows", "when", "where", "which", "will", "is", "are", "was",
    "were", "the", "a", "an", "for", "in", "at", "on", "of", "to", "it", "be", "do", "does",
    "give", "tell", "show", "check", "get", "know", "like", "much", "many", "there", "today",
    "tomorrow", "tonight", "yesterday", "now", "morning", "afternoon", "evening", "next",
    "last", "past", "this", "days", "hours", "weeks", "please", "pls", "hi", "hello",
}


async def find_in(solr, text: str, limit: int = 2) -> list[dict]:
    """Places named in a sentence the tagger tagged nothing in.

    The tagger is a model over a 623,000-name vocabulary, so it will always have gaps -
    "Guntur weather" is two tokens and almost no context, and it came back with no location
    at all. The index is the authority on what is a place, so the words that could be one get
    one lookup each before the turn gives up and asks.

    Only reached when the tagger found nothing and the turn needs somewhere, which today ends
    in "which place should I check?" - so the worst case is the dead end we already had.
    """
    seen, found = set(), []
    for word in re.findall(r"[A-Za-z][A-Za-z'\-]{2,}", text or ""):
        low = word.lower()
        if low in NOT_A_PLACE or low in seen:
            continue
        seen.add(low)
        if (place := await resolve(solr, word)):
            found.append(place)
            if len(found) >= limit:
                break
    return found


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
    # every lookup is data now, so an empty or half-written file is a failure and not a quiet
    # degradation - four empty sets would pass every assertion below except this one
    assert ALIASES and STATE_ALIASES and SELF_NAMED_STATES and RELATIVE_LOCATIONS, \
        f"{ALIAS_FILE} did not load: {len(ALIASES)} aliases, {len(STATE_ALIASES)} states, " \
        f"{len(SELF_NAMED_STATES)} self-named, {len(RELATIVE_LOCATIONS)} relative"
    assert normalize("KKD") == "Kakinada"
    assert normalize("bza") == "Vijayawada"
    assert normalize("Kakinada") == "Kakinada"          # unknown text passes through
    assert canonical_state("AP") == "Andhra Pradesh"
    assert canonical_state("andhrapradesh") == "Andhra Pradesh"
    assert canonical_state("Kakinada") is None
    # doubled letters are a spelling, not a different place - but a changed letter is
    assert _squeeze("Beeramguda")[:3] == _squeeze("beramguda")[:3]
    assert _squeeze("Kompally")[:3] == _squeeze("kompaly")[:3]
    assert _squeeze("Belamguda")[:3] != _squeeze("beramguda")[:3]
    assert is_relative("my field") and not is_relative("Guntur")
    # v4 tags no span for these, so the sentence is where they have to be found
    assert relative_in("soil moisture in my field") == ["my field"]
    assert relative_in("will it rain near me") == ["near me"]
    assert relative_in("will it rain in Guntur") == []
    print(f"locations demo OK: {len(ALIASES)} aliases, {len(STATE_ALIASES)} state "
          f"spellings, {len(SELF_NAMED_STATES)} self-named states, "
          f"{len(RELATIVE_LOCATIONS)} relative places - all from {ALIAS_FILE.name}")


if __name__ == "__main__":
    demo()
