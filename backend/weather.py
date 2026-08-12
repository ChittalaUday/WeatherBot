"""
WeatherSnap API clients - the deterministic layer MODEL_RULES.md keeps out of the model.

Endpoints and credentials come from the Postman environment (production):
  gfsApiUrl        /interpolate  daily forecast, /hrlydata  hourly forecast
  solrUrl          location text -> lat/lng
  infestServerUrl  /api/centroids  lat/lng -> place name
"""

from __future__ import annotations

import os

import httpx

GFS_URL = os.getenv("GFS_API_URL", "https://gfsapi.niruthiapptesting.com")
SOLR_URL = os.getenv("SOLR_URL", "https://solr.apps.niruthi.com")
INFEST_URL = os.getenv("INFEST_SERVER_URL", "https://infestserver.apps.niruthi.com")
SOLR_AUTH = os.getenv("SOLR_AUTH_HEADER", "Basic YXBpOk5pcnV0aGlAMjRVc2Vy")

TIMEOUT = httpx.Timeout(20.0, connect=10.0)

# Relative locations (Rule 4.1) carry no coordinates - the browser has to supply them.
RELATIVE_LOCATIONS = {
    "near me", "nearby", "near by", "here", "my location", "this area", "my area",
    "my field", "feild", "my feild", "my farm", "my village", "my vilage", "our village",
    "my plot", "my place", "this village",
}


# Everyday names the Solr index does not carry.
NICKNAMES = {
    "vizag": "Visakhapatnam", "bombay": "Mumbai", "bangalore": "Bengaluru",
    "madras": "Chennai", "calcutta": "Kolkata", "poona": "Pune", "mysore": "Mysuru",
    "trivandrum": "Thiruvananthapuram", "hyd": "Hyderabad", "vijaywada": "Vijayawada",
    "pondy": "Puducherry", "gurgaon": "Gurugram", "cochin": "Kochi",
}


def is_relative(name: str) -> bool:
    return name.strip().lower() in RELATIVE_LOCATIONS


def _first(doc: dict, key: str):
    """Solr multi-valued fields come back as single-element lists."""
    value = doc.get(key)
    return value[0] if isinstance(value, list) and value else value


# Squashed / abbreviated state names -> the spelling the Solr index actually uses.
# "angara andhrapradesh" must land in Andhra Pradesh, not in the first Angara Solr returns.
STATE_ALIASES = {
    "andhrapradesh": "Andhra Pradesh", "ap": "Andhra Pradesh", "andra pradesh": "Andhra Pradesh",
    "arunachalpradesh": "Arunachal Pradesh", "tamilnadu": "Tamil Nadu", "tn": "Tamil Nadu",
    "madhyapradesh": "Madhya Pradesh", "mp": "Madhya Pradesh",
    "uttarpradesh": "Uttarpradesh", "uttar pradesh": "Uttarpradesh", "up": "Uttarpradesh",
    "westbengal": "West Bengal", "wb": "West Bengal",
    "himachalpradesh": "Himachal Pradesh", "hp": "Himachal Pradesh",
    "jammuandkashmir": "Jammu And Kashmir", "jk": "Jammu And Kashmir",
    "chhattisgarh": "Chhatisgarh", "chattisgarh": "Chhatisgarh",
    "telangana": "Telangana", "ts": "Telangana", "tg": "Telangana", "orissa": "Odisha",
    "tamil nadu": "Tamil Nadu", "madhya pradesh": "Madhya Pradesh",
    "west bengal": "West Bengal", "himachal pradesh": "Himachal Pradesh",
}
SELF_NAMED_STATES = {
    "telangana", "karnataka", "kerala", "odisha", "maharashtra", "gujarat", "rajasthan",
    "uttarakhand", "jharkhand", "bihar", "punjab", "haryana", "goa", "assam", "tripura",
    "manipur", "meghalaya", "mizoram", "nagaland", "sikkim", "delhi", "ladakh", "puducherry",
    "chandigarh", "andaman", "lakshadweep",
}


def canonical_state(text: str) -> str | None:
    """'andhrapradesh' / 'AP' / 'andhra pradesh' -> 'Andhra Pradesh'. None if not a state."""
    key = " ".join(text.lower().split()).strip(" ,.")
    if key in STATE_ALIASES:
        return STATE_ALIASES[key]
    if key.replace(" ", "") in STATE_ALIASES:
        return STATE_ALIASES[key.replace(" ", "")]
    if key in SELF_NAMED_STATES:
        return key.title()
    return None


def _clean_name(name: str) -> str:
    cleaned = name.replace('"', "").replace("?", "").strip(" ,.")
    return NICKNAMES.get(cleaned.lower(), cleaned)


def _pick_level(doc: dict, wanted: str, query: str) -> dict | None:
    """Most specific level in a Solr doc whose label resembles what the user typed."""
    head = wanted.lower()[:4]
    for level in ("village", "sub_district", "district", "state"):
        label = _first(doc, level)
        lat, lon = _first(doc, f"{level}_latitude"), _first(doc, f"{level}_longitude")
        if label and lat and lon and label.lower().startswith(head):
            return {"query": query, "name": label, "level": level,
                    "lat": float(lat), "lon": float(lon),
                    "district": _first(doc, "district"), "state": _first(doc, "state")}
    return None


async def _solr(client: httpx.AsyncClient, query: str, rows: int = 8) -> list[dict]:
    response = await client.get(
        f"{SOLR_URL}/solr/location_data/select",
        params={"q": query, "rows": rows, "wt": "json"},
        headers={"Authorization": SOLR_AUTH},
    )
    response.raise_for_status()
    return response.json().get("response", {}).get("docs", [])


async def resolve_location(client: httpx.AsyncClient, name: str) -> dict | None:
    """Location text -> {name, lat, lon, ...}.

    Handles three things the plain prefix query got wrong:
      - qualifiers: "angara andhrapradesh" must land in Andhra Pradesh, not Ranchi
      - misspellings: "hyderbad" must still find Hyderabad (Solr fuzzy ~1)
      - nicknames: "vizag" -> Visakhapatnam
    """
    cleaned = _clean_name(name)
    if not cleaned:
        return None

    # Split "angara, andhra pradesh" or "angara andhrapradesh" into place + qualifier. Every
    # split point is tried, longest head first, so multi-word village names still work.
    candidates: list[tuple[str, str | None]] = [(cleaned, None)]
    if "," in cleaned:
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        candidates.insert(0, (parts[0], ", ".join(parts[1:])))
    words = cleaned.split()
    for cut in range(len(words) - 1, 0, -1):
        candidates.append((" ".join(words[:cut]), " ".join(words[cut:])))

    for head, qualifier in candidates:
        head = _clean_name(head)
        if not head:
            continue
        exact = (f'(village:"{head}"^4 OR sub_district:"{head}"^3 OR '
                 f'district:"{head}"^2 OR state:"{head}")')
        query = exact
        if qualifier:
            state = canonical_state(qualifier)
            narrow = f'state:"{state}"' if state else f'(district:"{qualifier}" OR state:"{qualifier}")'
            query = f"{exact} AND {narrow}"
        for doc in await _solr(client, query):
            if (found := _pick_level(doc, head, name)):
                return found

    # nothing exact: prefix, then fuzzy for misspellings ("hyderbad" -> Hyderabad)
    for query in (f"(village:{head}* OR district:{head}* OR state:{head}*)",
                  f"(village:{head}~1 OR district:{head}~1 OR state:{head}~1)"):
        for doc in await _solr(client, query):
            for level in ("village", "sub_district", "district", "state"):
                label = _first(doc, level)
                lat, lon = _first(doc, f"{level}_latitude"), _first(doc, f"{level}_longitude")
                if label and lat and lon and label.lower()[:3] == head.lower()[:3]:
                    return {"query": name, "name": label, "level": level,
                            "lat": float(lat), "lon": float(lon),
                            "district": _first(doc, "district"), "state": _first(doc, "state")}
    return None


async def reverse_geocode(client: httpx.AsyncClient, lat: float, lon: float) -> dict:
    """Browser coordinates -> place name, so the reply can say where it looked."""
    try:
        response = await client.get(f"{INFEST_URL}/api/centroids",
                                    params={"latitude": lat, "longitude": lon})
        response.raise_for_status()
        place = response.json().get("location", {}).get("location", {})
        if isinstance(place, dict) and place.get("type") == "Point":
            place = response.json().get("location", {})
        name = place.get("village") or place.get("sub_dist") or place.get("district") or "your location"
        return {"query": "your location", "name": name, "level": "village",
                "lat": lat, "lon": lon,
                "district": place.get("district"), "state": place.get("state")}
    except (httpx.HTTPError, ValueError, KeyError):
        # a nameless point still has coordinates, which is all the forecast needs
        return {"query": "your location", "name": "your location", "level": "point",
                "lat": lat, "lon": lon, "district": None, "state": None}


async def daily_forecast(client: httpx.AsyncClient, lat: float, lon: float) -> list[dict]:
    response = await client.get(f"{GFS_URL}/interpolate", params={"lat": lat, "lon": lon})
    response.raise_for_status()
    return response.json().get("Forecast data", [])


async def hourly_forecast(client: httpx.AsyncClient, lat: float, lon: float) -> list[dict]:
    response = await client.get(f"{GFS_URL}/hrlydata", params={"lat": lat, "lon": lon})
    response.raise_for_status()
    return response.json().get("Forecast data", [])


async def suggest_locations(client: httpx.AsyncClient, term: str, rows: int = 8) -> list[dict]:
    """Autocomplete for the frontend's location picker."""
    if len(term.strip()) < 2:
        return []
    query = f"(village:{term}* OR district:{term}* OR state:{term}*)"
    response = await client.get(
        f"{SOLR_URL}/solr/location_data/select",
        params={"q": query, "rows": rows, "wt": "json"},
        headers={"Authorization": SOLR_AUTH},
    )
    response.raise_for_status()
    seen, out = set(), []
    for doc in response.json().get("response", {}).get("docs", []):
        village, district, state = (_first(doc, k) for k in ("village", "district", "state"))
        lat, lon = _first(doc, "village_latitude"), _first(doc, "village_longitude")
        label = ", ".join(part for part in (village, district, state) if part)
        if label and label not in seen and lat and lon:
            seen.add(label)
            out.append({"label": label, "name": village, "lat": float(lat), "lon": float(lon)})
    return out


def client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=TIMEOUT, headers={"User-Agent": "WeatherBot/1.0"})
