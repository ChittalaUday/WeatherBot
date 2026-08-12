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


async def solr_query(client: httpx.AsyncClient, query: str, rows: int = 8) -> list[dict]:
    """Raw Solr search. backend.locations turns these docs into resolved places."""
    response = await client.get(
        f"{SOLR_URL}/solr/location_data/select",
        params={"q": query, "rows": rows, "wt": "json"},
        headers={"Authorization": SOLR_AUTH},
    )
    response.raise_for_status()
    return response.json().get("response", {}).get("docs", [])


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
        return {"raw": "my location", "normalized": name, "type": "village", "name": name,
                "lat": lat, "lon": lon, "matches": [], "ambiguous": False,
                "district": place.get("district"), "state": place.get("state")}
    except (httpx.HTTPError, ValueError, KeyError):
        # a nameless point still has coordinates, which is all the forecast needs
        return {"raw": "my location", "normalized": "your location", "type": "point",
                "name": "your location", "lat": lat, "lon": lon, "matches": [],
                "ambiguous": False, "district": None, "state": None}


async def daily_forecast(client: httpx.AsyncClient, lat: float, lon: float) -> list[dict]:
    response = await client.get(f"{GFS_URL}/interpolate", params={"lat": lat, "lon": lon})
    response.raise_for_status()
    return response.json().get("Forecast data", [])


async def hourly_forecast(client: httpx.AsyncClient, lat: float, lon: float) -> list[dict]:
    response = await client.get(f"{GFS_URL}/hrlydata", params={"lat": lat, "lon": lon})
    response.raise_for_status()
    return response.json().get("Forecast data", [])


def _first(doc: dict, key: str):
    value = doc.get(key)
    return value[0] if isinstance(value, list) and value else value


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
