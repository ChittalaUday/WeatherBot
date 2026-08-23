"""
WeatherSnap API clients - the deterministic layer MODEL_RULES.md keeps out of the model.

Endpoints and credentials come from the Postman environment (production):
  gfsApiUrl        /interpolate  daily forecast, /hrlydata  hourly forecast
  solrUrl          location text -> lat/lng
  infestServerUrl  /api/centroids  lat/lng -> place name
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import datetime

import httpx

from backend.config import (GFS_HISTORICAL_URL, GFS_URL, HTTP_CONNECT_TIMEOUT, HTTP_TIMEOUT,
                            INFEST_URL, SOLR_AUTH, SOLR_URL, ZARR_KEY, ZARR_URL)

TIMEOUT = httpx.Timeout(HTTP_TIMEOUT, connect=HTTP_CONNECT_TIMEOUT)


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
        # census tags, not part of the name anyone says out loud:
        # "Hyderabad (m Corp+og) (part)" -> "Hyderabad"
        name = re.sub(r"\s*\([^)]*\)", "", name).strip() or "your location"
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


# The hourly feed reports sunshine in SECONDS (3600 = a full sunny hour) while the daily feed
# reports HOURS (10.57). Nothing downstream can be expected to know that - summing the hourly
# column as-is produced "19112.7h of sun" in a drying verdict.
HOURLY_SECONDS_FIELDS = ("SunSD",)


async def hourly_forecast(client: httpx.AsyncClient, lat: float, lon: float) -> list[dict]:
    response = await client.get(f"{GFS_URL}/hrlydata", params={"lat": lat, "lon": lon})
    response.raise_for_status()
    rows = response.json().get("Forecast data", [])
    for row in rows:
        for field in HOURLY_SECONDS_FIELDS:
            if isinstance(row.get(field), (int, float)):
                row[field] = round(row[field] / 3600.0, 4)
    return rows


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


# --- historical and archive adapters -----------------------------------------
# Everything below returns rows keyed by the CANONICAL field names the forecast feed uses, so
# render.py, analysis.py, quality.py and advice.py never learn that a second schema exists.
# Only the mapping table below knows.

# Zarr column -> canonical. Tavg is absent from the feed and derived after mapping.
ZARR_FIELD_MAP = {
    "rainfall_mm": "Rainfall", "temp_max_c": "Tmax", "temp_min_c": "Tmin",
    "humidity_pct": "RH", "wind_ms": "Wind_Speed", "day_length_hrs": "DayLength",
}
# The archive also carries climatic normals, which nothing else has. Kept under their own
# prefix so "wetter than usual" becomes answerable without colliding with the readings.
ZARR_NORMAL_MAP = {
    "normal_rainfall_mm": "Normal_Rainfall", "normal_temp_max_c": "Normal_Tmax",
    "normal_temp_min_c": "Normal_Tmin", "normal_humidity_pct": "Normal_RH",
    "normal_wind_ms": "Normal_Wind_Speed", "normal_day_length_hrs": "Normal_DayLength",
}


def _zarr_date(value: str) -> str:
    """'07-08-2026' -> '2026-08-07T00:00:00'. Zarr speaks DD-MM-YYYY; everything else ISO."""
    try:
        d, m, y = str(value).split("-")
        return f"{y}-{m}-{d}T00:00:00"
    except ValueError:
        return str(value)


def map_zarr_row(row: dict) -> dict:
    """One archive row in canonical form, with Tavg derived and normals kept."""
    out = {"Date_time": _zarr_date(row.get("date", ""))}
    for source_key, canonical in {**ZARR_FIELD_MAP, **ZARR_NORMAL_MAP}.items():
        if row.get(source_key) is not None:
            out[canonical] = row[source_key]
    # the feed sends no average; the two extremes are what it has
    if out.get("Tmax") is not None and out.get("Tmin") is not None:
        out["Tavg"] = round((float(out["Tmax"]) + float(out["Tmin"])) / 2, 2)
    return out


def _ddmmyyyy(iso: str) -> str:
    date_part = str(iso)[:10]
    y, m, d = date_part.split("-")
    return f"{d}-{m}-{y}"


async def historical_forecast(client: httpx.AsyncClient, lat: float, lon: float,
                              days: int = 7) -> list[dict]:
    """GFS lookback. Same column names as the forecast feed, so no mapping is needed."""
    response = await client.get(f"{GFS_HISTORICAL_URL}/interpolate/historical",
                                params={"days": max(1, min(days, 60)), "lat": lat, "lon": lon})
    response.raise_for_status()
    return response.json().get("historical_data", [])


async def zarr_point(client: httpx.AsyncClient, lat: float, lon: float,
                     start: str, end: str) -> list[dict]:
    """Archive for one point, any dates. Internal network only."""
    response = await client.get(
        f"{ZARR_URL}/weather",
        params={"lat": lat, "lon": lon, "startDate": _ddmmyyyy(start), "endDate": _ddmmyyyy(end)},
        headers={"x-api-key": ZARR_KEY} if ZARR_KEY else {},
    )
    response.raise_for_status()
    return [map_zarr_row(r) for r in response.json().get("data", [])]


async def zarr_bulk(client: httpx.AsyncClient, places: list[dict],
                    start: str, end: str, daily: bool = True) -> dict:
    """Archive for several points in one call -> {location_id: [canonical rows]}."""
    endpoint = "daily" if daily else "aggregated"
    body = [{"location_id": p.get("name") or str(i), "lat": p["lat"], "lon": p["lon"]}
            for i, p in enumerate(places)]
    response = await client.post(
        f"{ZARR_URL}/bulk-get-weather/{endpoint}",
        params={"startDate": _ddmmyyyy(start), "endDate": _ddmmyyyy(end)},
        json=body, headers={"x-api-key": ZARR_KEY} if ZARR_KEY else {},
    )
    response.raise_for_status()
    payload = response.json()
    results = payload.get("data", payload) if isinstance(payload, dict) else payload
    out: dict = {}
    if isinstance(results, dict):
        for key, rows in results.items():
            out[key] = [map_zarr_row(r) for r in rows or []]
    else:
        for entry in results or []:
            key = entry.get("location_id", "")
            out[key] = [map_zarr_row(r) for r in entry.get("data", []) or []]
    return out


@dataclass
class Fetched:
    """What one plan actually returned, and what it had to settle for."""

    per_place: list          # one list of canonical rows per place, same order as `places`
    source: str
    ok: bool = True
    error: str = ""
    fell_back_from: str = ""
    note: str = ""


async def fetch_for(client: httpx.AsyncClient, plan, places: list[dict]) -> Fetched:
    """Execute a QueryPlan against whichever source it chose, degrading rather than failing.

    The archive lives on an internal address, so "unreachable" is a normal Tuesday for anyone
    running this outside the office. When that happens the recent past is still servable from
    the GFS lookback, and the reply says which one answered - silently returning a different
    window would be worse than the error.
    """
    from backend.pipeline.plan import Source                   # local: avoids a cycle at import

    source = plan.source
    days_back = 0
    if plan.start:
        days_back = max((datetime.now() - datetime.fromisoformat(plan.start)).days, 0)

    async def gfs_lookback(reason_from: str = "") -> Fetched:
        rows = await asyncio.gather(*(historical_forecast(client, p["lat"], p["lon"],
                                                          days=max(days_back, 1)) for p in places))
        return Fetched(list(rows), Source.GFS_HISTORICAL.value, fell_back_from=reason_from,
                       note="served from the 60-day lookback" if reason_from else "")

    try:
        if source is Source.GFS_HOURLY:
            rows = await asyncio.gather(*(hourly_forecast(client, p["lat"], p["lon"]) for p in places))
            return Fetched(list(rows), source.value)
        if source is Source.GFS_DAILY:
            rows = await asyncio.gather(*(daily_forecast(client, p["lat"], p["lon"]) for p in places))
            return Fetched(list(rows), source.value)
        if source is Source.GFS_HISTORICAL:
            return await gfs_lookback()
        if source is Source.ZARR_POINT:
            rows = await asyncio.gather(*(zarr_point(client, p["lat"], p["lon"],
                                                     plan.start, plan.end) for p in places))
            return Fetched(list(rows), source.value)
        if source is Source.ZARR_BULK:
            bulk = await zarr_bulk(client, places, plan.start, plan.end)
            return Fetched([bulk.get(p.get("name") or str(i), [])
                            for i, p in enumerate(places)], source.value)
        if source is Source.POSTGRES_AGG:
            # not implemented: the pre-aggregated admin tables need a DB session, not HTTP.
            # A district centroid is a point, so the point sources answer it meanwhile.
            if days_back:
                return await gfs_lookback(reason_from=source.value)
            rows = await asyncio.gather(*(daily_forecast(client, p["lat"], p["lon"]) for p in places))
            return Fetched(list(rows), Source.GFS_DAILY.value, fell_back_from=source.value,
                           note="area averages not wired yet - using the centroid")
    except (httpx.HTTPError, OSError) as exc:
        # the archive is the only internal host; everything else failing is a real outage
        if source in {Source.ZARR_POINT, Source.ZARR_BULK}:
            if days_back <= 60:
                try:
                    return await gfs_lookback(reason_from=source.value)
                except (httpx.HTTPError, OSError) as inner:
                    exc = inner
            else:
                return Fetched([[] for _ in places], source.value, ok=False,
                               error=f"the archive is unreachable from here and that window is "
                                     f"{days_back} days back, past what the forecast feed keeps")
        return Fetched([[] for _ in places], source.value if source else "", ok=False,
                       error=f"{type(exc).__name__}: {exc}")

    return Fetched([[] for _ in places], "", ok=False, error="no source chosen")
