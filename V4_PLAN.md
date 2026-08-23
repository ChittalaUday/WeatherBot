# WeatherBot v4 — plan of record

> **Status: shipped.** Every layer below is built and v4 is the default served model. This
> file is kept as the written contract for the v4 taxonomy (§2) and the data-source routing
> rules (§4) — nothing else records them. §6 lists what is still open.

The organising idea, which is what this file is really for:

**the NLU extracts, it does not decide.** Everything about which source can serve a request,
how many rows that costs, and what the numbers mean is downstream of the model and
deterministic. The model's whole job is to turn one sentence into slots.

```
USER
  │
  ▼
NLU  ──────────  intent · weather_intent · variables · activity · aggregation
  │              locations · times · entities                    (src/v4)
  ▼
QUERY PLANNER    which source · which resolution · how many rows · is it affordable
  │                                              (backend/pipeline/plan.py)
  ▼
DATA LAYER       Zarr point · Zarr bulk · GFS forecast · GFS historical · Postgres aggregates
  │                                           (backend/pipeline/sources.py)
  ▼
AGGREGATOR       SUM / AVG / MAX / MIN / TREND over the returned rows
  │                                       (backend/pipeline/analysis.py)
  ▼
ADVICE ENGINE    activity + numbers → verdict + reason
  │                                         (backend/pipeline/advice.py)
  │
  ▼
RESPONSE
```

---

## 1. Where we are

| Piece | State | Where |
|---|---|---|
| v4 label contract | **done** | [src/v4/schema.py](src/v4/schema.py) |
| Entity gazetteer | **done**, 134 terms / 6 types | [src/v4/entities.py](src/v4/entities.py) |
| v4 dataset generator | **done**, 23,968 rows | [src/v4/dataset.py](src/v4/dataset.py) |
| Location vocabulary | **done**, 1,166 names + codes + misspellings | [src/fetch_locations.py](src/fetch_locations.py) |
| v4 model (heads) | **done**, 18,518 train rows | [src/v4/model.py](src/v4/model.py) |
| Query planner | **done** | [backend/pipeline/plan.py](backend/pipeline/plan.py) |
| Data-source adapters | **done** except Postgres area aggregates, which fall back to the centroid | [backend/pipeline/sources.py](backend/pipeline/sources.py) |
| Advice engine | **done**, 11 rules | [backend/pipeline/advice.py](backend/pipeline/advice.py) |
| Backend wiring | **done**, v4 is the default | [backend/api/](backend/api/) |

Dataset build today:

```
19,982 rows | 100% unique texts | no template over 6 per split | spans verbatim
splits disjoint | shortcut 43.5% (every activity >=20% keyword-free)
entity leak 20.0% | redundancy 8.3% in 788 paraphrase groups
```

---

## 2. The contract

### 2.1 Intent — 11, one classifier head

The shape of the answer, not its subject. Three families.

| Family | Values | Behaviour |
|---|---|---|
| Answered from data | `INFORMATION` `ADVICE` `COMPARISON` | goes to the planner |
| Answered from a template | `GREETING` `THANKS` `GOODBYE` `SMALL_TALK` `CAPABILITY` | no API call, reply from `REPLIES` |
| Declined | `UNSUPPORTED_METRIC` `OUT_OF_SCOPE` `UNCLEAR` | no API call, reason decides the wording |

`UNSUPPORTED_METRIC` means **no source has the reading** — AQI, pollen, snow depth, tides.
It does *not* mean "too far away" or "too much data": range is a planner verdict, never a
label. `OUT_OF_RANGE` and `RANGE_TOO_LONG` were removed for exactly this reason.

### 2.2 Weather intent — 5, **derived, no head**

Read off the normalised time span by `weather_intent_for()`, so it can never contradict
`times`.

`NONE` · `CURRENT` · `FORECAST` · `TOMORROW` · `HISTORICAL`

`NONE` is for turns that ask for no weather at all. `HISTORICAL` now covers **any past date**,
not just "yesterday" — see §3.

### 2.3 Variables — 10, multi-label head

`GENERAL` `TEMPERATURE` `RAIN` `HUMIDITY` `WIND` `CLOUD` `SUNSHINE` `UV` `SOIL_MOISTURE`
`SOIL_TEMPERATURE`

`UV` has **no field in any source** and is served by a sunshine + cloud proxy. The answer must
say it is a proxy. `TEMPERATURE_MIN/MAX` and `WIND_DIRECTION` do not exist as labels —
"how hot does it get" is `TEMPERATURE` + `aggregation=MAX`.

### 2.4 Activity — 12, one classifier head

A label exists **only when the distinction changes which fields are fetched or which threshold
applies.** Anything finer is an entity or a sub-activity.

| Activity | Fields the advice engine reads |
|---|---|
| `NONE` | — |
| `OUTDOOR_ACTIVITY` | rain, temp, wind |
| `TRAVEL` | rain, wind |
| `RAIN_PROTECTION` | rain |
| `SUN_PROTECTION` | UV proxy, sunshine, cloud |
| `CLOTHING` | temp, rain |
| `DRYING` | rain, humidity, sunshine |
| `SOW` | soil moisture, soil temp, rain |
| `IRRIGATE` | soil moisture, rain, temp |
| `FERTILIZE` | rain window, soil moisture |
| `SPRAY` | wind, rain |
| `HARVEST` | dry spell, humidity |

The five farming activities stay separate because they return **opposite verdicts on the same
day** — 35mm coming in 48h is good for `SOW` and bad for the other four. There is no safe
default, so the distinction cannot live in a gazetteer.

### 2.5 Action — 6, **derived, no head**

`group_for(activity)`: `NONE` `OUTDOOR_ACTIVITY` `TRAVEL` `CLOTHING` `HOUSEHOLD` `AGRICULTURE`

### 2.6 Sub-activity — **open string, derived, no head, no enum**

`sub_activity_for(activity, entities, text)`. Descriptive: it may shift a threshold, but every
activity that carries one has a safe default, so a miss degrades the answer rather than
breaking it. Mostly free from entities already extracted; `SUB_KEYWORDS` covers the handful
with nothing behind them (walk, run, concrete, gardening).

Adding a new sport or vehicle costs one line in `ENTITY_VOCAB` and **never** a regenerated
dataset or a retrained head.

### 2.7 Entities — gazetteer, not a classifier

| Type | Terms | Example |
|---|---|---|
| `sport` | 14 | cricket, badminton |
| `transport` | 20 | bike, tractor, bus |
| `crop` | 45 | paddy, cotton, ragi |
| `material` | 19 | urea, pesticide, npk |
| `clothing` | 19 | raincoat, sarees |
| `event` | 17 | wedding, picnic |

Closed vocabularies, so a lookup is right 100% of the time on terms it holds. Location is
tagged instead, because 600k village names are not a closed list.

### 2.8 Aggregation — 6, one head · Operation — 5 · Time bucket — 16, derived

`RAW` `SUM` `AVG` `MAX` `MIN` `TREND` — the "determine" functions.
`SET` `REPLACE` `MODIFY` `INHERIT` `COMPARE` — multi-turn context.
`TimeBucket` including the new `DATE` for explicit calendar dates.

### 2.9 Heads to actually train

**Four classifier heads + one span tagger.** Everything else is a lookup.

| Component | Type |
|---|---|
| intent | 11-class |
| variables | 10-label multi-label |
| activity | 12-class (forced to `NONE` unless intent is ADVICE) |
| aggregation | 6-class |
| locations / times | span tagger ([src/tagger.py](src/tagger.py)) |
| weather_intent · action · sub_activity · time_bucket · entities | **derived** |

---

## 3. Historical — what changed

The archive serves **any date**, so a dated question is a normal `HISTORICAL` turn. The bot
used to refuse these; it must not.

- `bucket_for` gained `DATE` and matches `15 august 2023`, `2023-08-15`, `12/06/2021`,
  `march 2022`, bare years.
- A named year below `REFERENCE_YEAR` routes to `HISTORICAL`; anything else the planner
  resolves at runtime.
- Long spans (`from 2010 to 2025`, `over the last 5 years`) are ordinary `INFORMATION`
  turns. **Length is never an NLU label.**

### Resolution ladder — `resolutions_for(span_days)`

| Span | Resolutions offered | Rows |
|---|---|---|
| ≤ 10 days | hourly, daily | 24–240 / 1–10 |
| ≤ 31 days | daily | ≤ 31 |
| ≤ 366 days | daily, weekly, monthly | 365 / 52 / 12 |
| ≤ 10 years | monthly, yearly | 120 / 10 |
| > 10 years | yearly | n |

`rows_for(span, resolution)` gives the number a query budget gates on. Ten years of rainfall is
120 monthly rows out of a `GROUP BY`, not a million observations pulled into the app.

---

## 4. Data sources and their capabilities

The planner is the only layer that knows any of this.

| # | Source | Call | Geography | Range | Notes |
|---|---|---|---|---|---|
| 1 | GFS daily forecast | `GET {gfsApiUrl}/interpolate?lat&lon` | point | ~10 days ahead | in use today |
| 2 | GFS hourly forecast | `GET {gfsApiUrl}/hrlydata?lat&lon` | point | ~10 days, hourly | in use today |
| 3 | **GFS historical** | `GET stg-gfs.../interpolate/historical?days=N&lat&lon` | point | N days back | **new** — same field names as 1/2, drop-in |
| 4 | **Zarr point** | `GET {zarr}/weather?lat&lon&startDate&endDate` | point / village | **any dates**, DD-MM-YYYY | **new** — different field names, **plus climatic normals** |
| 5 | **Zarr bulk aggregated** | `POST {zarr}/bulk-get-weather/aggregated?startDate&endDate` | many points | any | **new** — body: `[{location_id, lat, lon}]` |
| 6 | **Zarr bulk daily** | `POST {zarr}/bulk-get-weather/daily?startDate&endDate` | many points | any | **new** |
| 7 | **Postgres aggregates** | SQL | country / state / district / block | any | **new** — `weather.daily_{state,district,block}_weather`, pre-computed |
| 8 | Solr | `GET {solrUrl}/solr/location_data/select` | — | — | location resolution, in use |
| 9 | Infest centroids | `GET {infest}/api/centroids` | — | — | reverse geocode, in use |

`{zarr}` = `http://172.16.16.111:8550`, header `x-api-key`. **Internal network only** — it will
not resolve from outside, so the bot must degrade to source 3 when unreachable.

### Routing rules

```
location is a village / coordinate  → Zarr (4) or GFS (1,2,3)
location is a district / state      → Postgres (7)      cheaper, pre-aggregated
several locations at once           → Zarr bulk (5,6)
future, <= horizon                  → GFS forecast (1,2)
past, few days                      → GFS historical (3)  same field names, no mapping
past, arbitrary dates               → Zarr point (4)
span too large for the resolution   → coarsen, or offer the user a coarser one
```

### Zarr → canonical field mapping

Zarr uses its own names, so an adapter is required. Sources 1–3 already use the canonical set.

| Zarr | Canonical |
|---|---|
| `rainfall_mm` | `Rainfall` |
| `temp_max_c` | `Tmax` |
| `temp_min_c` | `Tmin` |
| `humidity_pct` | `RH` |
| `wind_ms` | `Wind_Speed` |
| `day_length_hrs` | `DayLength` |
| `normal_rainfall_mm`, `normal_temp_max_c`, … | **new** — climatic normals, no canonical equivalent |

The normals are a genuine capability nothing else has: they make "wetter than usual" answerable
rather than just "42mm".

---

## 5. Steps

Ordered so nothing is built on top of something unbuilt. The model is deliberately **not**
first: the dataset is stable, so training is mechanical, while nothing downstream works without
the planner.

### Step 1 — Query planner · `backend/pipeline/plan.py`

Turn slots into an executable plan.

- `plan(slots) -> QueryPlan{source, resolution, start, end, rows_estimate, verdict}`
- Verdicts: `EXECUTE` · `COARSEN` (auto, and say so) · `ASK` (offer a coarser resolution)
- Resolve `times` to absolute `start`/`end`, including `DATE` spans and ranges
  (`from 2010 to 2025`). Extends [backend/pipeline/timewindow.py](backend/pipeline/timewindow.py), which handles
  relative wording only.
- Route by geography and range per §4.

**Check:** `test_queryplan.py` — a table of (question, expected source, resolution, row count).
No network.

### Step 2 — Data adapters · extend `backend/pipeline/sources.py`

- `historical_forecast(days)` → source 3, no field mapping needed
- `zarr_point(lat, lon, start, end)` → source 4 + the mapping table above
- `zarr_bulk(locations, start, end)` → sources 5/6
- `postgres_aggregate(level, id, start, end, resolution)` → source 7
- Every adapter returns **canonical field names**, so `respond.py` and `insights.py` are
  untouched.
- Zarr key from env, server-side only. Fall back to source 3 when the internal host is
  unreachable.

**Check:** one recorded response fixture per adapter; assert the mapper produces canonical keys.

### Step 3 — Advice engine · `backend/pipeline/advice.py`

One rules table keyed by the 12 activities, plus derived metrics (`rain_window`, `dry_spell`,
`find_window`, `sun_fraction`).

- `evaluate(activity, sub_activity, rows) -> {verdict, headline, reasons, evidence}`
- Verdicts `YES` / `NO` / `CAUTION`, each with the numbers that produced it
- Soil thresholds carry a `# ponytail:` note — they need field calibration
- `sub_activity` tunes thresholds only; a missing one falls back to the default

**Check:** `test_advice.py` — fabricated rows asserting each verdict **flips at its threshold**
(24.9mm → YES, 25.1mm → NO).

### Step 4 — v4 model · `src/v4/model.py`

Four heads + tagger on the shared TF-IDF encoder (same shape as
[src/v3/model.py](src/v3/model.py)).

- Force `activity=NONE` unless intent is `ADVICE`; force empty variables for `NO_DATA_NEEDED`
- `evaluate()` reports **per component**, including the derived ones end-to-end
- `--export` writes `models/nlu_v4.joblib` + `models/metrics_v4.json`

**Check:** `test_v4_model.py` with per-component floors, and smoke queries including the
confusion pairs (`spray fertilizer` → SPRAY).

### Step 5 — Backend wiring · `backend/api/chat.py`, `backend/nlu/registry.py`

- Registry serves v4; `Understanding` gains activity / sub_activity / entities
- Conversational and declined intents short-circuit before location resolution — a greeting
  must never reach the geocoder
- Planner verdict `ASK` becomes a clarify message offering the coarser resolution
- Payload gains `advice` and `plan` blocks

**Check:** extend [test_conversations.py](test_conversations.py) with a greeting, a declined
turn, a dated historical turn, and an advice turn.

### Step 6 — Frontend

Verdict card above the table; resolution/source note when the planner coarsened; canned replies
rendered without a table. [frontend/components/messages.tsx](frontend/components/messages.tsx),
[frontend/lib/types.ts](frontend/lib/types.ts).

---

## 6. Open items

1. **Credential.** `x-api-key` for the Zarr server is present in two git-tracked Postman files,
   uncommitted and not in history. Scrub to `{{zarrApiKey}}`, gitignore the environment export,
   and rotate — it has been pasted into a chat.
2. **UV has no source.** Served as a sunshine/cloud proxy; the reply must say so.
3. **Zarr is internal-only.** Decide the degradation path when `172.16.16.111` is unreachable —
   currently proposed as falling back to source 3 with a reduced range.
4. **Soil-moisture thresholds** for `IRRIGATE` / `SOW` are literature defaults (~0.15 dry,
   ~0.30 wet for loam) and need calibration against real fields.
5. **`DRYING` merges laundry and vehicle washing.** Same fields, different humidity term,
   carried by `sub_activity`. Revisit if the verdicts read wrong.
6. **Advice seeds.** 19 of the 42 hand-written rows in `v4_dataset.csv` import as INFORMATION
   because no activity cue matched. Adding an `activity` column fixes them.
