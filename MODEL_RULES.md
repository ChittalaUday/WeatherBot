# WeatherBot ML Model & Training Rules Book

## 1. Core Architectural Strategy

The WeatherBot NLU system follows a strict decoupling between **Statistical ML (NLU)** and **Deterministic Application Logic**.

```text
                    USER TEXT
                       │
                       ▼
              ┌─────────────────┐
              │   ML MODEL      │
              └────────┬────────┘
                       │
       ┌───────────────┼────────────────┐
       ▼               ▼                ▼
 WEATHER INTENT      ACTION           ENTITIES
       │               │           ┌────┴────┐
       │               │           ▼         ▼
       │               │       LOCATION     TIME
       │               │
       ▼               ▼
  RAIN / TEMP       GET / COMPARE
  WIND / HUMIDITY   ALERT
  CLOUD / SOIL
       │
       └───────────────┬─────────────────┘
                       ▼
              DETERMINISTIC LAYER
                       │
              ┌────────┴────────┐
              ▼                 ▼
            Solr           Time Parser
              │                 │
              ▼                 ▼
           lat/lng          datetime
              │                 │
              └────────┬────────┘
                       ▼
                 Weather Data
```

### ML Model Scope (Strict Boundary)
The ML model is responsible **ONLY** for predicting 4 target variables from user text:
1. `weather_intent` (Weather Parameter / Dimension classification)
2. `action` (User Action classification)
3. `LOCATION` (Raw entity text token extraction)
4. `TIME` (Raw temporal entity text token extraction)

### Deterministic Downstream Scope (DO NOT include in ML Model)
- Resolving location text to lat/lng coordinates (handled downstream via Solr).
- Normalizing relative time text (e.g. "tomorrow", "next week", "6 PM") into ISO datetimes (handled downstream via Time Parser).
- Mapping `weather_intent` to exact database/API field names (e.g. `Rainfall`, `Tmin`, `Tmax`).
- Determining whether single vs multi-location mode is active based on entity count.

---

## 2. Intent Taxonomy & Validation Rules

### Rule 2.1: Allowed `weather_intent` Labels (V1 Taxonomy)

The model must classify queries into exactly **ONE** of the following 14 weather intent classes:

| Intent Label | Target Weather Metric | Example User Queries |
| :--- | :--- | :--- |
| `CURRENT_CONDITIONS` | Overall current weather status | "What's the weather like right now?" |
| `FORECAST` | General forecast summary | "Give me the weather forecast for tomorrow" |
| `TEMPERATURE` | General / average temperature | "What is the temperature in Hyderabad?" |
| `TEMPERATURE_MIN` | Minimum temperature | "Minimum temperature in Vizag tonight?" |
| `TEMPERATURE_MAX` | Maximum temperature | "What's the high temperature tomorrow?" |
| `RAIN` | Precipitation / rainfall | "Will it rain in Rajahmundry today?" |
| `HUMIDITY` | Relative humidity | "How humid is it outside?" |
| `DEW_POINT` | Dew point temperature | "What's the dew point right now?" |
| `WIND_SPEED` | Wind velocity / strength | "How strong is the wind in Vizag?" |
| `WIND_DIRECTION` | Wind vector / cardinal direction | "Which direction is the wind blowing?" |
| `SUNSHINE` | Sunshine duration / solar exposure | "Will it be sunny this afternoon?" |
| `CLOUD_COVER` | Low/total cloud cover percentage | "How cloudy will it be tomorrow?" |
| `SOIL_MOISTURE` | Soil moisture levels (10cm/40cm) | "What's the soil moisture in the field?" |
| `SOIL_TEMPERATURE` | Soil temperature levels (10cm) | "Is the soil warm enough for planting?" |

### Rule 2.2: Forbidden `weather_intent` Labels

- **PROHIBITED:** `CURRENT_WEATHER` label. (Temporal context is controlled strictly by the `TIME` entity. `TEMPERATURE` + `time: "now"` = Current Temp; `TEMPERATURE` + `time: "tomorrow"` = Forecast Temp).
- **PROHIBITED:** Temporal parameter intents (`TODAY`, `TOMORROW`, `FRIDAY`, `MORNING`, `EVENING`, `CURRENT`, `FUTURE`).
- **PROHIBITED:** Spatial parameter intents (`NEAR_ME`, `ONE_LOCATION`, `TWO_LOCATIONS`, `THREE_LOCATIONS`).
- **PROHIBITED:** Raw location names as intents (`Hyderabad`, `Chennai`, `Vizag`, `Delhi`).
- **PROHIBITED:** Derived thresholds (`HEAVY_RAIN`, `HEATWAVE`, `FROST_WARNING`) until explicit deterministic rule-based thresholds are established.

---

## 3. Action Taxonomy & Rules

### Rule 3.1: Allowed `action` Labels (V1 Taxonomy)

The model must classify the user's intended action into one of the following 3 classes:

| Action Label | Definition | Example User Queries |
| :--- | :--- | :--- |
| `GET` | Requesting weather information for location/time | "What is the temperature tomorrow?" |
| `COMPARE` | Comparing weather metrics across locations/times | "Compare rainfall between Hyderabad and Vizag" |
| `ALERT` | Checking or setting warnings/alerts | "Alert me if it rains tomorrow" / "Is there a rain warning?" |

*(Note: `TRACK` and `EXPLAIN` are reserved for V2 and must NOT be trained in V1 dataset).*

---

## 4. Entity Extraction (NER / Slot Filling) Rules

### Rule 4.1: `LOCATION` Entity
- Extract exact raw text tokens identifying locations.
- **Allowed Spans:** City names ("Hyderabad", "Vizag", "Rajahmundry"), Region/State names ("AP", "Telangana"), relative terms ("near me", "here", "my location").
- **Output:** List of extracted location string spans (e.g. `["Hyderabad", "Chennai"]`). Empty list `[]` if no location is mentioned in prompt.
- **Strict Rule:** Do NOT normalize, correct spelling, or convert to lat/lng in the ML model.

### Rule 4.2: `TIME` Entity
- Extract exact raw text tokens identifying temporal expressions.
- **Allowed Spans:** Relative dates ("now", "today", "tonight", "tomorrow"), parts of day ("tomorrow morning", "this evening"), days of week ("Friday", "next week"), specific times ("6 PM", "6 PM to 9 PM").
- **Output:** List of extracted time string spans (e.g. `["tomorrow morning"]`). Empty list `[]` if no time is mentioned in prompt.
- **Strict Rule:** Do NOT resolve to ISO timestamps or date objects in the ML model.

### Rule 4.3: `time_normalized` (Canonical Surface Form)

Raw spans are unusable as query keys - the same instant arrives as "tomorrow", "tommorrow",
"tmrw" or "2moro". Alongside the raw `time` list the model emits `time_normalized`,
positionally aligned one-to-one, folding each span to a single shape:

| Input variants | `time_normalized` |
| :--- | :--- |
| "tommorrow", "tmrw", "2moro", "Tomorrow" | `tomorrow` |
| "rn", "right now", "RIGHT NOW" | `now` |
| "6:45 pm", "at 6:45 PM" | `18:45` (24h `HH:MM`) |
| "from 7 AM to 11 AM" | `07:00-11:00` (`HH:MM-HH:MM`) |
| "in 45 minutes", "next 3 days" | `next 45 minutes`, `next 3 days` |
| "on Friday" | `friday` |

- **Still NOT a datetime.** This canonicalises the *surface form* only; resolving to an
  actual instant remains the deterministic Time Parser's job (Section 1).
- **Never drops information.** An expression with no canonical match ("sowing week") passes
  through lowercased - the Time Parser can still inspect it, and `time` always keeps the
  verbatim original.
- Implemented by `normalize_time()` in `src/tagger.py`.

---

## 5. Dataset & Annotation Rules

1. **Schema Requirement:** Training data must contain 4 target annotations per prompt:
   - `weather_intent` (string from Rule 2.1)
   - `action` (string from Rule 3.1)
   - `location` (list of text spans or BIO slot tags)
   - `time` (list of text spans or BIO slot tags)
2. **Dataset Balance:** All 14 weather intents and 3 actions must have balanced representation across common phrasing patterns.
3. **No Synthetic Overfitting:** Synthetic prompts must cover varying syntactic order (e.g., location first vs time first vs question first).

---

## 6. Training & Evaluation Rules

1. **Model Outputs:** The ML model pipeline must output a structured dictionary/JSON matching:
   ```json
   {
     "weather_intent": "<INTENT_ENUM>",
     "action": "<ACTION_ENUM>",
     "entities": {
       "location": ["<RAW_LOCATION_SPAN>"],
       "time": ["<RAW_TIME_SPAN>"],
       "time_normalized": ["<CANONICAL_TIME>"]
     }
   }
   ```
   `time` and `time_normalized` are the same length and in the same order (Rule 4.3). Query
   on `time_normalized`; keep `time` for auditing what the user actually typed.
2. **Evaluation Metrics:**
   - Classification accuracy, precision, recall, F1 per `weather_intent` and `action`.
   - Entity extraction span F1 / exact-match accuracy for `location` and `time`.
