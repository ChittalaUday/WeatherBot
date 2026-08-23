# WeatherBot ML Model & Training Rules Book — Model 1 (`v3`)

> **Scope.** This is the rules book for **Model 1 only**. Model 2 (`v4`) is the default served
> model and has a different taxonomy — 16 intents, 10 variables, 12 activities, and it derives
> what Model 1 predicts. Model 2's contract is [V4_PLAN.md §2](V4_PLAN.md); the machinery for
> both is [ARCHITECTURE.md](ARCHITECTURE.md).
>
> Kept because it is still load-bearing: the accuracy floors in §10 are what `test_model.py`
> asserts, and Rules 1.1, 5.1–5.4 and 8.5 are version-neutral — the span tagger, the
> normalizer and the human-labels-win policy are shared by both models.

Model 1 replaced two earlier attempts: a 14-class single-variable classifier (v1) and a
coarse-intent slot filler (v2). Both are gone. Model 1 keeps their taxonomy where it was right
and predicts three more things neither of them could.

The architecture id is `v3` — it names the bundle (`models/nlu_v3.joblib`), the module
(`src/v3/`) and the tag on stored turns. "Model 1" is the name everywhere a human looks.

---

## 1. Core Architectural Strategy

The WeatherBot NLU system follows a strict decoupling between **Statistical ML (NLU)** and
**Deterministic Application Logic**.

```text
                              USER TEXT
                                 │
                                 ▼
                        ┌─────────────────┐
                        │     MODEL 1     │
                        └────────┬────────┘
                                 │
        ┌────────────────┬───────┴────────┬────────────────┐
        ▼                ▼                ▼                ▼
     INTENT          VARIABLES        ENTITIES        PRESENTATION
        │            (multi-label)   ┌────┴────┐     ┌─────┴─────┐
        │                 │          ▼         ▼     ▼     ▼     ▼
   CURRENT            RAIN / TEMP  LOCATION   TIME DETAIL CHART INSIGHTS
   FORECAST           WIND / SOIL                          │
   HISTORICAL         HUMIDITY ...                         │
   COMPARE / ALERT         │                               │
        │                  │            AGGREGATION        │
        └──────────────────┴─────────────────┬─────────────┘
                                             ▼
                                    DETERMINISTIC LAYER
                                             │
                          ┌──────────────────┼──────────────────┐
                          ▼                  ▼                  ▼
                        Solr            Time Parser        Field Map
                          │                  │            (FIELD_SETS)
                          ▼                  ▼                  ▼
                       lat/lng            datetime           columns
                          │                  │                  │
                          └──────────────────┼──────────────────┘
                                             ▼
                                       Weather Data
```

### ML Model Scope (Strict Boundary)

Model 1 predicts **eight** targets from user text, in one pass over one shared feature matrix:

| # | Target | Kind | Section |
| :-- | :--- | :--- | :--- |
| 1 | `intent` | single-label, 6 classes | 2 |
| 2 | `variables` | **multi-label**, 13 classes | 3 |
| 3 | `aggregation` | single-label, 6 classes | 4 |
| 4 | `locations` | raw span extraction | 5 |
| 5 | `times` (+ `times_normalized`) | raw span extraction | 5 |
| 6 | `detail` | single-label, 3 classes | 6 |
| 7 | `chart` | single-label, 5 classes | 6 |
| 8 | `insights` | **multi-label**, 9 classes | 6 |

### Deterministic Downstream Scope (DO NOT include in the model)

- Resolving location text to lat/lng coordinates (handled downstream via Solr).
- Normalizing relative time text ("tomorrow", "6 PM") into ISO datetimes (Time Parser).
- Mapping a `(variable, detail)` pair to exact API field names — `FIELD_SETS` in
  `src/v3/schema.py`. The model chooses the *detail level*; the table chooses the *columns*.
- Fetching, joining or aggregating the actual weather rows.
- Rendering the chart the model asked for.

### Rule 1.1: The model decides, it never asks

Model 1 always commits to a reading. There is no clarification path: every turn returns an
intent, a variable set, a detail level and a chart, and reports what it assumed in `assumed`.
`backend/nlu/registry.py::NEVER_ASKS` encodes this, and `test_model.py::check_always_decides`
enforces it — a turn that comes back undecided is a test failure, not a prompt to the user.

---

## 2. Intent Taxonomy & Validation Rules

### Rule 2.1: Allowed `intent` Labels

Deliberately small, because the weather variable is a slot now (Section 3), not part of the
intent. The model must classify into exactly **ONE** of 6 classes:

| Intent Label | Meaning | Example User Queries |
| :--- | :--- | :--- |
| `CURRENT` | what it is doing right now | "What's the weather like right now?" |
| `FORECAST` | what it will do | "Will it rain in Rajahmundry tomorrow?" |
| `HISTORICAL` | what it did | "How much rain did we get last week?" |
| `COMPARE` | two or more places, or two or more times | "Compare rainfall in Guntur and Vizag" |
| `ALERT` | tell me when / warn me | "Alert me if wind crosses 40 kmph tonight" |
| `UNKNOWN` | nothing weather-shaped was said | "who won the match" |

`COMPARE` and `ALERT` carry the action; everything else maps to `GET` downstream
(`backend/nlu/registry.py::_understand`).

### Rule 2.2: Forbidden `intent` Labels

- **PROHIBITED:** weather variables as intents (`RAIN`, `TEMPERATURE`, `SOIL_MOISTURE`).
  That was the v1 mistake — it made "rain and temperature" unrepresentable. Variables are a
  multi-label slot.
- **PROHIBITED:** `CURRENT_WEATHER` as a variable-bearing label. Temporal context comes from
  the `TIME` entity: `TEMPERATURE` + `now` is current temp, `TEMPERATURE` + `tomorrow` is
  forecast temp.
- **PROHIBITED:** temporal parameter intents (`TODAY`, `TOMORROW`, `FRIDAY`, `MORNING`).
- **PROHIBITED:** spatial parameter intents (`NEAR_ME`, `ONE_LOCATION`, `TWO_LOCATIONS`).
- **PROHIBITED:** raw location names as intents (`Hyderabad`, `Chennai`, `Vizag`).
- **PROHIBITED:** derived thresholds (`HEAVY_RAIN`, `HEATWAVE`, `FROST_WARNING`) until
  explicit deterministic thresholds are established. `ALERT` + `THRESHOLD` insight covers it.

---

## 3. Variable Taxonomy & Rules

### Rule 3.1: Allowed `variables` Labels (multi-label)

A query may name several measurements. The head is one-vs-rest logistic regression over 13
classes with a calibrated threshold, so `variables` is a **set**, never a single value:

| Variable | Target Weather Metric |
| :--- | :--- |
| `GENERAL` | "weather" with no specific measure named |
| `TEMPERATURE` | general / average temperature |
| `TEMPERATURE_MIN` | minimum temperature |
| `TEMPERATURE_MAX` | maximum temperature |
| `RAIN` | precipitation / rainfall |
| `HUMIDITY` | relative humidity |
| `DEW_POINT` | dew point temperature |
| `WIND_SPEED` | wind velocity / strength |
| `WIND_DIRECTION` | wind vector / cardinal direction |
| `SUNSHINE` | sunshine duration / solar exposure |
| `CLOUD_COVER` | low / total cloud cover percentage |
| `SOIL_MOISTURE` | soil moisture (10cm / 40cm) |
| `SOIL_TEMPERATURE` | soil temperature (10cm) |

### Rule 3.2: Never empty

If no label clears the threshold the head falls back to its highest-scoring class
(`V3Model._multi`). An empty variable set is not a valid prediction — `GENERAL` is the answer
when nothing specific was named, not `[]`.

### Rule 3.3: Ordered by confidence

`variables` comes back sorted by descending probability, and `fields_for()` walks it in that
order. "rain and temperature" therefore puts `Rainfall` before `Tavg` in the table.

---

## 4. `aggregation` — How to Reduce the Rows

A target orthogonal to intent and variable: the variable picks the **field**, the aggregation
picks what to do with the **rows** the time expression selected.

| Label | Meaning | Example |
| :--- | :--- | :--- |
| `RAW` | show the values as they come | "rainfall in Guntur tomorrow" |
| `SUM` | total across the range | "total rainfall next 7 days" |
| `AVG` | mean across the range | "average humidity this week" |
| `MAX` | largest value, and when | "peak wind speed tomorrow" |
| `MIN` | smallest value, and when | "lowest soil temperature this week" |
| `TREND` | direction and turning point | "when will the temperature start dropping?" |

- `TEMPERATURE_MIN` + `RAW` is the `Tmin` field for one day. `TEMPERATURE` + `MIN` is the
  coldest value across a range. Different questions; must not be conflated.
- **Lexical support required.** A reduction is always spoken out loud ("total", "average",
  "peak"). `backend/pipeline/analysis.py::confirm_aggregation` drops a non-`RAW` prediction when no
  such word appears — a short prompt like "weather in KKD" is not a request for a maximum.
- `TREND` forces hourly granularity: a turning point cannot be read off one daily row.

---

## 5. Entity Extraction (NER / Slot Filling) Rules

### Rule 5.1: `locations` Entity
- Extract exact raw text tokens identifying locations.
- **Allowed Spans:** city names ("Hyderabad", "Vizag"), region/state names ("AP",
  "Telangana"), addresses ("Angara, East Godavari"), relative terms ("near me", "here",
  "my location", "my field").
- **Output:** list of raw location strings, e.g. `["Hyderabad", "Chennai"]`. Empty list `[]`
  if no location is mentioned.
- **Strict Rule:** do NOT normalize, spell-correct, or convert to lat/lng in the model.

### Rule 5.2: `times` Entity
- Extract exact raw text tokens identifying temporal expressions.
- **Allowed Spans:** relative dates ("now", "today", "tonight", "tomorrow"), parts of day
  ("tomorrow morning"), days of week ("Friday", "next week"), clock times ("6 PM",
  "6 PM to 9 PM").
- **Output:** list of raw time strings. Empty list `[]` if no time is mentioned.
- **Strict Rule:** do NOT resolve to ISO timestamps or date objects in the model.

### Rule 5.3: Spans must be verbatim
Every returned span must appear character-for-character in the prompt. Enforced by
`test_model.py::check_spans_verbatim` across the whole English eval set.

### Rule 5.4: `times_normalized` (Canonical Surface Form)

Raw spans are unusable as query keys — the same instant arrives as "tomorrow", "tommorrow",
"tmrw" or "2moro". Alongside `times` the model emits `times_normalized`, positionally aligned
one-to-one, folding each span to a single shape:

| Input variants | `times_normalized` |
| :--- | :--- |
| "tommorrow", "tmrw", "2moro", "Tomorrow" | `tomorrow` |
| "rn", "right now", "RIGHT NOW" | `now` |
| "6:45 pm", "at 6:45 PM" | `18:45` (24h `HH:MM`) |
| "from 7 AM to 11 AM" | `07:00-11:00` (`HH:MM-HH:MM`) |
| "in 45 minutes", "next 3 days" | `next 45 minutes`, `next 3 days` |
| "on Friday" | `friday` |

- **Still NOT a datetime.** This canonicalises the *surface form* only; resolving to an actual
  instant remains the deterministic Time Parser's job (Section 1).
- **Never drops information.** An expression with no canonical match ("sowing week") passes
  through lowercased — the Time Parser can still inspect it, and `times` always keeps the
  verbatim original.
- Implemented by `normalize_time()` in `src/tagger.py`.

---

## 6. Presentation Rules — What Model 1 Added

v1 and v2 said what was asked and let Python decide how to answer: a lookup table picked the
columns, a branch on series count picked the chart, and every applicable insight was emitted
every time. Those choices are in the wording, and the wording is what a model can read.

### Rule 6.1: `detail` — How Much To Show

| Label | Meaning | Cue |
| :--- | :--- | :--- |
| `MINIMAL` | a single number, no table | "just tell me", "quickly", "only the total" |
| `NORMAL` | the headline field per variable | the default when nothing is said |
| `FULL` | every related field | "in detail", "full details", "everything about" |

`detail` selects a column *set* per variable through `FIELD_SETS` — "temperature in detail"
means `Tmin`, `Tmax` and `Tavg`. The mapping stays deterministic; only the level is predicted.

### Rule 6.2: `chart` — Which Picture Helps

| Label | When |
| :--- | :--- |
| `NONE` | one value, or a question a table answers better |
| `STAT` | a single big number worth showing large |
| `LINE` | one series over time |
| `MULTI_LINE` | several variables over time, one place |
| `GROUPED_BAR` | places or periods side by side |

Decided from the question, not from row counts. `MINIMAL` detail implies `NONE` — asking for
one number is asking for no chart.

### Rule 6.3: `insights` — What Is Worth Saying (multi-label)

| Label | Observation |
| :--- | :--- |
| `TOTAL` | summed over the range |
| `AVERAGE` | mean over the range |
| `RANGE` | min–max spread |
| `PEAK` | highest value and when |
| `LOW` | lowest value and when |
| `TREND` | rising, falling, turning point |
| `THRESHOLD` | crossings worth a warning |
| `COMPARISON` | which place or period wins |
| `DRY_SPELL` | consecutive days under a rain threshold |

Two to four per turn is normal; a comparison over a week wants several.
`backend/pipeline/analysis.py` computes only what the model selected — without a selection it emits
everything applicable, which is the old pre-Model-1 behaviour.

### Rule 6.4: Presentation is advisory, never load-bearing
A wrong chart is a worse answer, not a broken one. The deterministic layer must produce a
correct table whatever the presentation heads predict.

---

## 7. Dataset & Annotation Rules

1. **Schema Requirement.** A training row carries every target in Section 1:
   `text, intent, variables, aggregation, locations, times, detail, chart, insights`, plus
   `chat_id, turn, operation, ctx_locations, ctx_times, split, source, lang`.
   `variables` and `insights` are pipe-delimited; span lists are JSON.
2. **Dataset Balance.** All intents, variables and actions must have balanced representation
   across common phrasing patterns. The generated splits are checked cell-by-cell
   (`test_dataset.py::check_generated`, min ≥ 0.8 × max).
3. **No Synthetic Overfitting.** Synthetic prompts must vary syntactic order (location first
   vs time first vs question first), and must carry misspellings, chat fillers ("pls", "asap")
   and dropped question marks. A model that only ever sees clean typing fails on real users.
4. **Detail phrasings are generated fresh.** No v1 or v2 template ever said "in detail" or
   "just tell me", so `src/v3/dataset.py::detail_prompts` writes those rows specifically.
5. **The evaluation set is hand-written.** `data/eval_manual.csv` covers typos, code-mixing
   and edge cases no template produces. **Never regenerate it from a script** — the moment
   training material reaches it, it stops measuring generalization.
6. **Held-out entity vocabulary.** A fifth of the sampled place names (`split == eval` in
   `data/locations.csv`) is never generated into train or test, so the eval set measures
   generalization to unseen spans, not memorisation.

---

## 8. Training & Evaluation Rules

### Rule 8.1: Model Output Contract

```json
{
  "text": "<RAW USER TEXT>",
  "intent": "<INTENT_ENUM>",
  "aggregation": "<AGGREGATION_ENUM>",
  "slots": {
    "variables": ["<VARIABLE_ENUM>"],
    "locations": ["<RAW_LOCATION_SPAN>"],
    "times": ["<RAW_TIME_SPAN>"],
    "times_normalized": ["<CANONICAL_TIME>"]
  },
  "presentation": {
    "detail": "<DETAIL_ENUM>",
    "chart": "<CHART_ENUM>",
    "insights": ["<INSIGHT_ENUM>"]
  },
  "confidence": {"intent": 0.0},
  "scores": {"<INTENT_ENUM>": 0.0},
  "assumed": ["<WHAT IT COMMITTED TO INSTEAD OF ASKING>"],
  "model_version": "v3"
}
```

`times` and `times_normalized` are the same length and in the same order (Rule 5.4). Query on
`times_normalized`; keep `times` for auditing what the user actually typed.

### Rule 8.2: Evaluation Metrics

- Per-target accuracy for `intent`, `aggregation`, `detail` and `chart`.
- Exact-set match for `variables`; micro-F1 for `insights`.
- Exact-match (case-insensitive, order-insensitive) for `locations` and `times`.
- `everything`: all eight targets correct on the same turn. This is the honest number.
- Multi-turn context is **not** measurable per utterance — "and there?" carries no place, no
  time and no measurement. It is scored separately by replaying whole conversations through
  the real pipeline (`test_conversations.py`).

### Rule 8.3: Floors, Not Targets

`test_model.py::FLOORS` sits below the measured numbers so it catches regressions without
going red on run-to-run noise. Raise the floors when the model genuinely improves.

Measured on the 219 hand-written English eval utterances:

| Target | Measured | Floor |
| :--- | ---: | ---: |
| `intent` | 0.959 | 0.92 |
| `aggregation` | 0.954 | 0.92 |
| `detail` | 1.000 | 0.97 |
| `chart` | 0.840 | 0.80 |
| `variables` | 0.922 | 0.88 |
| `locations` | 0.945 | 0.90 |
| `times` | 0.959 | 0.92 |
| `insights` (F1) | 0.907 | 0.86 |
| `everything` | 0.680 | 0.62 |

### Rule 8.4: Code-mixed rows are a diagnostic, never a gate
English is the target language. The 16 code-mixed eval rows are reported and never asserted.

### Rule 8.5: Human corrections outrank the model
A thumbs-down refined into a correction becomes a training row whose labels win over the
model's (`backend/store.py::training_rows`). One opinion per turn — a later correction updates
the same record rather than appending a second.
