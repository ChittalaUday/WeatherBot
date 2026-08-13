# Model 1 — Full Architecture

How Model 1 works, what every file in this repo does, and why it is the whole model rather
than one of three. The taxonomy and the training rules live in
[MODEL_RULES.md](MODEL_RULES.md); this file is the machinery.

---

## 1. Why Model 1 Is Complete

Three models were built. Two are deleted. This is what each one could and could not do.

| | v1 — deleted | v2 — deleted | **Model 1** |
| :--- | :--- | :--- | :--- |
| Intent | 14 classes, variable baked in | 6 coarse classes | **6 coarse classes** |
| Variables | one, as part of the intent | multi-label slot | **multi-label slot** |
| Multiple places / times | no | yes | **yes** |
| Aggregation | yes | yes | **yes** |
| Spans + canonical time | yes | yes | **yes** |
| How much to show | Python lookup table | Python lookup table | **predicted (`detail`)** |
| Which chart | branch on row count | branch on row count | **predicted (`chart`)** |
| Which insights | all applicable, always | all applicable, always | **predicted (`insights`)** |
| Below confidence | asks a clarifying question | asks a clarifying question | **commits, reports assumption** |

v1's flaw was structural: folding the variable into the intent made "rain **and** temperature"
unrepresentable at any accuracy. v2 fixed the shape but still answered every question the same
way — the same columns, the same chart rule, every insight every time — because presentation
lived in Python. Model 1's claim is that presentation is *in the wording*, and wording is
exactly what a classifier can read:

```text
"temp in Guntur tomorrow"              -> detail NORMAL   chart NONE         insights RANGE
"temperature in Guntur in detail"      -> detail FULL     chart LINE         insights RANGE, PEAK, LOW
"compare rain in Guntur and Vizag"     -> detail NORMAL   chart GROUPED_BAR  insights COMPARISON, TOTAL
"just tell me the temp in Guntur"      -> detail MINIMAL  chart NONE         insights RANGE
```

That is the whole argument for calling it complete: it predicts everything needed to answer,
not just everything needed to look the answer up.

**What survives of v1 and v2 is library code, not models.** Model 1 imports the vectorizer and
`clean_text` from `src/nlu.py`, the span tagger from `src/tagger.py`, the slot enums from
`src/v2/schema.py`, and the conversation generator from `src/v2/dataset.py`. Those modules stay
because Model 1 is built out of them. Only their trained bundles and intermediate CSVs are gone.

---

## 2. The Model

One shared feature matrix, six trained heads, one span tagger. `src/v3/model.py`.

```text
                          raw text
                             │
                     clean_text()  (src/nlu.py)
                             │
                  build_vectorizer()  — word + char n-grams
                             │
              ┌──────────────┴───────────────┐
              │                              │
       feature matrix (shared)          raw text
              │                              │
   ┌──────┬───┴───┬────────┬───────┬─────┐   │
   ▼      ▼       ▼        ▼       ▼     ▼   ▼
intent  vars  aggregation detail chart insights   SpanTagger (BIO)
   │      │       │        │       │     │            │
   │      │       │        │       │     │      ┌─────┴─────┐
   │      │       │        │       │     │      ▼           ▼
   │      │       │        │       │     │  locations     times
   │      │       │        │       │     │                  │
   │      │       │        │       │     │           normalize_time()
   └──────┴───────┴────────┴───────┴─────┴──────────┬───────┘
                                                    ▼
                                                V3Result
```

| Head | Estimator | Why |
| :--- | :--- | :--- |
| `intent` | `CalibratedClassifierCV(LinearSVC(), cv=3)` | needs a real probability — the confidence gate and the score ladder both read it |
| `variables` | `OneVsRestClassifier(LogisticRegression(C=4.0))` | multi-label; a query can name several measurements |
| `insights` | `OneVsRestClassifier(LogisticRegression(C=4.0))` | multi-label; a week-long comparison wants several observations |
| `aggregation` | `LinearSVC(class_weight="balanced")` | `RAW` dominates ~93% of rows, so the classes must be reweighted |
| `detail` | `LinearSVC(class_weight="balanced")` | same imbalance — `NORMAL` is the default reading |
| `chart` | `LinearSVC(class_weight="balanced")` | same |

**Multi-label thresholds are calibrated, not guessed.** `_calibrate()` holds out the last 15%
of training rows, sweeps cuts from 0.20 to 0.50, and keeps the one with the best micro-F1 —
separately for `variables` and for `insights`. If nothing clears the cut, the head falls back
to its top-scoring class, so a prediction is never empty (Rule 3.2).

**Spans are a separate model.** `SpanTagger` (`src/tagger.py`) is a BIO tagger over the raw
text, not the cleaned features, because Rules 5.1/5.2 require spans to be verbatim. Its
vocabulary cutoff is chosen by `choose_min_word_freq()` from the training texts, and its
metric-noun list comes from `v2_dataset.VARIABLE_WORDS` so it knows "rainfall" is not a place.

Bundle: `models/nlu_v3.joblib`, 20.5 MB, joblib-pickled `V3Model`. Metrics written alongside
as `models/metrics_v3.json` at export time.

---

## 3. The Request Path

One WebSocket message, end to end. `backend/main.py`.

```text
 user text
     │
  1  normalize()                src/normalize.py     shorthand + typos folded, audit kept
     │
  2  registry.understand()      backend/registry.py  Model 1 -> Understanding (8 targets)
     │
  3  confirm_aggregation()      backend/insights.py  drop a reduction the prompt never said
     │
  4  detect_reference()         backend/state.py     "and there?" -> what does "there" mean
     is_follow_up()
     │
  5  context.apply()            backend/state.py     merge this turn into the chat's slots
     │                                               -> ConversationState + Operation
  6  locations.resolve()        backend/locations.py Solr lookup, alias table, GPS fallback
     │                                               -> lat/lng places
  7  planner.plan()             backend/planner.py   canonical time window, daily vs hourly
     │
  8  weather.daily_forecast()   backend/weather.py   WeatherSnap API
     weather.hourly_forecast()
     │
  9  respond.select_rows()      backend/respond.py   pick the rows the time expression meant
     respond.build_table()                           columns from understanding.fields()
     respond.summarize()
     │
 10  insights.apply_aggregation()  backend/insights.py
     insights.build_chart()        <- chart kind from the model
     insights.build_insights()     <- insight set from the model
     │
 11  store.record_turn()        backend/store.py     SQLite, tagged "[v3] <text>"
     │
     ▼
 frontend renders table + chart + notes
```

Steps 6–10 are the deterministic layer MODEL_RULES Section 1 keeps out of the model. The model
never sees a coordinate, a timestamp or a weather row.

**Where Model 1's three extra predictions land:**
- `detail` → `Understanding.fields()` → `fields_for(variables, detail)` → which columns step 8
  fetches and step 9 renders.
- `chart` → `insights.build_chart(kind=...)` — the model's choice wins over the row-count rule.
- `insights` → `insights.build_insights(wanted=...)` — only the selected observations compute.

---

## 4. File Map

### The model
| File | Role |
| :--- | :--- |
| `src/v3/model.py` | **Model 1.** `V3Model`, `train()`, `evaluate()`, `--export` CLI |
| `src/v3/schema.py` | `Detail`, `ChartKind`, `Insight`, `Presentation`, `V3Result`, `FIELD_SETS`, `fields_for()` |
| `src/v3/dataset.py` | Builds `data/v3_dataset.csv` — v2 turns relabelled with presentation, plus fresh detail phrasings |
| `src/v2/schema.py` | `Intent`, `Variable`, `Aggregation`, `Slots` — Model 1's slot contracts |
| `src/v2/dataset.py` | Conversation generator + `VARIABLE_WORDS`; upstream of the v3 dataset |
| `src/nlu.py` | `build_vectorizer()`, `clean_text()` — the shared encoder |
| `src/tagger.py` | `SpanTagger` (BIO), `normalize_time()`, `choose_min_word_freq()` |
| `src/normalize.py` | Pre-model text normalizer: shorthand, typos, casing |
| `src/schema.py` | `ConversationState`, `Operation`, `Reference`, `Verdict` — the context contracts |
| `src/build_dataset.py` | Template generator: balanced cells, misspellings, fillers, address forms |
| `src/data_loader.py` | CSV loading for the seed and eval files |
| `src/fetch_locations.py` | Samples real place names into `data/locations.csv` |

### The backend
| File | Role |
| :--- | :--- |
| `backend/main.py` | FastAPI app, the WebSocket, the request path above |
| `backend/registry.py` | Loads Model 1 once; `Understanding`, the shape the rest consumes |
| `backend/state.py` | Context engine — what the conversation remembers |
| `backend/planner.py` | Time windows, daily vs hourly, answer-or-ask |
| `backend/locations.py` | Solr resolution, aliases, relative places, GPS |
| `backend/weather.py` | WeatherSnap API clients |
| `backend/respond.py` | Rows → table + summary; `INTENT_FIELDS` |
| `backend/insights.py` | Aggregation guard, chart building, insight computation |
| `backend/store.py` | SQLite turn/feedback log and the retraining export |

### The frontend
| File | Role |
| :--- | :--- |
| `frontend/app/page.tsx` | Chat page; holds the model id and chat id |
| `frontend/components/composer.tsx` | Input + model pill, fed by `/api/models` |
| `frontend/components/ai-input.tsx` | Input primitives; `DEFAULT_MODELS` fallback |
| `frontend/components/messages.tsx` | Transcript, ratings, correction entry point |
| `frontend/components/correction.tsx` | Turns a thumbs-down into a labelled training row |
| `frontend/components/result-table.tsx` | The table Model 1's `detail` sized |
| `frontend/components/result-chart.tsx` | The chart Model 1 chose |
| `frontend/lib/use-weather-socket.ts` | WebSocket client |

### Data
| File | Tracked | Role |
| :--- | :--- | :--- |
| `data/v3_dataset.csv` | yes | **Model 1's training data**, 13,533 train rows |
| `data/intents.csv` | yes | Hand-written seed, head of the generation chain |
| `data/eval_manual.csv` | yes | Hand-written evaluation set — never generated |
| `data/locations.csv` | yes | 1,068 real place names, ~94% inside India |
| `data/location_aliases.json` | yes | Nicknames and spellings Solr will not match |
| `data/conversations.db` | no | Runtime chat log; export labels, do not commit |
| `data/processed/hard_cases.csv` | no | Harvested failures from the notebook |

### Tests
| File | Checks |
| :--- | :--- |
| `test_model.py` | Model 1: smoke, presentation, spans verbatim, canonical time, always-decides, accuracy floors |
| `test_conversations.py` | Replays 150 multi-turn conversations through the real pipeline |
| `test_dataset.py` | Eval-set coverage, location vocabulary; generated-split quality when present |

---

## 5. The Training Chain

Model 1 trains from `data/v3_dataset.csv` alone. That file is produced by a four-step chain
whose intermediates are **not** kept — they are byte-identical on every rebuild (fixed seeds),
so storing them was redundant.

```text
data/intents.csv          (hand-written seed, tracked)
data/locations.csv        (sampled vocabulary, tracked)
        │
        │  python src/build_dataset.py --split train     ~0.2 s
        │  python src/build_dataset.py --split test
        ▼
data/processed/nlu_dataset.csv, nlu_test.csv     ← intermediate, not kept
        │
        │  python -m src.v2.dataset --build
        ▼
data/v2_dataset.csv                              ← intermediate, not kept
        │
        │  python -m src.v3.dataset --build
        ▼
data/v3_dataset.csv        13,533 train rows     ← TRACKED, this is what trains Model 1
        │
        │  python -m src.v3.model --export       ~18 s
        ▼
models/nlu_v3.joblib + models/metrics_v3.json
```

**To retrain from the shipped dataset** (the common case — no intermediates needed):

```bash
python -m src.v3.model --export
python test_model.py
```

**To rebuild the dataset from the seed** (after editing `data/intents.csv` or folding in user
corrections), run all four steps in order. `test_dataset.py` checks the intermediates'
balance and noise properties whenever they are on disk, and skips those checks when they are
not.

**To fold real usage back in**: `python -m backend.store --export data/from_users.csv`, then
`src/v2/dataset.py::from_users()` picks it up on the next build. Human labels outrank the
model's (Rule 8.5).

---

## 6. Measured Performance

`models/metrics_v3.json`, written at export time. Trained on 13,533 rows.

| Split | rows | intent | vars | detail | chart | locs | times | insights F1 | **everything** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test (generated) | 2,157 | .964 | .961 | .999 | .981 | .957 | .956 | .935 | **.839** |
| detail phrasings | 270 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | .900 | .976 | **.900** |
| eval (hand-written English) | 219 | .959 | .922 | 1.00 | .840 | .945 | .959 | .907 | **.680** |

Multi-turn context, replayed through the real pipeline (`test_conversations.py`, 150
conversations / 437 turns): operation 100%, locations 100%, times 100%, variables 99.8%.

`everything` — all eight targets right on the same turn — is the honest number. The gap
between generated test (.839) and hand-written eval (.680) is the cost of typos, code-mixing
and phrasings no template produced; `chart` at .840 is the weakest single head on real
wording and is where the next round of training data should go.

---

## 7. What Was Removed

Deleted in the Model 1 cleanup. All of it is recoverable — the datasets from git history, the
bundles by retraining.

| Removed | Size | Replacement |
| :--- | ---: | :--- |
| `models/nlu_pipeline.joblib` | 19.5 MB | Model 1 |
| `models/nlu_v2.joblib` | 14.0 MB | Model 1 |
| `models/metrics.json`, `metrics_v2.json` | — | `models/metrics_v3.json` |
| `data/v2_dataset.csv` | 2.7 MB | regenerable intermediate |
| `data/processed/nlu_dataset.csv`, `nlu_test.csv` | 1.0 MB | regenerable intermediates |
| `frontend/components/model-switch.tsx` | — | unused; the composer pill lists models |
| v1/v2 adapters in `backend/registry.py` | — | one `_understand()` |

`models/` went from 52 MB to 20 MB.

The version switcher is gone from the serving path: `/api/models` returns one entry,
`registry.get()` ignores the version argument callers still pass, and `DEFAULT_VERSION` is
the only version. Old turns in `data/conversations.db` tagged `[v1]` or `[v2]` still display;
turns with no tag at all report as `legacy`.

---

## 8. Operations

```bash
# run it
./scripts/run_app.sh                   # backend :8787 + frontend :3000

# check it
python test_model.py                   # Model 1 floors and invariants
python test_conversations.py           # multi-turn context
python test_dataset.py                 # data coverage
python backend/registry.py             # 5-second smoke test

# ask it one question
python -m src.v3.model "rain and temperature in Guntur tomorrow"

# retrain it
python -m src.v3.model --export        # ~18 s from data/v3_dataset.csv
```

`/api/health` reports whether the bundle is loaded; `/api/models` reports its size and
description; `/api/labels` serves Model 1's five enum sets to the correction form.
