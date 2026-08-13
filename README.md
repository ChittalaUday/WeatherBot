# WeatherBot ML Training Setup

Machine Learning training environment and Jupyter Notebook setup for WeatherBot.

## Project Structure

```
WeatherBot/
├── data/
│   ├── raw/                # Raw dataset copies
│   └── processed/          # Cleaned / preprocessed dataset features
├── models/                 # Saved model checkpoints (*.joblib)
├── notebooks/              # Jupyter notebooks for EDA and prototyping
│   └── exploration.ipynb
├── src/                    # Modular Python source code
│   ├── __init__.py
│   ├── build_dataset.py    # Generates the train / test splits from templates + shapes names
│   ├── data_loader.py      # CSV loading, span parsing
│   ├── fetch_locations.py  # Read-only sampler for the `shapes` schema
│   ├── nlu.py              # Train / export / serve the model  (entry point)
│   ├── schema.py           # WeatherIntent / Action enums, NLUOutput
│   └── tagger.py           # BIO span tagger for LOCATION / TIME
├── .gitignore
├── requirements.txt        # Python package dependencies
└── README.md
```

## Quickstart Setup

### 1. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Register Jupyter Kernel

Register the virtual environment as a selectable Jupyter kernel in your IDE:

```bash
python -m ipykernel install --user --name=weatherbot --display-name "Python (WeatherBot)"
```

### 4. Build the Datasets

First refresh the LOCATION vocabulary - real village / block / district / state names
sampled **read-only** from the `shapes` schema (needs `psql` and `DB_*` in `.env`):

```bash
python src/fetch_locations.py               # -> data/locations.csv (1068 names, ~94% inside India)
```

Only `SELECT`s against `shapes.*` are issued, on a `default_transaction_read_only`
session, and only names + shape centroids are exported - never identifiers. Without the
CSV the builder falls back to a small built-in city list.

Then generate the two training splits, which share no prompt with each other or with the
evaluation set:

```bash
python src/build_dataset.py --split train   # data/processed/nlu_dataset.csv  6300 rows, fitting
python src/build_dataset.py --split test    # data/processed/nlu_test.csv     1512 rows, iterate on failures
python test_dataset.py                      # validates all three sets against MODEL_RULES.md
```

The evaluation set, `data/eval_manual.csv`, is **hand-written and must stay that way** -
209 rows of typos ("temprature", "tomorow"), Hindi/Telugu code-mixing ("kal barish hogi
kya", "vaana padutunda ... repu"), ellipsis ("temp Tarora", "rain?"), ALL CAPS, addresses,
odd clock times, thresholds and negation. A generator cannot evaluate its own templates, so
`build_dataset.py` never writes this file. It also uses only place names reserved as
`split == eval` in `data/locations.csv`, which are never generated into train or test.

The builder enforces `MODEL_RULES.md`: labels come from the enums in `src/schema.py`,
every `LOCATION`/`TIME` annotation is a verbatim substring of the prompt, all 14 × 3
cells are equally sized, and prompts vary syntactic order (question-first,
location-first, time-first) plus casing, punctuation, relative locations
("near me", "in my field") and missing-entity cases.

About one prompt in six carries an injected spelling mistake (always outside the annotated
spans, so every span stays verbatim), and the metric vocabulary itself ships misspellings
("temprature", "wnd speed", "humidty") and Hindi/Telugu words ("mausam", "barish", "dhoop").

Locations appear both bare ("Angara") and as partial or full addresses
("Angara, East Godavari", "Angara, Rajahmundry, East Godavari, Andhra Pradesh"), villages
dominating. Times mix calendar-relative phrases with generated wall-clock values - mostly
round hours and half hours, with a tail of odd minutes ("6 PM", "9:30 am", "1:48 PM",
"from 7 AM to 11 AM", "in 45 minutes").

Retraining loop: read `data/processed/hard_cases.csv` (written by the notebook), fold the
missing phrasings into the generator, rebuild, retrain. Never copy an eval row into the
generator - the moment training material reaches `data/eval_manual.csv`, it stops measuring
anything.

### 5. Entity Extraction

`src/tagger.py` holds the BIO span tagger for `LOCATION` / `TIME` (`O`, `B-LOC`, `I-LOC`,
`B-TIME`, `I-TIME`), sklearn only. It replaced a gazetteer, which could only return spans it
had memorised and so found 13% of the locations in the hand-written eval set against the
tagger's 76%. Two design points carry that difference:

- **rare-word masking** - words below a frequency cut are hidden behind `<rare>` while
  fitting, so the model cannot pass training by memorising village names and has to learn
  context and word shape instead. `choose_min_word_freq()` picks the cut from the *training*
  split (smallest cut where no real place name survives in the vocabulary); the eval set is
  never consulted.
- **the gazetteer survives as a feature**, not as the decider, so memorisation still helps
  where it is right.

Wall-clock and duration expressions get a deterministic regex pass that wins on overlap -
`4:15 pm` is a time whether or not that exact string was ever generated.

```bash
python src/tagger.py                        # self-check: tags a village it never saw
```

### 6. Train, Export and Serve

`src/nlu.py` owns the model, so the notebook and the exported bundle cannot drift apart:

```bash
python src/nlu.py --export                      # -> models/nlu_pipeline.joblib + models/metrics.json
python src/nlu.py                               # interactive: type a query, ~5 ms per answer
python src/nlu.py "will it rain in Guntur at 6:45 pm tomorrow?"   # one shot, prints JSON
python src/nlu.py --info                        # bundle size, components, latency
```

Serving, from any process with `src/` importable (the tagger's feature builder lives in
`src/tagger.py`, which keeps inference on byte-identical features):

```python
from src.nlu import NLUModel

model = NLUModel.load()
model.predict("compare max temp between Nokha and Buxar this weekend")  # -> NLUOutput
model.confidence("...")            # max intent probability, for an "ask the user" fallback
```

Check a build before shipping it:

```bash
python test_model.py                            # smoke queries, verbatim spans, accuracy floors
```

It fails if the bundle is missing, a smoke query breaks, a predicted span is not verbatim in
its prompt, or English eval accuracy drops below the floors in `FLOORS` - and warns when the
bundle is older than the training CSV. Code-mixed rows are printed, never asserted.

**Time comes back in one shape.** Every raw `time` span gets a positionally aligned
`time_normalized` twin, so downstream queries never see the user's spelling (Rule 4.3):

| user typed | `time` (raw, Rule 4.2) | `time_normalized` (query on this) |
| :-- | :-- | :-- |
| will it rain tommorrow | `["tommorrow"]` | `["tomorrow"]` |
| rain in Nokha from 7 AM to 11 AM | `["7 AM to 11 AM"]` | `["07:00-11:00"]` |
| temp at 6:45 pm | `["6:45 pm"]` | `["18:45"]` |
| wind rn | `["rn"]` | `["now"]` |

Clock times become 24h `HH:MM`, ranges `HH:MM-HH:MM`, durations `next N <unit>`. Unknown
expressions pass through lowercased rather than being guessed at. This canonicalises the
surface form only - resolving to an actual datetime stays with the deterministic Time Parser.

**Language policy.** English is the target. The evaluation set carries 16 Hindi/Telugu
code-mixed rows tagged `lang == "mixed"`; they are scored separately as a diagnostic and
never folded into the headline number.

Accuracy as exported (`models/metrics.json`), where *all 4* means every target correct on
one prompt - what the deterministic layer downstream actually consumes:

| set | intent | action | location F1 | time F1 | all 4 |
| :-- | --: | --: | --: | --: | --: |
| set | intent | action | aggregation | location F1 | time F1 | all 5 |
| :-- | --: | --: | --: | --: | --: | --: |
| test (generated, 1512) | 96.6% | 99.4% | 98.3% | 0.954 | 0.988 | 89.3% |
| **eval (English, 219)** | **94.1%** | **97.7%** | **95.0%** | **0.963** | **0.964** | **80.4%** |
| eval (code-mixed, 16) | 75.0% | 18.8% | 100% | 0.732 | 0.000 | 0.0% |

"all 5" means every target correct on one prompt - the number the deterministic layer
actually depends on. It fell from the earlier 4-target 88.1% because the eval set gained
26 harder hand-written rows (conversational padding, aggregations, 3-way comparisons) and
a fifth target to get right.

**Location resolution is a separate layer** (`backend/locations.py`), never the model's job:
the model reports `"KKD"` verbatim (Rule 4.1) and the resolver turns it into
Kakinada, Andhra Pradesh via `data/location_aliases.json` -> Solr -> ranked candidates.
Teaching the system a new abbreviation is a one-line edit to that JSON, not a retraining run.
When two real places match equally well ("Angara" in Jharkhand and in Andhra Pradesh) the
resolver returns every candidate and the chat asks; a district seat never counts as ambiguous
against a same-named hamlet.

The tagger deliberately does **not** get a "is this token in the gazetteer" feature. During
fitting every training place name is in the gazetteer by construction, so the model learned
`in gazetteer -> LOCATION` and treated the first unseen village as an ordinary word - eval
location F1 was 0.79 with it and 0.97 without. Character shingles and context carry the
decision instead, which is what transfers to `hyderbad` and `kukatpalle`.

### 7. Chat App (backend + frontend)

```bash
./scripts/run_app.sh          # FastAPI on :8787, Next.js on :3001
```

Or run the halves separately:

```bash
.venv/bin/uvicorn backend.main:app --port 8787 --reload
cd frontend && npm run dev
```

Ports 8000 and 3000 are taken by other apps on this machine, so the app is pinned to
**8787** (backend) and **3001** (frontend).

**Backend** (`backend/`) - one WebSocket carries the whole conversation, streaming each
stage so the UI shows progress instead of a spinner:

| direction | message |
| :-- | :-- |
| client → | `{"type":"query","text":"will it rain in Nokha tommorrow?"}` |
| client → | `{"type":"location","text":"<pending query>","lat":17.38,"lon":78.48}` |
| ← server | `{"type":"status","stage":"understanding\|locating\|fetching"}` |
| ← server | `{"type":"nlu","intent","action","entities","confidence"}` |
| ← server | `{"type":"need_location", ...}` → browser geolocation, resent as `location` |
| ← server | `{"type":"result","summary","table","places","series","unresolved"}` |

- `backend/weather.py` - WeatherSnap clients: Solr for text → lat/lng (with nicknames, so
  "Vizag" finds Visakhapatnam), `/api/centroids` for browser coords → place name,
  `/interpolate` and `/hrlydata` for the forecast.
- `backend/respond.py` - the deterministic half of MODEL_RULES.md Section 1: intent →
  API field (`RAIN` → `Rainfall`, `SOIL_MOISTURE` → `Soilm10/Soilm40`), canonical time →
  which rows, `COMPARE` → a column per place. No model involved.
- `GET /api/suggest?q=` - Solr autocomplete, polled by TanStack Query.
- **Guardrails**: below 45% intent confidence the backend sends `clarify` with the top three
  readings instead of a table, and a `COMPARE` naming only one place asks for the second
  rather than quietly answering about one. "angara vs hyderbad" names no metric at all, so
  it asks rather than guessing rain one turn and temperature the next.
- Login is deliberately unused: the collection's `/user/login` is never called.

**Frontend** (`frontend/`) - Next.js App Router, shadcn/ui, Tailwind, lucide icons,
TanStack Query (health + suggestions) and TanStack Table (sortable result tables). The chat
shows what the NLU understood as chips - intent, action, each location, and each time span
with its canonical form (`tommorrow → tomorrow`) - then the summary and the table.

**Location flow.** When the text names no place, or names a relative one ("my field",
"near me"), the backend replies `need_location` instead of guessing. The UI then asks the
browser for coordinates and replays the original question with them attached. Deny the
permission and it tells you to name a place instead.

### 8. Pipeline

Every stage is deterministic except the two model calls, and each one can be tested alone:

```text
USER TEXT
   │
   ▼  src/normalize.py        "What's da wthr in KKD tmrw?" -> "what is the weather in KKD tomorrow?"
Normalizer                    every substitution recorded; place names never rewritten
   │
   ▼  backend/state.py        "there", "what about ..." matched by rules, not a model
Cheap rules
   │
   ▼  src/nlu.py              intent + action + aggregation + LOCATION/TIME spans + scores
Small NLU model
   │
   ▼  backend/state.py        SET / REPLACE / MODIFY / INHERIT / COMPARE
Context engine                a low-confidence fragment inherits the previous intent
   │
   ▼  backend/locations.py    alias table -> Solr -> ranked candidates -> ambiguity asked
Location resolver
   │
   ▼  backend/planner.py      canonical wording -> absolute window, by arithmetic
Time resolver
   │
   ▼  backend/planner.py      READY / CLARIFY / REJECT
Validator
   │
   ▼  backend/weather.py      WeatherSnap
Weather API
   │
   ▼  backend/respond.py + insights.py   table, chart, aggregation, insights - templates only
Response
```

No generative model anywhere. The contracts (`Normalized`, `NLUResult`, `ConversationState`,
`ResolvedQuery` in `src/schema.py`) are frozen, so TF-IDF can be swapped for fastText or a
MiniLM encoder without touching a single resolver.

**Multi-turn**, all deterministic:

| turn | operation | state after |
| :-- | :-- | :-- |
| "What's da wthr in KKD?" | SET | Kakinada, near term |
| "what about tomorrow?" | MODIFY | Kakinada, tomorrow |
| "what about Rajahmundry?" | REPLACE | Rajahmundry, tomorrow |
| "and there?" | INHERIT | Rajahmundry, tomorrow |
| "total rainfall there next 3 days" | MODIFY | Rajahmundry, next 3 days, SUM |

**Confidence routing** comes from `python src/nlu.py --calibrate`, not from taste:

| band | measured accuracy | share of turns | behaviour |
| :-- | --: | --: | :-- |
| >= 0.95 | 98.9% | 83% | answer |
| 0.45 - 0.95 | ~75% | 15% | answer, flag it, queue for review |
| < 0.45 | 0-67% | 2% | ask instead of guessing |

### 9. Conversation Store & Retraining Loop

Every turn lands in SQLite (`data/conversations.db`, gitignored) - text, the 4 predicted
targets, confidence, resolved places, outcome and latency:

```bash
python -m backend.store --stats           # volume, outcomes, feedback counts
python -m backend.store --recent 20       # last 20 turns
python -m backend.store --export data/from_users.csv
python -m backend.store --selfcheck       # label in -> training row out
python -m backend.store --confusion       # predicted vs actual, from labels
python -m backend.store --competing       # turns where two intents were close
python src/nlu.py --calibrate             # accuracy per confidence band
```

Each turn stores the **full score vector**, not just the winner, so you can see which
intents competed. `--competing` ranks the closest calls: those are the examples worth
labelling first, and `--confusion` says which pair keeps colliding.

The export writes the exact schema of `data/intents.csv`, so real usage folds straight into
the next build: append it to the seed, `python src/build_dataset.py --split train`,
`python src/nlu.py --export`, then confirm the frozen hand-written eval set improved.

Not all feedback is equal, and the store treats it that way:

| signal | source | used as a label? |
| :-- | :-- | :-- |
| `choice` | user picked an intent from a clarify prompt | **yes** - the model asked, a human answered |
| `correction` | user said what it should have been | **yes** |
| `up` | thumbs up | only above 0.9 confidence, and only with `--include-approved` |
| `down` | thumbs down | no - says something is wrong, not what |

Honest naming: **this is not reinforcement learning.** There is no reward signal and no
policy - it is supervised retraining fed by real users instead of templates. The clarify
prompts are what make it work: a question the model asks turns into a free gold label.

### 10. Model v2 (selectable, alongside v1)

v1 is untouched and still the default. v2 restructures the targets as you would for an AI
context builder - **one coarse intent, everything else a multi-value slot**:

| | v1 | v2 |
| :-- | :-- | :-- |
| intent | 14 classes (variable folded in) | 6 coarse: CURRENT / FORECAST / HISTORICAL / COMPARE / ALERT / UNKNOWN |
| weather variable | *is* the intent, one per query | multi-label slot, 13 labels, several per query |
| locations / times | multi-span | multi-span (unchanged tagger) |
| training data | 3 CSV files | one SQLite table, `split` and `source` columns |
| training time | ~3 min (SVC + probability) | ~20 s (LinearSVC + calibration) |

```bash
python -m src.v2.dataset --build     # data/v2_dataset.db from v1 rows + multi-variable + user labels
python -m src.v2.dataset --stats
python -m src.v2.model --export      # -> models/nlu_v2.joblib
python -m src.v2.model "rain and temperature in Guntur and Vizag tomorrow"
```

The multi-label threshold is calibrated on a held-out slice at training time (0.25 this run),
not hand-picked: 0.35 was silently dropping TEMPERATURE at 0.205 from "temperature, humidity
and rainfall".

| test split | intent | variables F1 | exact set | locations | all slots |
| :-- | --: | --: | --: | --: | --: |
| all rows (1887) | 97.5% | 0.971 | 95.0% | 96.1% | 85.4% |
| **multi-variable only** | **100%** | **0.996** | **98.3%** | **100%** | **89.2%** |

The difference on the query v1 cannot express:

```text
"rain and temperature in Guntur tomorrow"
  v1  TEMPERATURE / COMPARE, 28%   -> asks which reading you meant
  v2  FORECAST, [RAIN, TEMPERATURE], 83%   -> one table, both columns
```

**Choosing a model.** `GET /api/models` lists what is deployed with each version's metrics;
every WebSocket query takes an optional `"model": "v1" | "v2"`; the UI has a v1/v2 switch in
the header and tags each answer with the version that produced it. `backend/registry.py`
adapts both into one `Understanding`, so the context engine, resolvers and response builder
never learn which model answered.

### 11. Using Jupyter Notebooks

1. Open `notebooks/exploration.ipynb` in your IDE.
2. In the top right corner, select the kernel **`Python (WeatherBot)`**.
3. Run cells to perform EDA and model evaluation interactively!
