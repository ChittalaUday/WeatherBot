# WeatherBot ML Training Setup

Machine Learning training environment and Jupyter Notebook setup for WeatherBot.

**This project ships one model: Model 1.** It predicts eight things per turn — intent, a set
of weather variables, an aggregation, location and time spans, and the three presentation
decisions (how much detail, which chart, which insights). It never asks a clarifying
question; it commits to a reading and reports what it assumed.

- [MODEL_RULES.md](MODEL_RULES.md) — the taxonomy, the annotation rules, the accuracy floors
- [ARCHITECTURE.md](ARCHITECTURE.md) — the heads, the request path, the file map, the
  training chain, and what the two earlier models could not do

## Project Structure

```
WeatherBot/
├── data/
│   ├── raw/                # Raw dataset copies
│   ├── processed/          # Build-time intermediates (regenerable, not kept)
│   ├── intents.csv         # Hand-written seed - head of the generation chain
│   ├── eval_manual.csv     # Hand-written evaluation set - never generated
│   ├── locations.csv       # 1068 real place names sampled from `shapes`
│   └── v3_dataset.csv      # Model 1's training data
├── models/                 # nlu_v3.joblib + metrics_v3.json  (bundles are gitignored)
├── notebooks/              # Jupyter notebooks for EDA and prototyping
│   └── exploration.ipynb
├── src/                    # Modular Python source code
│   ├── build_dataset.py    # Template generator - balanced cells, misspellings, fillers
│   ├── data_loader.py      # CSV loading, span parsing
│   ├── fetch_locations.py  # Read-only sampler for the `shapes` schema
│   ├── nlu.py              # Shared encoder: build_vectorizer(), clean_text()
│   ├── normalize.py        # Pre-model text normalizer
│   ├── schema.py           # Conversation-state contracts
│   ├── tagger.py           # BIO span tagger for LOCATION / TIME
│   ├── v2/                 # Slot contracts + conversation generator (library code)
│   └── v3/                 # Model 1: model.py, schema.py, dataset.py  (entry point)
├── backend/                # FastAPI + WebSocket serving layer
├── frontend/               # Next.js chat app
├── .gitignore
├── requirements.txt        # Python package dependencies
└── README.md
```

> `src/v2/` and `src/v3/` name the *architecture generations*, not selectable models. Model 1
> is `src/v3/`; `src/v2/` survives only as the slot enums and the dataset generator Model 1
> is built from. See [ARCHITECTURE.md §1](ARCHITECTURE.md#1-why-model-1-is-complete).

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

Then run the generation chain. The first three outputs are **build-time intermediates** -
byte-identical on every rebuild (fixed seeds), so they are not kept in the repo. Only
`data/v3_dataset.csv` at the end is tracked, and that is the file Model 1 trains from:

```bash
python src/build_dataset.py --split train   # data/processed/nlu_dataset.csv  6300 rows   (intermediate)
python src/build_dataset.py --split test    # data/processed/nlu_test.csv     1512 rows   (intermediate)
python -m src.v2.dataset --build            # data/v2_dataset.csv             turns in chats (intermediate)
python -m src.v3.dataset --build            # data/v3_dataset.csv             + presentation labels
python test_dataset.py                      # validates the sets against MODEL_RULES.md
```

The splits share no prompt with each other or with the evaluation set. `test_dataset.py`
checks the intermediates' balance and noise when they are on disk, and skips those checks
when they are not - so a clean checkout that only retrains from `v3_dataset.csv` still passes.

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

`src/v3/model.py` owns Model 1, so the notebook and the exported bundle cannot drift apart:

```bash
python -m src.v3.model --export                 # -> models/nlu_v3.joblib + models/metrics_v3.json  (~18 s)
python -m src.v3.model "will it rain in Guntur at 6:45 pm tomorrow?"   # one shot, prints JSON
python backend/registry.py                      # 5-second smoke test through the serving adapter
```

Serving, from any process with `src/` importable (the tagger's feature builder lives in
`src/tagger.py`, which keeps inference on byte-identical features):

```python
from src.v3.model import V3Model

model = V3Model.load()
result = model.predict("compare max temp between Nokha and Buxar this weekend")
result.slots.variables           # [Variable.TEMPERATURE_MAX]
result.presentation.chart        # ChartKind.GROUPED_BAR
result.confidence["intent"]      # calibrated intent probability
```

Check a build before shipping it:

```bash
python test_model.py                            # smoke, presentation, verbatim spans, accuracy floors
python test_conversations.py                    # multi-turn context through the real pipeline
```

`test_model.py` fails if the bundle is missing, a smoke query breaks, a predicted span is not
verbatim in its prompt, any turn comes back undecided, or English eval accuracy drops below
the floors in `FLOORS` - and warns when the bundle is older than `data/v3_dataset.csv`.
Code-mixed rows are printed, never asserted.

**Time comes back in one shape.** Every raw `times` span gets a positionally aligned
`times_normalized` twin, so downstream queries never see the user's spelling (Rule 5.4):

| user typed | `times` (raw, Rule 5.2) | `times_normalized` (query on this) |
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

Accuracy as exported (`models/metrics_v3.json`), where *everything* means all eight targets
correct on one prompt - what the deterministic layer downstream actually consumes:

| set | intent | variables | detail | chart | location | time | insights F1 | everything |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| test (generated, 2157) | 96.4% | 96.1% | 99.9% | 98.1% | 95.7% | 95.6% | 0.935 | 89.3%* |
| detail phrasings (270) | 100% | 100% | 100% | 100% | 100% | 90.0% | 0.976 | 90.0% |
| **eval (English, 219)** | **95.9%** | **92.2%** | **100%** | **84.0%** | **94.5%** | **95.9%** | **0.907** | **68.0%** |

<sub>* generated test `everything` is 83.9%; 89.3% is the four-target number the earlier model
reported, kept here only for comparison.</sub>

`everything` is the honest number, and it is lower than any single head because it demands
all eight at once. The gap between generated test and hand-written eval is the cost of typos,
code-mixing and phrasings no template produced. `chart` at 84.0% is the weakest head on real
wording - that is where the next round of training data should go.

**Location resolution is a separate layer** (`backend/locations.py`), never the model's job:
the model reports `"KKD"` verbatim (Rule 5.1) and the resolver turns it into
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

**Frontend** (`frontend/`) - Next.js App Router, shadcn/ui, Tailwind, TanStack Query
(health + models) and TanStack Table (sortable result tables).

- the composer is the `ai-input` component from chamaac.com, wired to our socket. Its own
  `AIInput` keeps an internal message list and fakes a reply, which would fight the real
  transcript, so `components/composer.tsx` reuses its dropdown and pill primitives and keeps
  the state ours. Its stock model list (GPT-4o, Claude…) is replaced by our two NLU bundles -
  there is no third-party LLM anywhere in this app
- icons are the animated lucide set from animateicons.in, which animate on hover
- one sky-blue accent across light and dark, `next-themes` toggle in the header, chart series
  derived from the same tokens so nothing is tuned twice
- messages grow up from the composer rather than stranding one answer at the top of a tall
  screen; the empty state centres the composer under a short hero with example prompts

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
the next build: append it to the seed, rerun the generation chain (Section 4), retrain with
`python -m src.v3.model --export`, then confirm the frozen hand-written eval set improved.

Not all feedback is equal, and the store treats it that way:

| signal | source | used as a label? |
| :-- | :-- | :-- |
| `choice` | user picked an intent from a clarify prompt | **yes** - the model asked, a human answered |
| `correction` | user said what it should have been | **yes** |
| `up` | thumbs up | only above 0.9 confidence, and only with `--include-approved` |
| `down` | thumbs down | no - says something is wrong, not what |

Every answer is rateable, including one reopened from history: the stored payload carries
its own `turn_id`, and the UI only reports success once the request actually succeeded.

**One verdict per turn.** A rating is the user's current opinion, not a log of clicks, so
`feedback.turn_id` is UNIQUE and every submission upserts: thumbs-up, changed to thumbs-down,
then corrected, leaves one row with `revisions = 2` and the corrected label - not three rows,
two of which would train the model on a turn that was already relabelled. `ts` is when it was
first rated, `updated_at` when it last changed, and fields left blank keep their previous
value, so fixing only the place does not erase an intent that was already right. Reopening a
chat shows the verdict already recorded, with a **change** link to revise it.

**Thumbs-down opens a correction form** rather than just registering displeasure, because a
bare "wrong" cannot be trained on. The form is pre-filled with the model's own reading, so
fixing it is usually one tap:

```text
answer:     HUMIDITY in Kakinada tomorrow          <- the model's reading
correction: DEW_POINT in Kakinada day after tomorrow
stored:     kind=correction, intent=DEW_POINT, location=["Kakinada"],
            time=["day after tomorrow"], error_type=intent_confusion, model=v1
exported:   humidity in Kakinada tomorrow,DEW_POINT,GET,...,correction,v1,intent_confusion
```

The correction form offers Model 1's 13 variables, multi-select, fetched from
`GET /api/labels` so the enums stay the single source of truth. `python -m backend.store --review` lists what still needs a label: everything
flagged wrong, plus uncertain turns nobody judged.

Full loop:

```bash
python -m backend.store --review                  # what needs labelling
python -m backend.store --export data/from_users.csv

python -m src.v2.dataset --build                  # reads the store directly (source=users)
python -m src.v3.dataset --build                  # relabel with presentation
python -m src.v3.model --export
python test_model.py && python test_conversations.py
```

`src/v2/dataset.py::from_users()` picks up the exported rows, so a correction reaches Model 1
without touching `data/intents.csv`. Human labels outrank the model's (MODEL_RULES Rule 8.5).

Honest naming: **this is not reinforcement learning.** There is no reward signal and no
policy - it is supervised retraining fed by real users instead of templates. The clarify
prompts are what make it work: a question the model asks turns into a free gold label.

### 10. Model 1 - one model, eight decisions

Model 1 is the only model served. Two earlier ones were built and deleted: a 14-class
single-variable classifier, and a coarse-intent slot filler that still left presentation to
Python. The full comparison is in [ARCHITECTURE.md §1](ARCHITECTURE.md#1-why-model-1-is-complete);
the short version is that Model 1 predicts everything needed to *answer*, not just everything
needed to look the answer up.

| head | labels | replaces |
| :-- | :-- | :-- |
| `intent` | CURRENT / FORECAST / HISTORICAL / COMPARE / ALERT / UNKNOWN | 14 classes with the variable folded in |
| `variables` | 13 labels, **multi-label** | one variable per query |
| `aggregation` | RAW / SUM / AVG / MAX / MIN / TREND | a keyword rule |
| `detail` | MINIMAL / NORMAL / FULL | `INTENT_FIELDS[intent]`, a fixed column list |
| `chart` | NONE / STAT / LINE / MULTI_LINE / GROUPED_BAR | a branch on how many series came back |
| `insights` | 9 labels, **multi-label** | computing every observation, every time |

```bash
python -m src.v3.model --export      # -> models/nlu_v3.joblib   (~18 s)
python -m src.v3.model "temperature in Guntur tomorrow in detail"
python -m src.v3.dataset --stats
python test_conversations.py         # replay held-out chats through the pipeline
```

What the presentation heads buy, on the same query:

```text
"temperature in Guntur tomorrow"            before  Avg temp             now  Avg temp
"temperature in Guntur tomorrow in detail"  before  Avg temp             now  Min, Max, Avg
"full temperature breakdown this week"      before  Avg temp, line       now  Min, Max, Avg, line,
                                                                              peak + low + threshold
"rain, temperature and humidity next week"  before  5 columns, 1 line    now  3 columns, 3-series
                                                    (incl. humidity max/min)  multi-line
"just the rainfall in Nokha this week"      before  asks which Nokha     now  answers, one column,
                                                                              no chart
"rain and temperature in Guntur tomorrow"   before  one variable, 28%    now  both, 83%, one table
```

**The training data is turns grouped into chats**, not a pile of isolated sentences -
13,533 training rows carrying `chat_id`, `turn` and the context slots:

| chat_id | turn | text | operation | variables |
| :-- | --: | :-- | :-- | :-- |
| gen-tr-00337 | 0 | what is the wind in Tirupati tomorrow? | SET | WIND_SPEED |
| gen-tr-00337 | 1 | what about Chikmagalur then? | REPLACE | WIND_SPEED |
| gen-tr-00337 | 2 | same for Kollam? | REPLACE | WIND_SPEED |

A follow-up row carries the slots the user *means*, which is not what their words say -
"what about Chikmagalur then?" names no measurement. So the file is two things: per-turn
training data for the heads, and a replay script for the context engine. Per-utterance
metrics cannot judge a follow-up, so `test_conversations.py` replays every held-out chat
through model + normalizer + context engine and scores the **state** after each turn:
operation 100%, locations 100%, times 100%, variables 99.8% over 437 turns.

The multi-label thresholds are calibrated on a held-out slice at training time, not
hand-picked: 0.35 was silently dropping TEMPERATURE at 0.205 from "temperature, humidity and
rainfall".

**Model 1 never asks.** Below the confidence floor, on a one-sided comparison, or on an
ambiguous place name it commits and reports the assumption - `Assumed: angara = Angara,
Jharkhand` - rather than interrupting (`registry.NEVER_ASKS`).

**Chats and history.** Every turn belongs to a `chat_id` the browser owns (localStorage), so
a reload resumes the same conversation with its slots intact; **New** starts a fresh one.
**History** lists past conversations and reopens any of them.

- `GET /api/chats` - recent conversations, titled by their opening question
- `GET /api/chats/{chat_id}` - one conversation, with each answer as it was rendered
- `GET /api/models` - the served model, its size and its exported metrics
- `GET /api/labels` - Model 1's five enum sets, for the correction form

Each answered turn stores its **rendered payload** (summary, table, chart, insights) and the
slot state that produced it. Reopening a chat replays what was actually shown rather than
re-querying a forecast that has since moved on, and the restored chat keeps its context, so
"what about tomorrow?" still works after the reload - even across a backend restart, because
the state is rebuilt from the stored turns. Turns are tagged with the architecture id that
answered them (`[v3]`); older turns from the deleted models still display as they were.

Stated plainly: the chart and insight labels start as rules distilled into the model, so
Model 1 began no smarter than the teacher. What it gained immediately is generalisation over
phrasing - "in detail", "full breakdown" and "all the numbers" all widen the table without
any of them being enumerated - and a wrong chart becomes a labelled example instead of an
argument about an if-statement. Beating the teacher needs the correction loop in Section 9.

### 11. Using Jupyter Notebooks

1. Open `notebooks/exploration.ipynb` in your IDE.
2. In the top right corner, select the kernel **`Python (WeatherBot)`**.
3. Run cells to perform EDA and model evaluation interactively!
