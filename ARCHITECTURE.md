# Architecture

What every part of this repo does and why it is shaped that way. The label taxonomy and the
annotation rules live in [MODEL_RULES.md](MODEL_RULES.md); this file is the machinery.

---

## 1. The shape of it

Four layers, each with one job, each replaceable without touching the others.

```text
                       HTTP  (backend/api)
                            │
      ┌─────────────────────┼──────────────────────┐
      │                     │                      │
  conversation          understanding            answer
  backend/nlu/          backend/nlu/           backend/pipeline
  context.py            registry.py                 │
      │                 llm.py                      │
      │                     │                       │
      └──── slots ──────────┴──── Understanding ────┤
                                                    │
                                            backend/generation
                                            (the answer, in words)
```

| Layer | Package | Owns |
| :--- | :--- | :--- |
| Transport | `backend/api` | HTTP routes, SSE framing, the turn log |
| Understanding | `backend/nlu` | text → `Understanding`; what the conversation remembers |
| Answer | `backend/pipeline` | places, plan, fetch, quality, analysis, advice, table |
| Wording | `backend/generation` | retrieval, prompts, the local model |
| Config | `backend/config.py` | every setting, read once |
| Storage | `backend/store.py` | turns, feedback, the retraining export |

The rule that keeps it honest: **the model never sees a coordinate, a timestamp or a weather
row, and the generation layer never sees anything the pipeline did not compute.** Everything
between those two is deterministic.

---

## 2. One turn, end to end

`backend/api/chat.py::turn` is an async generator. The endpoint streams it; the self-check
collects it into a list. Nothing about the turn knows it is being served over HTTP.

```text
 POST /api/chat  {"text": "will it rain in Guntur tomorrow", "chat_id": "chat-1"}
     │
  1  normalize_text()          src/normalize.py       shorthand + typos folded, audit kept
     │
  2  registry.understand()     backend/nlu/registry   -> Understanding
     │
     ├─ family != data ────────────────────────────>  {"type":"chat"}   greeting, control, refusal
     │
  3  detect_reference()        backend/nlu/context    "and there?" -> what "there" means
     is_follow_up()
     │
  4  context.apply()           backend/nlu/context    SET / REPLACE / MODIFY / INHERIT / COMPARE
     │                                                 -> the merged conversation state
  5  resolve_places()          backend/pipeline       Solr, alias table, GPS fallback
     │                                                 -> lat/lon, or {"type":"need_location"}
  6  pipeline.run()            backend/pipeline       everything below, as one call
     │      plan()                  which source, what resolution, how many rows
     │      sources.fetch_for()     GFS / archive / lookback, degrading rather than failing
     │      served_fields()         drop columns the feed never sent
     │      quality.assess()        what actually came back
     │      analysis.*              reduction, chart, observations
     │      advice.evaluate()       the verdict, if this was an advice turn
     │      render.summarize()      the deterministic conclusion
     │
  7  generation.build()        backend/generation     the retrieval context, in sections
     generation.stream()                              -> {"type":"thinking"} / {"type":"delta"}
     │
  8  store.record_turn()       backend/store.py       tagged "[v4] <text>"
     store.attach_payload()
     │
     ▼
  {"type":"result", ...}       table + chart + insights + advice + plan + quality
```

**Where the conversation ends and the answer begins.** Steps 1-5 and 8 are `api/chat.py`:
they are about *this chat*. Step 6 is `pipeline.run`, which has no conversation, no socket and
no id - hand it an `Understanding` and it produces an `Answer`. That split is why the compare
view can run the same pipeline three times on one sentence without a second implementation.

### Transport: SSE over POST, not a WebSocket

One question has one answer. There is nothing to hold open between turns, so there is no
reconnect loop, no ping/pong, and no half-open socket that silently stops answering. The
streaming that mattered - the phrasing, which is the slowest step by a wide margin - is
exactly what server-sent events are for, and it survives any proxy that speaks HTTP.

Every turn ends in exactly one terminal event: `result`, `chat`, `clarify`, `need_location`
or `error`. The client's spinner is driven off that, so a stream that dies mid-turn is
reported rather than left spinning.

---

## 2b. Time expressions

Four tiers, first hit wins, and every one of them is checked by `timewindow.resolve` before it
is believed — `understood=False` means the window was a guess, and a guessed window is refused.

| tier | placed by | cost | example |
| :--- | :--- | :--- | :--- |
| rules | `src/tagger.py` tables | ~0 | `tomorrow`, `next 3 days`, `last week` |
| **duckling** | `rasa/duckling` container | ~6 ms | `last summer`, `last few days`, `so far today`, `between 6pm and 9pm`, `tonight at 6`, `at 6` |
| model | ollama, cached per phrase | ~150 ms | `prior days`, `the other day`, `couple of days back` |
| unplaceable | nothing could | — | the turn stops and says so, rather than answering next week |

```bash
docker run -d --name duckling-service -p 8008:8000 rasa/duckling:latest
```

`DUCKLING_LOCALE` must stay `en_IN` (or `en_GB`). `en_US` reads `11/06/2026` as 6 November and
nothing downstream can tell; the startup probe fails loudly on exactly that. `DUCKLING_TZ` must
be pinned too — the container defaults to US Pacific, which puts every window 12.5 hours out.

The container is optional. Stopped, `duckling.canonical` returns `""` and the other three tiers
carry the turn — verified by running the suites with it down.

Duckling is asked **concurrently with the intent model** (`backend/api/chat.py`): it needs only
the raw sentence, so its round trip overlaps the sklearn predict instead of following it.
Measured 15.0 ms serial → 9.3 ms together.

---

## 3. The models

Two, answering the same contract, so they can be compared on one sentence.

| | **Model 2 (`v4`) — served** | Model 3 (`llm`) |
| :--- | :--- | :--- |
| Where | `models/nlu_v4.joblib`, 46.6 MB | hosted, via `API_KEY` |
| Intents | 16, incl. chat / control / declined | 16, from the prompt |
| Variables | 10, multi-label | 10 |
| Activities | 12, for the advice engine | 12 |
| Presentation | derived, not predicted | derived |
| Latency | single-digit ms | a network round trip |
| Code | `src/v4/` | `backend/nlu/llm.py` |
| Reachable from | every endpoint | `POST /api/compare` only |

`Understanding` (`backend/nlu/registry.py`) is the common denominator, so the pipeline genuinely
cannot tell which model answered.

Model 1 (`v3`) is **retired**: 6 coarse intents, 13 variables, and three heads that predicted
`detail`, `chart` and `insights`. All three are now derived - answer width off the words, the
chart off the wording, every applicable insight with a cap. `src/v3/`, `models/nlu_v3.joblib`,
`models/metrics_v3.json` and `tests/test_model.py` are gone; the registry is still a registry so
that a v5 is a `MODELS` entry and a loader branch, not a rewrite of every caller.

Model 3 loads no bundle and is not in the registry - it is a client that coerces whatever the
hosted model emits onto the same enums. A label it invents is dropped; a location span that is
not a verbatim substring of the question is dropped (Rule 4.1).

### Model 2's heads

```text
                       raw text
                          │
                   clean_text()  (src/nlu.py)
                          │
                build_vectorizer()  — word + char n-grams
                          │
        ┌─────────────────┴──────────────────┐
        │                                    │
  feature matrix (shared)               raw text
        │                                    │
  ┌─────┼──────┬──────────┐                  ├──────────────┐
  ▼     ▼      ▼          ▼                  ▼              ▼
intent vars activity aggregation      SpanTagger (BIO)  entity gazetteer
                                       locations, times   crop, material, …
```

Everything else is derived rather than predicted: `weather_intent` from the resolved window,
`action` from the intent, `sub_activity` from the entities, `detail` from the wording. A head
that can be replaced by four lines of arithmetic is a head that can be wrong.

---

## 4. File map

### Backend

| File | Role |
| :--- | :--- |
| `backend/config.py` | Every setting, `.env` loaded once. Nothing else calls `os.getenv`. |
| `backend/store.py` | SQLite turn/feedback log, review queue, retraining export |
| `backend/api/__init__.py` | The FastAPI app: middleware, routers, startup |
| `backend/api/chat.py` | `POST /api/chat` — one turn, streamed |
| `backend/api/compare.py` | `POST /api/compare` — every model on one sentence |
| `backend/api/meta.py` | health, models, labels, location autocomplete |
| `backend/api/history.py` | past conversations, replayed from the log |
| `backend/api/feedback.py` | thumbs, corrections, the review queue |
| `backend/api/deps.py` | the three process-lifetime singletons |
| `backend/nlu/registry.py` | `Understanding`, `Registry`, the trained bundles |
| `backend/nlu/context.py` | the context engine — what the conversation remembers |
| `backend/nlu/llm.py` | Model 3: the hosted model, coerced onto the contract |
| `backend/pipeline/__init__.py` | `run()` and `Answer` — the single answer path |
| `backend/pipeline/timewindow.py` | **the only calendar**: wording → window, and row selection |
| `backend/pipeline/plan.py` | source capability, resolution ladder, row budget |
| `backend/pipeline/windows.py` | **when** a condition holds - the runs of readings, not the total |
| `backend/pipeline/sources.py` | GFS, historical, Zarr point/bulk; canonical field names |
| `backend/pipeline/places.py` | Solr resolution, aliases, relative places, ambiguity |
| `backend/pipeline/quality.py` | what actually came back, before anything is computed |
| `backend/pipeline/analysis.py` | reduction, chart, observations (`Note.kind`) |
| `backend/pipeline/advice.py` | 11 activity rules → YES / NO / CAUTION, with a window to act in |
| `backend/pipeline/render.py` | table, deterministic summary, the field vocabulary |
| `backend/generation/context.py` | retrieval: the labelled sections the model may use |
| `backend/generation/prompts.py` | prompt blocks, composed by what the turn has |
| `backend/generation/llm.py` | the local model, and the fallbacks when it is not there |

### Models and data

| File | Role |
| :--- | :--- |
| `src/v4/model.py` | **Model 2.** `V4Model`, `train()`, `evaluate()`, `--export` |
| `src/v4/schema.py` | the v4 taxonomy: intents, variables, activities, resolutions, fields |
| `src/v4/dataset.py` | builds `data/v4_dataset.csv` |
| `src/v4/entities.py` | the entity gazetteer — crop, material, vehicle, garment |
| `src/v2/schema.py` | legacy slot enums, still read by the dataset chain |
| `src/v2/dataset.py` | conversation generator, upstream of the datasets |
| `src/nlu.py` | `build_vectorizer()`, `clean_text()` — the shared encoder |
| `src/tagger.py` | `SpanTagger` (BIO), `normalize_time()` |
| `src/normalize.py` | pre-model text normalizer |
| `src/schema.py` | `ConversationState`, `Operation`, `Reference` |
| `src/build_dataset.py` | template generator: balanced cells, misspellings, fillers |
| `src/fetch_locations.py` | samples real place names into `data/locations.csv` |

| Data | Tracked | Role |
| :--- | :--- | :--- |
| `data/v4_dataset.csv` | no | Model 2's training data, 23,968 rows — regenerable |
| `v4_dataset.csv` (root) | **yes** | 41 hand-written advice seeds. **Unrecoverable if lost.** |
| `data/eval_v4.csv` | yes | hand-written v4 evaluation set, 183 rows |
| `data/eval_v4_hard.csv` | yes | 63 hard cases: ambiguity, implicit advice, code-mixing |
| `data/v3_dataset.csv` | yes | the retired model's training data — kept as the only shipped multi-turn fixture, replayed by `test_conversations.py` |
| `data/eval_manual.csv` | yes | hand-written eval, 235 rows (219 en + 16 mixed) |
| `data/intents.csv` | yes | hand-written seed, head of the generation chain |
| `data/locations.csv` | yes | 1,166 real place names, 86% inside India |
| `data/location_aliases.json` | yes | nicknames and spellings Solr will not match |
| `data/conversations.db` | no | runtime chat log; export labels, do not commit |

### Frontend

| File | Role |
| :--- | :--- |
| `frontend/lib/use-chat.ts` | the transport: POST + SSE reader, one conversation |
| `frontend/lib/types.ts` | the wire format, in one place |
| `frontend/lib/utils.ts` | `apiUrl()` — the single definition of where the backend is |
| `frontend/app/page.tsx` | the chat page; holds the model id and the chat id |
| `frontend/components/composer.tsx` | input + model pill, fed by `/api/models` |
| `frontend/components/messages.tsx` | transcript, ratings, correction entry point |
| `frontend/components/compare.tsx` | the three-model comparison |
| `frontend/components/correction.tsx` | a thumbs-down turned into a labelled training row |
| `frontend/components/result-table.tsx` | the table |
| `frontend/components/result-chart.tsx` | the chart the pipeline chose |
| `frontend/components/chat-history.tsx` | past conversations |
| `frontend/components/health-badge.tsx` | `/api/health`, polled |

### Tests

Every non-trivial module carries a runnable self-check. `python -m backend.pipeline.plan`
asserts the routing table; `python -m backend.pipeline.timewindow` asserts the calendar. They
run in under a second and need no network.

| File | Checks |
| :--- | :--- |
| `tests/test_conversations.py` | 150 multi-turn conversations replayed through the real context engine |
| `tests/test_dataset.py` | eval-set coverage, location vocabulary |
| `tests/eval_v4.py` | Model 2 against the hand-written eval set |

---

## 5. The training chain

Model 2 trains from `data/v4_dataset.csv`, built from two hand-written seeds plus generated
combinations:

```text
data/intents.csv          (hand-written seed, tracked)
data/locations.csv        (sampled vocabulary, tracked)
v4_dataset.csv            (41 hand-written advice seeds, tracked, UNRECOVERABLE)
        │
        │  python -m src.v4.dataset --build
        ▼
data/v4_dataset.csv       23,968 rows          ← generated, gitignored
        │
        │  python -m src.v4.model --export
        ▼
models/nlu_v4.joblib + models/metrics_v4.json
```

`src/v4/dataset.py::build()` refuses to overwrite the root `v4_dataset.csv` - the two files
have the same name and one of them cannot be regenerated.

The legacy chain ran from `data/intents.csv` through `src/build_dataset.py` and
`src/v2/dataset.py`; its output `data/v3_dataset.csv` is kept because it is the only shipped
file with multi-turn conversations and gold context slots in it.

**To fold real usage back in**: `python -m backend.store --export data/from_users.csv`. Human
labels outrank the model's (Rule 8.5).

---

## 6. Measured performance

From `models/metrics_v4.json`, written at export time.

**Model 2 (v4)** — 18,518 training rows.

| Split | rows | intent | weather_intent | vars | activity | aggregation | locs | times | **everything** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| test (generated) | 3,408 | .997 | .986 | .928 | .998 | .945 | .968 | .925 | **.791** |
| eval | 2,042 | .999 | .982 | .930 | 1.00 | .946 | .929 | .919 | **.762** |
| implicit advice | 533 | 1.00 | .993 | .996 | 1.00 | 1.00 | .989 | .931 | **.917** |
| confusion pairs | 238 | 1.00 | .996 | .996 | 1.00 | 1.00 | .983 | .954 | **.937** |

`everything` — every target right on the same turn — is the honest number. Multi-turn context
replayed through the real engine (`test_conversations.py`, 150 conversations / 437 turns, Model
2): operation 99.8%, locations 89.5%, times 99.8%.

`times` is the weakest head and is where the next round of training data should go.
`variables` at .928 is second.

---

## 7. Design decisions worth knowing

**One calendar.** `pipeline/timewindow.py` is the only module that knows what "tomorrow"
means. There used to be three - a window resolver, an absolute-date parser and a row selector -
each with its own `PART_OF_DAY`, `DAY_OFFSET` and `WEEKDAYS` tables. They disagreed: "monday"
meant *next* Monday to one and *any* Monday to another; "this weekend" spanned one weekend or
two depending on which asked; "11 jun 2026" resolved to a single day in one and to the whole
of 2026 in the other.

**One reader of a column.** `quality.values()` filters `None`, `""`, `"NA"`, NaN and the
sentinels `{-999, -9999, 999, 9999}`. Six places used to read columns with `is not None`
instead, so a `-999` sentinel entered means and won `MIN` comparisons - `served_fields` counted
the column as served and the aggregator then averaged the sentinel in.

**Presentation is downstream of the model, never inside it.** Model 1 predicts `detail`,
`chart` and `insights`; Model 2 derives them. Either way `pipeline` applies them, so neither
model needs to know a table exists.

**A verdict is never computed from thin data.** `advice.evaluate` asks `quality.assess` first
and returns `UNKNOWN` rather than a verdict when the fields its rule reads are mostly empty. A
confident answer computed from two readings out of thirty looks exactly like a real one, which
is what makes it dangerous.

**Failure never depends on another thing working.** Every trouble line in
`generation/llm.py::TROUBLE_LINES` is a fixed sentence. The local model is asked to re-say it,
and the result is thrown away if it invented a figure or a weather word its input did not have.

**One gate decides whether the wording is shown at all.** `generation.usable()` drops a reply
that echoes the conclusion, narrates the machinery ("the data shows"), states a figure it was
not given, or reverses the verdict it was handed. Anything it drops falls back to the
deterministic sentence, which is what the reader would have got anyway.

The figure check allows rounding, because the prompt asks for it - "about 2mm" for 1.93mm is
how people speak, and a guard that rejected it would defeat the layer in the name of protecting
it. The tolerance comes from the precision the reply was written at: "2" may stand for anything
within 0.5, "2.4" only for something within 0.05. So 1.93 may be called "2" and may not be
called "2.4". A 1b model handed "12.5mm against 9.9mm" answered "about 3mm of rain is
expected"; nothing rounds to that.

The reversal check exists because the advice path is not a style problem. Asked whether clothes
would dry, a 1b handed "No - 2.4mm expected from 06:00" answered "making it ideal for drying
clothes". Someone hangs washing out on that.

**A misconfigured wording layer is loud.** Every failure inside a turn falls back silently,
which is right mid-turn and wrong for a deployment - a one-character typo in `OLLAMA_MODEL`
degraded every answer and nothing said so. `generation.probe()` runs at startup, prints what is
wrong, and reports it on `/api/health`. The `think` flag is also negotiated rather than
assumed: models without a reasoning mode reject it with a 400, which used to turn the whole
layer off.

**An activity question is a question about *when*.** Rain does not fall for a whole day; it
falls between two and four. `pipeline/windows.py` finds the runs of readings during which an
activity's conditions actually hold, and the verdict is about the best of those runs - so the
answer can be "not at two, but you have until noon". Collapsing the period into an accumulated
total, which is what every rule used to do, was wrong three ways: 8mm in one storm and 8mm of
all-day drizzle are the same number and opposite answers; a total only grows, so the same
weather scored worse the longer a period you asked about, and "should I spray today" disagreed
with "should I spray this week" about today; and a total can only ever say no - it cannot
suggest a time.

Rules state what *one reading* has to look like (`below`, `between`, `every`), and a missing
reading is never a suitable one - an unknown hour in the middle of a spraying window is the
hour you would least want to bet on. Spraying additionally needs the rain to stay away
afterwards, measured against the job's length rather than the window's: a ten-hour clear spell
against a two-hour job needs six clear hours from whenever you start, not fourteen. The
forecast simply ending is not rain, so a window that reaches the horizon is accepted and says
that it could not be confirmed.

The activities that are genuinely about state - irrigating, sowing, fertilising, what to wear -
still accumulate, because how much water arrives *is* their question. Each one says what it
summed over.

**Adding up is a reduction, and reductions are asked for.** `render.summary_stat` sums only
when `aggregation` is `SUM`; under `RAW` it means. Rainfall used to be totalled whenever the
field was additive, so "rain this week" answered with a week's accumulation presented as
though it were the rainfall, and the same weather scored higher the longer a window you asked
about. `analysis.confirm_aggregation` is symmetric: it drops a reduction the wording never
asked for, and promotes one the wording plainly states.

**Both models commit rather than ask.** `NEVER_ASKS` covers every trained model
(MODEL_RULES Rule 1.1), so an ambiguous place resolves to the ranked best and the answer says
which one it took. The only `clarify` left is a query the planner genuinely cannot serve - too
far ahead, too many rows, or an archive that is unreachable.

---

## 8. Operations

```bash
# run it — backend :8787, frontend :3001
./scripts/run_app.sh

# every module's self-check (no network needed for most)
python -m backend.pipeline.timewindow      # the calendar
python -m backend.pipeline.plan            # source routing and the row budget
python -m backend.pipeline.quality         # missing-data detection
python -m backend.pipeline.windows         # runs, spacing, labels
python -m backend.pipeline.advice          # timing beats totals; every verdict flips
python -m backend.pipeline.render          # tables and the deterministic summary
python -m backend.pipeline.analysis        # reductions, charts, observations
python -m backend.generation.context       # what the model is allowed to know
python -m backend.generation.prompts       # every prompt shape
python -m backend.nlu.registry             # the bundle answers
python -m backend.nlu.context              # the four-turn conversation
python -m backend.api.chat                 # one turn, end to end
python -m backend.api.compare              # every model on one sentence

# the suites
python tests/test_conversations.py               # multi-turn context, 150 conversations
python tests/test_dataset.py                     # data coverage
python tests/eval_v4.py                          # Model 2 against the hand-written eval

# retrain
python -m src.v4.model --export            # Model 2, from data/v4_dataset.csv

# ask one question without the server
python -m src.v4.model "rain and temperature in Guntur tomorrow"
```

`/api/health` reports whether the bundle is present and what is configured (no secret values).
`/api/models` serves each model's own exported metrics. `/api/labels?model=v4` serves the label
sets the correction form offers - per model, because they no longer agree.
