# WeatherSnap

A weather assistant for India that reads plain, misspelled, code-mixed questions and answers
them with real forecast data — and, when asked whether to *do* something, decides and shows
the numbers it decided from.

```text
"will it rain in Nokha tommorrow?"            -> a forecast, a table, a chart
"should i spray pesticide in Guntur tomorrow" -> "Good window - wind 2.1m/s, 0.4mm rain"
"compare rain in Guntur and Vizag this week"  -> both, side by side, with the gap named
"what about there next week?"                 -> the same question, the remembered place
"hey there"                                   -> a greeting. No forecast, no "which city?"
```

Nothing in an answer is generated. A local model does the wording; every figure in it was
computed by rules from data that was actually fetched.

- [ARCHITECTURE.md](ARCHITECTURE.md) — the layers, the request path, the file map, the numbers
- [MODEL_RULES.md](MODEL_RULES.md) — the taxonomy and the annotation rules
- [BACKLOG.md](BACKLOG.md) — known defects, with the evidence and the check command
- [V4_PLAN.md](V4_PLAN.md) — the v4 label contract and the data-source routing table

---

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env                     # works as-is; fill in the secrets you have
cp frontend/.env.example frontend/.env.local

python -m src.v4.model --export          # build the model bundle (~2 min)
ollama pull qwen3:1.7b                   # optional: the local model that words replies

./scripts/run_app.sh                     # backend :8787, frontend :3001
```

Without Ollama the answers are the rule-built sentences — correct, just blunter. Without
`API_KEY` the compare view shows two columns instead of three. Without `ZARR_API_KEY` dates
older than a week are refused up front rather than after a timeout.

---

## What it is made of

```text
backend/
  config.py      every setting, .env loaded once
  api/           HTTP: chat (SSE), compare, health, models, chats, feedback
  nlu/           text -> Understanding; time expressions; what the conversation remembers
  pipeline/      places -> plan -> fetch -> quality -> analysis -> advice -> table
  generation/    retrieval, prompts, the local model that words it
  store.py       SQLite turn log, feedback, the retraining export
src/
  v4/            Model 2 - the served model: 16 intents, 10 variables, 12 activities
  v2/            legacy slot enums and the conversation generator, upstream of the datasets
  nlu.py tagger.py normalize.py schema.py build_dataset.py
frontend/        Next.js chat UI
data/            seeds, evaluation sets, the place vocabulary
models/          the bundles (gitignored) + exported metrics
```

Every non-trivial module has a self-check you can run on its own:

```bash
python -m backend.pipeline.timewindow    # the calendar
python -m backend.pipeline.advice        # every verdict flips at its threshold
python -m backend.api.chat               # one turn, end to end
```

---

## The API

Plain HTTP. No WebSocket — one question has one answer, so there is nothing to hold open
between turns, and the one thing worth streaming (the wording, which is the slow step) is what
server-sent events are for.

| Route | What it does |
| :--- | :--- |
| `POST /api/chat` | one turn, streamed back as SSE |
| `POST /api/chat/reset` | forget this chat's slots, hand back a fresh id |
| `POST /api/compare` | the same sentence through every model, streamed as each finishes |
| `GET /api/health` | which bundles are present, what is configured |
| `GET /api/models` | every served model and its exported metrics |
| `GET /api/labels?model=v4` | the label sets the correction form offers |
| `GET /api/suggest?q=gun` | location autocomplete |
| `GET /api/chats`, `GET /api/chats/{id}` | past conversations, replayed as they were answered |
| `POST /api/feedback` | thumbs, corrections, clarify choices |
| `GET /api/review`, `GET /api/stats` | the labelling queue and usage |

```bash
curl -N -X POST localhost:8787/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"will it rain in Guntur tomorrow","chat_id":"c1"}'
```

Every turn ends in exactly one of `result`, `chat`, `clarify`, `need_location`, `error`,
preceded by `status`, `nlu`, and any number of `thinking` / `delta` pieces.

---

## The two models

| | **Model 2 (`v4`) — served** | Model 3 (`llm`) |
| :--- | :--- | :--- |
| Intents | 16, incl. chat / control / declined | 16, from the prompt |
| Variables | 10 | 10 |
| Activities | 12 | 12 |
| Presentation | derived | derived |
| Size / latency | 46.6 MB, ~5 ms | hosted, ~1 s |

Model 2 answers every turn: it tells a greeting from a question, decides an activity, and says
which source answered. Model 3 is a hosted general model with no training on this label set at
all, working from the schema in its prompt — it exists to benchmark Model 2 on the same
sentence, which is what compare mode is for.

Model 1 (`v3`) — 6 coarse intents, 13 variables, and three heads that predicted their own
presentation — is retired. What it predicted is now derived; see
[MODEL_RULES.md](MODEL_RULES.md) for the rules that outlived it.

`everything` (every target right on the same turn): **v4 .762** on its hand-written eval. Full
tables in [ARCHITECTURE.md §6](ARCHITECTURE.md).

---

## Three things it will not do

**It will not ask which city you meant.** The trained model commits to a reading and reports
what they assumed (MODEL_RULES Rule 1.1). "Angara" resolves to the ranked best match and the
answer says which one it took. The only question it will ask back is one the query planner
genuinely cannot serve — a forecast past the horizon, or an archive it cannot reach.

**It will not decide from thin data.** The advice engine checks coverage before it runs a
rule and returns "I cannot answer that from the data I got back" rather than a verdict
computed from two readings out of thirty — which would look exactly like a real one.

**It will not invent a number.** The local model that words the reply is handed the
conclusion and a labelled set of retrieved sections, and is told every figure must appear in
them. A reply that mentions weather its input did not is discarded and the fixed sentence goes
out instead.

---

## Retraining from real usage

Every turn is logged. A thumbs-down opens a correction form; the label a human picks outranks
the model's (Rule 8.5).

```bash
python -m backend.store --export data/from_users.csv
python -m src.v4.dataset --build          # folds it into the training set
python -m src.v4.model --export
python tests/eval_v4.py
```

`GET /api/review` lists what is waiting for a label — turns flagged wrong, and turns answered
from the middle confidence band that nobody rated.

---

## Security note

`WeatherSnap Environment (Production).postman_environment.json` was committed with live API
keys in it. It is now gitignored and removed from the index, **but it remains in git history**:
the Solr auth header and the Zarr API key both need rotating. `backend/config.py` also carries
a working Solr Basic-auth header as a compiled-in default — move it to `.env` and rotate it too.
