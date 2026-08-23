# WeatherBot — rules for anyone (or anything) editing this repo

## Layout

```
backend/config.py     every setting. Nothing else calls os.getenv.
backend/api/          HTTP only: routes, SSE framing, the turn log
backend/nlu/          text -> Understanding; what the conversation remembers
backend/pipeline/     Understanding -> Answer. No transport, no chat id.
backend/generation/   the Answer, said in words. Generates nothing it was not handed.
src/v4/               Model 2 — the default served model
src/v3/, src/v2/      Model 1 and its slot enums
```

Put a change in the layer that owns the thing it changes. A fix to how "tomorrow" is read goes
in `backend/pipeline/timewindow.py` — not in the caller that noticed.

## Hard rules

1. **The model extracts; it does not decide.** No coordinate, timestamp or weather row ever
   reaches a model. Source selection, row budgets, thresholds and units are deterministic.
2. **One implementation.** Before adding a helper, grep for it. This repo has already had
   three calendars, two location resolvers and six ways of reading a column, and every one of
   those pairs disagreed with itself in production.
3. **Read a column through `pipeline.quality.values()`.** It filters `None`, `""`, `"NA"`,
   NaN and the sentinels `{-999, -9999, 999, 9999}`. `is not None` is not good enough — a
   `-999` reaching a mean is a silently wrong answer.
4. **Spans stay verbatim** (Rule 4.1/5.1). A location or time span must be a substring of what
   the user typed. A span a model invented is worse than no span.
5. **Never state a figure the pipeline did not compute, and never reverse its verdict.** The
   generation layer is handed a conclusion and labelled retrieved sections and may use only
   those. Enforced by `generation.usable()`, which drops an echo, a scaffolding phrase, an
   ungrounded figure or a reversed verdict; the deterministic sentence goes out instead. Add a
   new failure mode there, not at a call site - every caller goes through that one gate.
6. **An activity question is about *when*, not whether.** Use `pipeline/windows.py`: state
   what one reading must look like and let it find the runs. Never threshold an accumulated
   total to decide whether something can be done - 8mm in one storm and 8mm of all-day drizzle
   are the same number and opposite answers. Only the state activities (irrigate, sow,
   fertilise, clothing) accumulate, and each says what it summed over.
7. **Never add up unless asked.** Summing is a reduction and a reduction is reported only
   when the wording asks for one (`render.summary_stat`, `analysis.confirm_aggregation`). A
   week of rainfall totalled and labelled "rainfall" is a bigger number for the same weather
   the longer a window you ask about.
8. **Never decide from thin data.** `advice.evaluate` asks `quality.assess` first and returns
   `UNKNOWN` rather than a verdict. A verdict from two readings out of thirty looks exactly
   like a real one.
9. **Never ask which city they meant.** Both trained models commit and report the assumption
   (Rule 1.1). The only `clarify` is a query the planner genuinely cannot serve.
10. **Every non-trivial module gets a `demo()` self-check** guarded by `__main__`, using plain
   `assert`. No test framework. It must run without the network where it can.
11. **Human labels outrank the model's** (Rule 8.5) when folding feedback into training data.

## Model contracts

- Model 2 (`v4`) is the default. Its taxonomy is [../V4_PLAN.md](../V4_PLAN.md) §2 —
  16 intents, 10 variables, 12 activities. Enums live in `src/v4/schema.py`.
- Model 1 (`v3`) is still served and selectable. Its rules book is
  [../MODEL_RULES.md](../MODEL_RULES.md); its floors are asserted by `tests/test_model.py`.
- `Understanding` (`backend/nlu/registry.py`) is the only shape the pipeline reads. Adding a
  model means producing one of those, not touching anything downstream.

Everything derivable is derived, not predicted: `weather_intent` from the window, `action`
from the intent, `sub_activity` from the entities, `detail` from the wording. A head that four
lines of arithmetic replace is a head that can be wrong.

## Before you commit

```bash
python -m backend.api.chat && python -m backend.api.compare      # end to end
python tests/test_model.py && python tests/test_conversations.py  # the suites
cd frontend && npx tsc --noEmit && npm run build
```

Do not commit: `models/*.joblib`, `data/conversations*.db*`, `data/v4_dataset.csv`, or any
`*.postman_environment.json` — that last one carries live API keys.
