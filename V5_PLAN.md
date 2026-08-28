# v5 - routed pipelines

One `run()` answers every question today. Forecast, history and advice want different fetches,
different reductions and different screens, so they become different routes over the *same*
stages. Nothing here is a new model: every routing decision is derived from slots v4 already
predicts.

```
text
 -> NLU              (unchanged contract; + weather_intent, + venue - both derived)
 -> Router           forecast | historical | activity | compare
 -> Profile          the per-route / per-activity config table
 -> Params           which API - how many places - which aggregation - which window
 -> Toolbox          named fetchers, called 0..n times by the route
 -> Stages           quality -> analysis -> advice          (shared, unchanged)
 -> Presenter        Answer -> ordered UI blocks
 -> Generator        conversational text over those blocks
```

---

## 0. What is already right, and stays

| Thing | Where | Verdict |
|---|---|---|
| `Understanding` as the model-agnostic contract | `backend/nlu/registry.py` | keep, add 2 derived fields |
| source / resolution / row-budget / verdict | `backend/pipeline/plan.py` | keep - this **is** the parameter model for time |
| per-activity rules, thresholds, default windows | `backend/pipeline/advice.py` | keep, split `OUTDOOR_ACTIVITY` |
| quality, windows, timewindow, analysis, render | `backend/pipeline/*` | keep, called by every route |
| generator as a re-sayer that cannot invent figures | `backend/generation/*` | keep, re-source its context |

The work is routing, config and the render contract. Not a rewrite.

---

## 1. Intent classification - derived, not a new head

No classifier is trained or retrained. `src/v4/schema.py` already has `weather_intent_for()`;
`backend/nlu/llm.py` already calls it and drops the result on the floor.

```python
# backend/pipeline/routes.py
def pick(u: Understanding, window: Window) -> str:
    if u.activity != "NONE":                       return ACTIVITY     # intent ADVICE
    if u.action == "COMPARE" and len(u.locations) > 1: return COMPARE
    if window.start.date() < date.today():          return HISTORICAL   # weather_intent HISTORICAL
    return FORECAST                                                     # CURRENT | TOMORROW | FORECAST
```

Four, not three: COMPARE changes the *fetch fan-out* (N places, bulk source, comparison
insight leads), which is the "how many places" axis you named. That earns a route; if after
step 3 it reads as forecast-with-a-flag, fold it back.

**Change to `Understanding`** (`backend/nlu/registry.py`), ~20 lines, no behaviour change:

```python
weather_intent: str = "FORECAST"   # NONE|CURRENT|TOMORROW|FORECAST|HISTORICAL - from times_normalized
venue: str = "outdoor"             # outdoor|indoor|mixed - from the gazetteer, below
```

Both derived at read time exactly like `sub_activity` and `Action` already are, so the two can
never disagree with the slots they come from.

---

## 2. Profiles - the config table per intent, per activity

`backend/pipeline/profiles.py` (new). One frozen dataclass, one dict.

```python
@dataclass(frozen=True)
class Profile:
    window: str            # what "no time named" means here
    fields: tuple          # columns this decision needs, whether or not the user said them
    resolution: str        # HOURLY | DAILY | AUTO  (AUTO = let the planner decide)
    aggregation: str       # default reduction when none is spoken
    blocks: tuple          # which UI blocks this route emits, in order
    venue: str = "outdoor"
    min_hours: float = 0   # a usable stretch, for timed activities
    settle_hours: float = 0
    thresholds: dict = field(default_factory=dict)   # overrides on advice.py constants
```

**Rule for adding a key:** a key exists only when two profiles disagree on it. Same test
`src/v4/schema.py` already applies to `Activity` labels - otherwise this rots into config for
a value that never changes.

### Data routes

| route | window | resolution | aggregation | blocks |
|---|---|---|---|---|
| FORECAST, end <= now+24h | `today` | HOURLY | RAW | stat, chart(line), table |
| FORECAST, longer | `next 7 days` | DAILY | RAW | chart(bar), table |
| HISTORICAL, span <= 7d | as asked | DAILY | RAW | stat, chart(line), table |
| HISTORICAL, span > 7d | as asked | AUTO (planner coarsens) | **SUM** for RAIN/SUNSD, **AVG** otherwise | stat, chart(bar), table |
| COMPARE | as asked | AUTO | as asked | stat-per-place, chart(grouped_bar), table |

Two behaviour changes, both deliberate:

- **Historical stops defaulting to RAW.** "Rainfall in Guntur last June" currently returns 30
  raw rows - a table, not an answer. Over 7 days with no reduction spoken, reduce and record
  it in `assumed` so the reply says so.
- **Historical asks for the `Normal_*` columns.** `render.LABELS` already names them; nothing
  requests them. Fetching them makes "wetter than usual" answerable, which today it is not.

### Activity routes - the indoor / outdoor / sport split

Keyed `(activity, sub_activity)` with fallback to `(activity, "")`. `sub_activity` is already
extracted from the gazetteer, so this costs a table row per sport, never a retrain.

| key | venue | window | min_h | settle | fields | why it differs |
|---|---|---|---|---|---|---|
| OUTDOOR_ACTIVITY / walk | outdoor | today | 1 | 0 | Rainfall, Tmax | tolerant, short |
| OUTDOOR_ACTIVITY / cricket | outdoor | next 2 days | 3 | 0 | Rainfall, Tmax, RH, Wind_Speed | pitch: needs 12h dry *before*, not just during |
| OUTDOOR_ACTIVITY / football | outdoor | today | 2 | 0 | Rainfall, Tmax | plays through light rain: DAMP, not WET |
| OUTDOOR_ACTIVITY / marathon | outdoor | today | 3 | 0 | Tmax, RH, Rainfall | heat + humidity dominate, rain barely matters |
| OUTDOOR_ACTIVITY / wedding | outdoor | that day | 6 | 0 | Rainfall, Wind_Speed | long window, wind ruins a shamiana |
| OUTDOOR_ACTIVITY / construction | outdoor | next 3 days | 6 | 8 | Rainfall, RH, Tmax | concrete has to cure |
| OUTDOOR_ACTIVITY / gardening | outdoor | today | 2 | 0 | Rainfall, Tmax | |
| OUTDOOR_ACTIVITY / badminton, gym, yoga, squash | **indoor** | today | - | - | Rainfall | see below |

**Indoor is a behaviour change, not a threshold change.** An indoor activity is answered from
the conditions of *getting there*, not of doing it: run the TRAVEL rule, and say plainly that
the weather does not reach the activity itself. Five lines in the route:

```python
if profile.venue == "indoor":
    verdict = advice.evaluate("TRAVEL", rows, sub_activity="", hourly=hourly)
    verdict.headline = f"{sub} is indoors - {verdict.headline.lower()} getting there"
```

**Where `venue` comes from** - `src/v4/schema.py`, alongside `ENTITY_VOCAB`:

```python
INDOOR_SPORTS = {"badminton", "table tennis", "squash", "chess", "carrom", "billiards",
                 "bowling", "gym", "yoga"}
# the word wins over the sport: "outdoor badminton court" -> outdoor, "indoor cricket" -> indoor
VENUE_WORDS = {"indoor": "indoor", "indoors": "indoor", "outdoor": "outdoor",
               "outdoors": "outdoor", "open ground": "outdoor", "terrace": "outdoor"}
```

Closed vocabulary, so a lookup - the same reasoning that already keeps sports and crops out of
the classifier. A lookup cannot be 87% right about whether badminton is indoors.

Farming activities keep their existing profiles verbatim; `advice.TIMED` / `STATE_NEEDS` /
`DEFAULT_WINDOW` become the seed data for this table rather than a second copy of it.

---

## 3. Params - the four decisions, in one object with its reasons

`backend/pipeline/params.py` (new).

```python
@dataclass
class Params:
    source: Source          # which API      <- planner
    resolution: Resolution  # ^
    start: str; end: str; label: str
    places: list[dict]      # how many places
    aggregation: str        # which action
    fields: list[str]
    why: dict               # decision -> one line, for the audit strip
```

`resolve(understanding, profile, places, now)` in order:

1. **window** - the user's span if they named one, else `profile.window`.
2. **aggregation** - `analysis.confirm_aggregation(text, slot)` if spoken, else
   `profile.aggregation`. Assumed reductions land in `assumed`.
3. **fields** - `union(v4_fields_for(variables, detail), profile.fields)`. The profile adds
   what the *decision* reads even when unnamed: spray needs wind whether or not "wind" was said.
   This is a bug fix - today the union happens nowhere and a rule can be handed a column it needs
   and did not get.
4. **places** - as resolved; COMPARE caps at 3.
5. **source + resolution** - `planner.plan(...)` unchanged, except `profile.resolution` may
   force HOURLY.

Each step writes one line into `why`. That goes into `stages["params"]` and out on the wire -
the debug strip already renders `stages`, so "why hourly" becomes readable without a debugger.

This is mostly a **rehousing** of decisions scattered across `run()`, `plan()` and
`confirm_aggregation()`. The win is one object, one audit trail, one place to change.

---

## 4. Toolbox - named fetchers a route can call more than once

`backend/pipeline/toolbox.py` (new), a thin naming layer over `sources.fetch_for`:

```python
TOOLS = {"forecast_daily", "forecast_hourly", "history_recent",
         "archive_point", "archive_bulk", "archive_normals", "area_agg"}
```

The new capability is calling **two**:

- **HISTORICAL** -> `archive_point` + `archive_normals`, giving "about a quarter wetter than a
  normal June". Not possible today.
- **ACTIVITY over a long window** -> `forecast_hourly` (first 24h, the only place "from 2pm"
  can come from) stitched onto `forecast_daily` (the rest). Today the planner picks one, so a
  HARVEST question either loses the *when* or loses the *how long*.

Every call is appended to `stages["tools"]` with its args, ms and row count.

### On the model choosing tools

You asked for the model to trigger the tools. Straight tool-choice over the primary source is
where this gets expensive: a model that picks `archive_bulk` for ten years across three places
orders a million rows and there is nothing downstream to stop it - the planner's row budget is
the only thing that does. So, split:

- **Primary source: rules.** `planner.plan()` keeps it. Cost-bounded by construction.
- **Enrichment tools: the model's, from a bounded menu.** Normals, a second window, a second
  place, hourly-on-top-of-daily. Each is one extra call with a known cost, so a wrong pick is
  slow, never unbounded.

The seam is `Route.tools(params) -> list[str]`. Swap the rule body for a model call and
everything downstream holds - so full model-driven selection stays one function away behind
`TOOL_CHOICE = "rules" | "model"` in config if you want it later.

---

## 5. Presenter - the UI contract

Today the payload ships `table`, `chart`, `insights`, `series` and the frontend infers a layout.
Replace with an ordered block list; `backend/pipeline/ui.py` (new) builds it.

```json
"ui": [
  {"type":"verdict","tone":"yes|no|caution","headline":"...","reasons":[...],"window":"..."},
  {"type":"stat","label":"Rainfall","value":12.5,"unit":"mm","caption":"total, next 3 days"},
  {"type":"chart","kind":"line|bar|grouped_bar","field":"Rainfall","unit":"mm","series":[...]},
  {"type":"table","columns":[...],"rows":[...],"collapsed":true},
  {"type":"notes","items":["Served from the archive."]},
  {"type":"suggest","items":["compare with Vizag","next 7 days"]}
]
```

Which blocks appear is `profile.blocks` - data, not code. One rule survives contact:

> A block is emitted only when the data behind it exists.

`ui.blocks()` drops any chart whose series has under two points and any table with one row.
That is the existing `simple` / `build_chart` guard, moved into one function instead of two
conditionals inside `run()`.

Frontend: one `frontend/components/result-blocks.tsx` switching on `type`, with the existing
`result-table.tsx` and `result-chart.tsx` as two of the branches. `messages.tsx` stops branching.

---

## 6. Generator

`generation/context.build` takes the **blocks** instead of a flat dict. Its labelled sections
already map onto block types one-to-one (Comparison, Figure asked for, Decision, Best window,
Range, Notable, Figures), so this is a re-source, not a rewrite. `prompts.py` is unchanged
except one line in `GROUNDING`:

> Refer to the chart once if there is one - "it tails off after four" - and never mention a
> table or chart that is not there.

---

## 7. Files

| File | New? | Lines |
|---|---|---|
| `backend/pipeline/routes.py` | new | ~150 |
| `backend/pipeline/profiles.py` | new | ~120 |
| `backend/pipeline/params.py` | new | ~80 |
| `backend/pipeline/toolbox.py` | new | ~60 |
| `backend/pipeline/ui.py` | new | ~120 |
| `backend/pipeline/__init__.py` | shrinks | 358 -> ~120 |
| `backend/nlu/registry.py` | +2 derived fields | +20 |
| `src/v4/schema.py` | venue gazetteer | +25 |
| `backend/pipeline/advice.py` | reads thresholds from profile | ~20 changed |
| `frontend/components/result-blocks.tsx` | new | ~60 |
| `tests/test_routes.py` | new | ~40 |

---

## 8. Order - each step ships alone

1. `Understanding.weather_intent` + `venue`. No behaviour change; existing suite must stay green.
2. `profiles.py` + `params.py`; `run()` reads them but keeps its single body. Only two payloads
   move: historical aggregation, activity field union.
3. `routes.py` - split `run()` into four bodies over the shared stages. Pure refactor, payloads
   byte-identical for everything but step 2's two cases.
4. `toolbox.py` + normals + the hourly/daily stitch. New capability, new tests.
5. `ui.py` blocks shipped **alongside** the old `table`/`chart` keys; frontend behind a flag.
6. Activity venue and sport profiles.
7. Drop the old payload keys.

**Test:** one `tests/test_routes.py`, an assert table of `question -> (route, source, resolution,
aggregation, block types)` over ~20 questions. Catches every routing regression in 40 lines. No
framework, matching the existing `python tests/test_pipeline_units.py` style.

---

## 9. What I would cut if it does not earn itself

- **Four routes.** If step 3 leaves them 90% identical, collapse to one body with a profile.
  The divergence has to be real - normals, best-window, fan-out - not aesthetic.
- **`suggest` blocks.** Speculative until someone asks for follow-up chips. Skipped in step 5;
  add when the UI wants them.
- **`TOOL_CHOICE = "model"`.** Seam only. Build the rule path; add the model path when a real
  question exists that the rules route wrongly.
