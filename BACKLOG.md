# Backlog

Everything found but not fixed, in the order it should be done. Each item is self-contained:
finish one, run its check, move to the next. Nothing here depends on anything below it.

Priorities are about **what the user gets**, not effort:

| | meaning |
|---|---|
| **P0** | Produces a confidently **wrong** answer. Worse than an error, because nobody knows. |
| **P1** | Produces **no** answer where one was possible. Visible, honest, still a failure. |
| **P2** | Produces a **worse** answer than it could — calibration, wording, coverage. |
| **P3** | Blocked, or not worth doing yet. Written down so it stops being re-discovered. |

Every claim below was measured on this machine, and the command that measured it is included.
Re-run it before starting: some of these will have moved.

---

## P0 — wrong answers

### 1. A locality resolves to a same-named village in another state, silently

```
"whats weather in madhapur"  ->  Madhapur, Nayagarh, ODISHA (20.35, 85.25)   assumed=[]
"whats weather in kondapur"  ->  Kondapur, Sangareddy (rural, ~50km out)     assumed=[]
```

Madhapur is 800km from the one the user meant, and `assumed=[]` means the UI shows no warning
at all. The index holds 43 Madhapurs; after `_dedupe` only one candidate survived, so
`_is_ambiguous` — which needs two — never fired.

**Fix.** Rank candidates by distance from context before name score: the coordinates already
shared this chat, then the places already resolved in this chat, then the state named in the
text. When the winner is in a different state from all of those, put it in
`understanding.assumed` so the answer says which Madhapur it picked.

**Files.** `backend/pipeline/places.py` (`_rank`, `_is_ambiguous`), `backend/api/chat.py` (pass context in).
**Size.** Half a day. **Check.** The two queries above must name a Telangana place, or say
which state they chose.

### 2. Non-places are tagged as locations, then looked up

From `store.failed_turns()` — the tagger hands the resolver words that are not places:

| typed | tagged as a place |
|---|---|
| `what is the rainfall fro whole day` | `whole` |
| `can you find the rainfall between may10 and…` | `may 10` |
| `what is the weather betwen 11 jan 2026 and…` | `11 jan` |
| `BHEL temeprature vs madhapur temperature` | `temeprature` |
| `which is cooler Nampally or Adyar today` | `cooler` |
| `what is my current temperature` | `my` |

Roughly half of all logged failures. Each one costs a Solr round trip and ends in a location
error for a question that had no location problem.

**Fix.** `is_probably_not_a_place()` already exists and already rejects `there`, `that place`,
`skies`. Extend it with the two closed vocabularies we own: time words (month names, `whole`,
`prior`, bare `DD mon`) and the variable words in `respond.LABELS` plus their common
misspellings. Deterministic, no model involved.

**Files.** `backend/pipeline/places.py`. **Size.** An hour. **Check.** Add each row of that table to
`locations.demo()` as an assert; `python -m backend.locations`.

### 3. Landmark names have no aliases

`BHEL` is missed entirely, and your own correction in the feedback table maps `Bhel weather` →
`madhapur`. Same class: `Gachibowli`, `Kukatpally`, `Bachupally`, `Lingampally` — all absent
from a census-village index (Hyderabad district holds **3** villages; rural districts hold
540–600).

**Fix.** An alias table beside the existing one in `locations.py`, seeded from
`store.failed_turns()` — the corrections are already labelled data for exactly this.

**Files.** `backend/pipeline/places.py`. **Size.** An hour, plus however long you spend listing
localities. **Check.** `"Bhel weather"` resolves without asking.

---

## P1 — missing answers

### 4. A missing name ends the turn, when coordinates would answer it

The weather API only ever needs coordinates — `/interpolate?lat=&lon=`. The location index is
nothing more than a name→lat/lon lookup, so **any** geocoder unblocks every name it lacks.
Beeramguda, Gachibowli and Kukatpally are all in OpenStreetMap.

**Fix.** After `resolve()` returns None and before giving up, try a forward geocoder. Photon or
Nominatim are free (mind the usage policy); MapmyIndia has the best Indian locality coverage.
Cache hits into the store so the same name is never fetched twice.

**Files.** `backend/pipeline/places.py`, `backend/pipeline/sources.py`. **Size.** A day including caching.
**Check.** `"whats weather in beramguda"` answers without needing shared coordinates.

### 5. The variable head drops a variable the user named — and adds one they didn't

```
"humidity and sunshine and rain in hyderabad today"
   variables = ['HUMIDITY', 'SUNSHINE']                      <- RAIN dropped
"rain and temperature in Guntur tomorrow"
   variables = ['TEMPERATURE', 'SOIL_TEMPERATURE', 'RAIN']   <- SOIL_TEMPERATURE invented
```

Unlike place names, this **is** trainable: 10 fixed labels, both directions of the error
present in the logs. This is the one part of the NLU where more data is the right answer.

**Fix.** Generate multi-variable examples (2 and 3 variables, every pair) into
`data/v4_dataset.csv`, retrain, and compare against `models/metrics_v4.json`.

**Files.** `src/v4/`, `data/v4_dataset.csv`. **Size.** A day. **Check.** `eval_v4.py` — the
multi-variable slice must improve and nothing else regress.

---

## P2 — worse than it could be

### 6. Three more advice rules still decide on summed rainfall

`_rain_protection` was fixed to decide per reading. The same defect is in three siblings:

| rule | gate | what a week of drizzle does |
|---|---|---|
| `_outdoor` | `total >= 5` → NO, `total >= 1` → CAUTION | cancels a day out |
| `_travel` | `total >= 10` → NO, `total >= 2` → CAUTION on a bike | says you will get soaked |
| `_drying` | `total >= 1` → NO | says laundry will not dry |

A sum only grows, so the same weather scores worse the longer the window asked about.

**Fix.** Same shape as `_rain_protection`: decide on the wettest reading and the count of wet
readings. **Each needs a threshold chosen** — "how much rain in one hour cancels an outing" is
a judgement about your users, not a refactor. Pick the three numbers, then it is three small
diffs.

**Files.** `backend/pipeline/advice.py`. **Size.** An hour once the numbers are decided.
**Check.** Extend `advice.demo()` with the window-length invariant already used for raincoats:
same weather over 3 rows and 7 rows must give the same verdict.

### 7. Solr matching, config only

Fixes spelling variants **for rows that exist** — it cannot help with names that are absent
(see #4). A phonetic field (Double Metaphone, or Beider-Morse for Indic names), edge n-grams,
and Solr's suggester. No new service, no new dependency.

**Size.** A day, mostly reindexing. **Check.** `kompaly`, `hyderbad`, `beramguda` resolve
without the `~1` fallback.

### 8. A bigger local model

`qwen3:0.6b` says "the data shows" in roughly one reply in three, and cannot be trusted to
re-word a failure message — given the question it invents a forecast inside the error. Both are
documented as `ponytail:` notes in `backend/generation/llm.py`.

`OLLAMA_MODEL` is now in `.env`, and `qwen3:30b-a3b` is already pulled on this machine. Try it,
measure the latency, decide. If it holds up, turn the failure path back on: the prompt is
written and kept in `phrase.explain`'s docstring.

**Size.** An hour of measuring. **Check.** `python -m backend.generation.llm`, and watch `llm_ms`.

---

## P3 — blocked, or deliberately not now

### 9. Retraining on logged feedback

`store.failed_turns()` and `store.training_rows()` both exist, so the harvest is done. The
volume is not:

| | |
|---|---|
| rows in `data/v4_dataset.csv` | **23,969** |
| corrections in the store | **16** (0.07%) |
| what all 16 are about | `error_type=time_resolution` |

16 rows cannot move the model, and there is no held-out slice proving they would not regress
it. Revisit at a few hundred corrections, or generate them (#5). The location failures in the
harvest are **not** retrain material — no number of examples teaches a model 623,920 village
names; `veedurumudi` was fixed by nine characters of `_squeeze`.

### 10. Semantic / vector search over place names

Two independent reasons it is not the answer. Retrieval cannot return a document that does not
exist, and every name in the P0/P1 items above is genuinely absent. And name matching is a
*lexical and phonetic* problem — "Beramguda" ≈ "Beeramguda" is string similarity, which
embeddings model worse than `~1` and metaphone already do.

Worth revisiting only for descriptive queries ("near BHEL", "the IT park"), and then hybrid
with lexical, using an off-the-shelf multilingual model. Never one trained here.

---

## Chores

- `update_metrics.py` (untracked, root) is the one-shot script that wrote the old metrics
  block. It patches strings that no longer exist. Delete it.
- `backend/pipeline/advice.py:154` — `low` is assigned and never read, and misleads: `_peak(rows,
  "Tmin")` is the *highest* minimum, while the line below recomputes the real coldest with
  `min()`. One-line delete.
- Add `python -m pyflakes backend/ src/` to whatever runs before a commit. It takes a second
  and would have caught the missing `import re` in `weather.py` that reached a user as
  *"name 're' is not defined"*.

---

## Done, for context

Fixed this session, with checks left behind: per-reading raincoat verdicts; sub-1mm is not
much rain; charts only when asked or when the answer is a series; columns the feed never sent
are dropped instead of shown as dashes and counted against data quality; large tables
summarised rather than truncated for the small model; `Tmin` thresholds test the low end;
comparison summaries name the winner and the direction; doubled-letter place spellings
(`veedurumudi` → `Vedurumudi`); streamed answers with visible thinking and per-stage metrics;
and every technical error replaced with a retrieval-augmented reply.
