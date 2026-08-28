"""
v4 dataset - built for uniqueness rather than row count.

    python -m src.v4.dataset --build
    python -m src.v4.dataset --stats
    python -m src.v4.dataset --check           # the quality invariants, as assertions
    python -m src.v4.dataset --samples 12

v3 was 14.6% duplicate text - one follow-up template alone produced 307 rows. So this builder
is bounded by diversity instead of by a row target:

  1. every text is unique, case-folded, across the whole file
  2. no template may produce more than MAX_PER_SKELETON rows - a skeleton is the text with
     its location and time spans masked out, so it identifies the template, not the sentence
  3. locations are drawn from a shuffled cycle over the whole vocabulary, so all 1,166 names
     appear rather than the first few hundred appearing often
  4. every name contributes four surface forms - "Hyderabad" / "Hyderabad, Telangana" /
     "Hydrabad" / "HYD" - each annotated verbatim as its own span
  5. cells are quota'd per (intent, weather_intent), and a cell that cannot be filled with
     unique rows is reported short instead of padded with repeats

Labels come from the generator, never from a regex over the finished sentence: the frame
already knows which activity and variables it was built for.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import cycle
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.tagger import normalize_time
from src.v4.entities import extract as extract_entities
from src.v4.schema import (
    ACTIVITY_VARIABLES,
    ENTITY_SUBSETS,
    ENTITY_VOCAB,
    NO_DATA_NEEDED,
    REPLIES,
    Activity,
    Aggregation,
    EntityType,
    Intent,
    Operation,
    TimeBucket,
    Variable,
    WeatherIntent,
    bucket_for,
    group_for,
    sub_activity_for,
    terms_for,
    weather_intent_for,
)

CSV_PATH = ROOT / "data" / "v4_dataset.csv"
LOCATIONS_CSV = ROOT / "data" / "locations.csv"
SEED_CSV = ROOT / "data" / "intents.csv"          # hand-written information seeds
ADVICE_SEED_CSV = ROOT / "v4_dataset.csv"          # hand-written advisory seeds

FIELDS = ["chat_id", "turn", "text", "intent", "weather_intent", "variables", "activity",
          "sub_activity", "action", "entities", "aggregation", "locations", "times",
          "times_normalized", "time_bucket", "operation", "ctx_locations", "ctx_times",
          "para_group", "split", "source", "lang"]

# --- quality knobs ----------------------------------------------------------
MAX_PER_SKELETON = 6      # rows one template may contribute, per split. v3's worst was 307.
# Rows per (source, weather_intent) cell, as a multiple of --per-cell. Reductions and
# comparisons are real but not half of what anyone asks.
QUOTA = {"information": 1.0, "aggregation": 0.45, "comparison": 0.6, "advice": 1.0,
         "implicit": 1.0, "confusion": 0.45, "comparative": 0.5, "history": 0.8,
         "longrange": 0.5}
# Share of emissions realised as a paraphrase group: same slots and labels, different wording.
# Under ~8% the model has no evidence that wording is not meaning.
REDUNDANCY = 0.065
# A per-activity floor, not a global ceiling: every activity needs keyword-free examples, or
# the model learns "spray" -> SPRAY and never reads the sentence.
MIN_IMPLICIT_SHARE = 0.20  # per activity: share of its ADVICE rows naming no cue for itself
MAX_ENTITY_LEAK = 0.25     # entity terms that only ever appear with one activity
TYPO_RATE = 0.22          # share of prompts with a misspelling outside every span
GRAMMAR_RATE = 0.18       # dropped articles, chat filler, missing question marks
QUALIFIED_RATE = 0.20     # "Guntur, Andhra Pradesh" instead of "Guntur"
MISSPELT_LOC_RATE = 0.16  # the place name itself misspelt, span updated to match
CODE_RATE = 0.08          # "HYD" instead of "Hyderabad"
# Every name in locations.csv is Title case. Without this the tagger learns Title case IS the
# location feature, and "rain in guntur" finds no place at all.
LOWER_RATE = 0.25
BARE_RATE = 0.30          # no location and/or no time at all - advisory questions omit both


# --- vocabulary -------------------------------------------------------------
VARIABLE_WORDS = {
    # "report"/"summary" ask for everything, and appeared nowhere in 24,000 rows - so
    # "last 7 days report" fell back on context and came out as RAIN.
    Variable.GENERAL: ["weather", "conditions", "forecast", "climate", "weather conditions",
                       # "update", "data", "details", "numbers" were tried and measurably
                       # hurt: "can you update where you check" is a verb.
                       "outlook", "report", "weather report", "weather summary",
                       "weather update", "weather data", "full report", "rundown",
                       "overview", "snapshot"],
    Variable.TEMPERATURE: ["temperature", "temp", "how hot it is", "heat", "warmth",
                           "how cold it is"],
    Variable.RAIN: ["rain", "rainfall", "showers", "precipitation", "downpour", "wet weather"],
    Variable.HUMIDITY: ["humidity", "moisture in the air", "how humid it is", "dampness",
                        "relative humidity"],
    Variable.WIND: ["wind", "wind speed", "breeze", "gusts", "how windy it is"],
    Variable.CLOUD: ["cloud cover", "clouds", "how cloudy it is", "overcast", "cloudiness"],
    Variable.SUNSHINE: ["sunshine", "sunlight", "sunshine hours", "how sunny it is",
                        "hours of sun"],
    Variable.UV: ["uv", "uv index", "sun strength", "how strong the sun is", "uv levels"],
    Variable.SOIL_MOISTURE: ["soil moisture", "ground moisture", "how wet the soil is",
                             "soil water", "field moisture"],
    Variable.SOIL_TEMPERATURE: ["soil temperature", "ground temperature", "soil temp",
                                "how warm the ground is"],
}

TIMES = {
    # Span quantifiers are times, not a blocklist: "rainfall fro whole day" once resolved
    # "whole" against the village index because no training text ever used one.
    WeatherIntent.CURRENT: ["now", "right now", "today", "this morning", "this afternoon",
                            "this evening", "tonight", "at the moment", "currently",
                            "later today", "in the next few hours", "at 6pm", "around noon",
                            "whole day", "the whole day", "all day", "the entire day",
                            "the full day", "rest of the day", "the rest of the day",
                            "throughout the day", "all night", "the whole night",
                            "the whole morning", "the entire afternoon", "for the day"],
    WeatherIntent.TOMORROW: ["tomorrow", "tomorrow morning", "tomorrow afternoon",
                             "tomorrow evening", "tomorrow night", "by tomorrow"],
    WeatherIntent.FORECAST: ["this week", "next week", "this weekend", "next weekend",
                             "whole week", "the whole week", "the entire week", "all week",
                             "the full week", "rest of the week", "the whole month",
                             "the entire month", "all month", "the whole weekend",
                             "over the next 30 days", "for the next 3 months",
                             "the next 3 days", "the next 5 days", "day after tomorrow",
                             "on monday", "on friday", "on sunday", "over the next few days",
                             "next month", "in 2 days", "this saturday", "this monday",
                             "next friday", "coming sunday", "on saturday", "on tuesday",
                             "this thursday", "by friday"],
    # Explicit dates and long spans are ordinary wording here - the archive serves any date,
    # and what a ten-year span comes back as is the query planner's decision, not the model's.
    WeatherIntent.HISTORICAL: ["yesterday", "last week", "last night", "last month",
                               "over the last few days", "the past week", "last sunday",
                               "on 15 august 2023", "in march 2022", "on 2023-08-15",
                               "on 12/06/2021", "in august 2020", "on 3 january 2024",
                               # a few past durations, plus the wordings that name no period
                               "last 7 days", "past 3 days", "previous week", "last 2 weeks",
                               "history", "historical", "past records", "historical data",
                               "on 21 june 2022", "back in july 2021", "in 2017",
                               "in august 2019", "in june 2018", "last saturday",
                               "on 5 may 2020", "in december 2019",
                               "from 2010 to 2025", "over the last 5 years", "for all of 2023",
                               "every year since 2015", "in the last decade",
                               "between 2015 and 2020", "for the last 90 days",
                               "over the past 6 months", "each year since 2018"],
}

# Kept separate from the frames so every frame gets every preposition.
PLACE_FORMS = ["in {loc}", "at {loc}", "for {loc}", "around {loc}", "near {loc}", "{loc}"]

# The windows that carry data. WeatherIntent.NONE is what a greeting gets - no span to draw.
WINDOWS = tuple(TIMES)

# Long enough for "year by year" to mean anything - "yesterday" would not.
LONG_TIMES = {
    WeatherIntent.HISTORICAL: ["in 2017", "from 2010 to 2025", "over the last 5 years",
                               "for all of 2023", "every year since 2015", "in the last decade",
                               "between 2015 and 2020", "over the past 6 months",
                               "each year since 2018", "for the last 90 days"],
    WeatherIntent.FORECAST: ["over the next 30 days", "for the next 3 months", "next month"],
}


# --- frames -----------------------------------------------------------------
# {v} variable noun   {loc} place phrase   {t} time phrase
INFORMATION_FRAMES = [
    "what is the {v} {loc} {t}",
    "what's the {v} {loc} {t}",
    "how is the {v} {loc} {t}",
    "tell me the {v} {loc} {t}",
    "give me the {v} {loc} {t}",
    "{v} {loc} {t}",
    "i want to know the {v} {loc} {t}",
    "can you check the {v} {loc} {t}",
    "any idea about the {v} {loc} {t}",
    "show me the {v} {loc} {t}",
    "what kind of {v} {loc} {t}",
    "how much {v} {loc} {t}",
    "is there any {v} {loc} {t}",
    "what are the {v} like {loc} {t}",
    "check {v} {loc} {t}",
    "update on the {v} {loc} {t}",
    "{t} {v} {loc}",
    "{loc} {v} {t}",
    # No frame said "will it rain tomorrow", so the nearest match was RAIN_PROTECTION's
    # "will it rain on me" and plain forecast questions came back as advice.
    "{v} {t}",
    "{v} {t} {loc}",
    "how much {v} {t}",
    "what is the {v} for {t}",
    "give me {v} for {t} {loc}",
    "will there be {v} {loc} {t}",
    "is there going to be {v} {loc} {t}",
    "will we get {v} {loc} {t}",
    "how likely is {v} {loc} {t}",
    "chances of {v} {loc} {t}",
    "expecting any {v} {loc} {t}",
    "do we get {v} {loc} {t}",
    "is {v} expected {loc} {t}",
]

# Comparative adjectives bound to the variable they mean. "which is hotter, {a} or {b}"
# carried no {v}, so the label was whatever the generator picked and "hotter" meant nothing.
COMPARATIVE = {
    Variable.TEMPERATURE: ["hotter", "cooler", "warmer", "colder", "milder"],
    Variable.RAIN: ["wetter", "rainier", "drier"],
    Variable.WIND: ["windier", "breezier", "calmer"],
    Variable.HUMIDITY: ["more humid", "muggier", "stickier", "less humid"],
    Variable.SUNSHINE: ["sunnier", "brighter"],
    Variable.CLOUD: ["cloudier", "greyer", "clearer"],
    Variable.SOIL_MOISTURE: ["wetter underfoot", "drier in the ground"],
}
COMPARATIVE_FRAMES = [
    "which is {c}, {a} or {b}, {t}",
    "which one is {c} {t}, {a} or {b}",
    "is {a} {c} than {b} {t}",
    "{a} or {b} which is {c} {t}",
    "which city is {c} {t}, {a} or {b}",
    "between {a} and {b} which is {c} {t}",
    "tell me which is {c} {t} - {a} or {b}",
    "{a} vs {b}, which is {c} {t}",
]

COMPARISON_FRAMES = [
    "compare the {v} in {a} and {b} {t}",
    "{a} vs {b} {v} {t}",
    "is the {v} higher in {a} or {b} {t}",
    "difference in {v} between {a} and {b} {t}",
    "how does the {v} in {a} compare to {b} {t}",
    "{v} in {a} versus {b} {t}",
    "which of {a} and {b} has more {v} {t}",
    "compare {a} with {b} on {v} {t}",
    "between {a} and {b}, where is the {v} better {t}",
]

# One entry per activity, phrased as a decision rather than a measurement. {loc} and {t} are
# optional in all of them - real advisory questions often name neither.
ADVICE_FRAMES = {
    Activity.RAIN_PROTECTION: [
        "should i take an umbrella {loc} {t}", "do i need an umbrella {loc} {t}",
        "can i leave the umbrella at home {t}", "will i get wet if i go out {loc} {t}",
        "should i carry a {clothing} {loc} {t}", "do i need rain gear {loc} {t}",
        "umbrella needed {loc} {t}", "will it rain on me {loc} {t}",
    ],
    Activity.SUN_PROTECTION: [
        "do i need sunscreen {loc} {t}", "should i apply sunscreen {loc} {t}",
        "will the sun be harsh {loc} {t}", "is sun protection needed {loc} {t}",
        "should i carry a cap {loc} {t}", "will i need shade {loc} {t}",
        "how strong will the sun be {loc} {t}",
    ],
    Activity.CLOTHING: [
        "do i need a {clothing} {loc} {t}", "should i wear a {clothing} {loc} {t}",
        "what should i wear {loc} {t}", "will i be cold {loc} {t}",
        "is it {clothing} weather {loc} {t}", "should i dress warm {loc} {t}",
    ],
    Activity.OUTDOOR_ACTIVITY: [
        "should i go outside {loc} {t}", "can we play {sport} {loc} {t}",
        "will the {sport} match be rained out {loc} {t}",
        "is it good weather for {sport} {loc} {t}", "can i go for a walk {loc} {t}",
        "is it too hot to jog {loc} {t}", "can i do some gardening {loc} {t}",
        "should i water the garden {loc} {t}", "can we pour concrete {loc} {t}",
        "is it safe to work on the site {loc} {t}",
        "we have a {event} {loc} {t}, will the weather hold",
        "should we move the {event} indoors {loc} {t}",
        "is it a good time to head out {loc} {t}", "can i step out {loc} {t}",
        "should we cancel the {sport} game {loc} {t}", "can the masons work {loc} {t}",
        "is it nice enough to be outdoors {loc} {t}", "can i take my evening walk {loc} {t}",
    ],
    Activity.TRAVEL: [
        "can i take the {transport} to {loc} {t}", "is it a good day to travel {loc} {t}",
        "should i postpone my trip {loc} {t}", "is it safe to drive {loc} {t}",
        "will the weather affect my journey {loc} {t}",
        "planning to go by {transport} {loc} {t}, will it hold",
        "will my commute be bad {loc} {t}", "should i leave early for work {loc} {t}",
        "should i take the {transport} to office {loc} {t}",
        "will the roads be bad for the {transport} {loc} {t}",
    ],
    Activity.DRYING: [
        "will my clothes dry {loc} {t}", "should i bring the clothes in {loc} {t}",
        "can i hang the washing outside {loc} {t}", "is it good drying weather {loc} {t}",
        "should i do the laundry {loc} {t}", "can i wash my {clothing} {t}",
        "will the {clothing} dry outside {loc} {t}",
        "should i wash the {transport} {loc} {t}",
        "is it pointless washing the {transport} {t}",
        "good day to clean the {transport} {loc} {t}",
        "should i clean the terrace {loc} {t}", "should i air out the {clothing} {t}",
    ],
    Activity.SOW: [
        "should i sow {crop} now {loc} {t}", "is the soil ready for planting {crop} {loc} {t}",
        "can i start sowing {loc} {t}", "is it a good time to plant {crop} {loc} {t}",
        "should i wait to sow the {crop} {loc} {t}",
    ],
    Activity.IRRIGATE: [
        "do i need to water the {crop} {loc} {t}", "should i irrigate {loc} {t}",
        "does the {crop} need watering {loc} {t}", "can i skip irrigation {loc} {t}",
        "should i run the pump for the {crop} {loc} {t}",
    ],
    Activity.FERTILIZE: [
        "should i apply {material} to the {crop} {loc} {t}",
        "is it a good time to apply {material} {loc} {t}",
        "should i fertilize the {crop} {loc} {t}", "will the {material} wash off {loc} {t}",
        "can i put {material} on the field {loc} {t}", "is {t} okay for fertilizing {loc}",
    ],
    Activity.SPRAY: [
        "can i spray {material} on the {crop} {loc} {t}", "is it too windy to spray {loc} {t}",
        "should i spray the {crop} {loc} {t}", "is {t} suitable for spraying {loc}",
        "when can i spray {material} safely {loc} {t}",
        "will the {material} spray drift {loc} {t}",
    ],
    Activity.HARVEST: [
        "can i harvest the {crop} {loc} {t}", "is it safe to harvest {loc} {t}",
        "should i wait to harvest the {crop} {loc} {t}",
        "will the {crop} stay dry for harvesting {loc} {t}",
        "is it dry enough to cut the {crop} {loc} {t}",
    ],
}

# Frames naming no keyword for their own activity. Before these, 61% of ADVICE rows gave the
# label away in a single word.
IMPLICIT_FRAMES = {
    Activity.SPRAY: [
        "should i protect the {crop} from insects {loc} {t}",
        "the {crop} has pest attack, what should i do {loc} {t}",
        "there are bollworms on the {crop} {loc} {t}, is it a good time to treat",
        "can i treat the {crop} for disease {loc} {t}",
        "fungal spots on the {crop} {loc} {t}, should i act now",
    ],
    Activity.IRRIGATE: [
        "the {crop} is wilting {loc} {t}, what should i do",
        "the field looks dry {loc} {t}, should i do something",
        "soil is cracking in my field {loc} {t}", "the plants are drooping {loc} {t}",
    ],
    Activity.FERTILIZE: [
        "the {crop} looks pale and yellow {loc} {t}, should i feed it",
        "is {t} right for top dressing the {crop} {loc}",
        "the {crop} needs nutrients {loc} {t}, when should i do it",
    ],
    Activity.HARVEST: [
        "the {crop} is ready {loc} {t}, should i bring it in",
        "grain is dry enough {loc}, is {t} a safe window",
        "can i get the {crop} off the field {t} {loc}",
        "grain is at the right moisture {loc} {t}, should i go ahead",
        "is {t} safe to bring the {crop} in {loc}",
    ],
    Activity.SOW: [
        "the rains have set in {loc}, should i begin field work {t}",
        "is the ground warm and wet enough for germination {loc} {t}",
        "kharif is starting {loc}, is {t} right to begin",
        "when can i start the season for {crop} {loc} {t}",
        "is the field ready to go in {t} {loc}",
        "monsoon looks settled {loc}, can i begin {t}",
        "is {t} a good start for the {crop} season {loc}",
    ],
    Activity.OUTDOOR_ACTIVITY: [
        "the pitch is wet {loc}, can we still play {t}",
        "we booked the ground for {t} {loc}, will it be usable",
        "kids want to be outside {t} {loc}, is that fine",
        "the {event} is outdoors {loc} {t}, should we worry",
    ],
    Activity.TRAVEL: [
        "i have a long drive {t} {loc}, anything to worry about",
        "is the highway going to be rough {t} {loc}",
        "leaving for {loc} {t}, how are things there",
    ],
    Activity.DRYING: [
        "everything is still damp {loc} {t}, will it help to leave it out",
        "no space indoors for the {clothing} {loc} {t}",
        "the terrace line is full {loc} {t}, is that a mistake",
        "will things left outside be fine {t} {loc}",
        "is it worth putting anything on the line {t} {loc}",
        "the {transport} is filthy {loc}, is {t} a waste of effort",
    ],
    Activity.RAIN_PROTECTION: [
        "i am walking to the station {t} {loc}, should i be worried",
        "no cover where i am going {t} {loc}",
    ],
    Activity.SUN_PROTECTION: [
        "i will be out in the open all afternoon {loc} {t}",
        "standing in the field all day {t} {loc}, anything i should take",
    ],
    Activity.CLOTHING: [
        "will i regret going out in a t shirt {loc} {t}",
        "taking the kids out {t} {loc}, how should i dress them",
    ],
}

# Deliberate crossings: the verb decides the activity, not the material. Without these, 51 of
# 131 entity terms mapped to exactly one activity and the model could skip the sentence.
CONFUSION_FRAMES = [
    ("should i spray {material} on the {crop} {loc} {t}", Activity.SPRAY,
     {EntityType.MATERIAL: "FERTILIZE"}),
    ("can i apply {material} by spraying {loc} {t}", Activity.SPRAY,
     {EntityType.MATERIAL: "FERTILIZE"}),
    ("is {t} good for spraying {material} {loc}", Activity.SPRAY,
     {EntityType.MATERIAL: "FERTILIZE"}),
    ("should i mix {material} into the soil {loc} {t}", Activity.FERTILIZE,
     {EntityType.MATERIAL: "SPRAY"}),
    ("can i broadcast {material} on the field {loc} {t}", Activity.FERTILIZE,
     {EntityType.MATERIAL: "SPRAY"}),
    # water: irrigating the field, versus spraying water on the leaves
    ("should i water the {crop} {loc} {t}", Activity.IRRIGATE, {}),
    ("should i spray water on the {crop} {loc} {t}", Activity.SPRAY, {}),
    ("does the field need water {loc} {t}", Activity.IRRIGATE, {}),
    # wash: same activity, different sub-activity, near-identical wording
    ("should i wash my {transport} {loc} {t}", Activity.DRYING, {}),
    ("should i wash my {clothing} {loc} {t}", Activity.DRYING, {}),
    ("can i clean the {transport} {loc} {t}", Activity.DRYING, {}),
    # riding for transport, versus riding for exercise
    ("can i take the {transport} to work {loc} {t}", Activity.TRAVEL, {}),
    ("can i take the {transport} out for exercise {loc} {t}", Activity.OUTDOOR_ACTIVITY, {}),
    # carrying an umbrella, versus wearing something warm
    ("should i take a {clothing} {loc} {t}", Activity.RAIN_PROTECTION,
     {EntityType.CLOTHING_ITEM: "RAIN"}),
    ("should i take a {clothing} {loc} {t}", Activity.CLOTHING,
     {EntityType.CLOTHING_ITEM: "WARM"}),
]
# Material pools the crossings draw from, by name so a frame can ask for the *other* one.
CROSS_POOLS = {
    "SPRAY": ["pesticide", "insecticide", "fungicide", "herbicide", "neem oil"],
    "FERTILIZE": ["urea", "dap", "npk", "fertilizer", "manure", "compost", "potash"],
    "RAIN": ["raincoat", "umbrella"],
    "WARM": ["jacket", "sweater", "woollens", "shawl"],
}

# Fragments that lean on the previous turn. In v3 six of these produced 1,455 rows.
FOLLOW_FRAMES = {
    Operation.REPLACE: ["what about {loc}", "and {loc}", "how about {loc}", "and in {loc}",
                        "{loc}", "same for {loc}", "now {loc}", "ok and {loc}",
                        "check {loc} too", "what about over in {loc}", "{loc} instead",
                        "do the same for {loc}", "and how about {loc}", "try {loc}",
                        "now show me {loc}", "same question for {loc}", "and near {loc}"],
    Operation.MODIFY: ["what about {t}", "and {t}", "{t}", "what about {t} then",
                       "and for {t}", "same but {t}", "ok and {t}", "how about {t}",
                       "and what about {t}", "change that to {t}", "{t} instead",
                       "same place {t}", "and {t} then", "what if i go {t}",
                       "now check {t}", "and later, {t}"],
    Operation.INHERIT: ["and there", "what about there", "same place", "and that place",
                        "how about there", "there", "same there", "and that one",
                        "what about that place", "and over there", "same as before",
                        "and the same there", "that place too", "how about that one"],
}

# Wording that carries a reduction. The label comes from the frame, not from a word search.
HISTORY_FRAMES = [
    "what was the {v} {loc} {t}",
    "show me the {v} {loc} {t}",
    "give me {v} {t} {loc}",
    "how much {v} did {loc} get {t}",
    "{v} {t} for {loc}",
    "what has the {v} been {loc} {t}",
    "look up {v} {loc} {t}",
    "pull the {v} {loc} {t}",
    "{v} records for {loc} {t}",
    "did it rain {loc} {t}",
    "was it hot {loc} {t}",
    "check {v} {loc} {t}",
]

# "history" as a NOUN, where the word is the time span. The modifier form was learned; this
# was not, because no frame ever put a time word in that position.
HISTORY_NOUN_FRAMES = [
    "{t} of {v} for {loc}",
    "{t} of {v} in {loc}",
    "{v} {t} for {loc}",
    "{v} {t} of {loc}",
    "give me the {t} of {v} {loc}",
    "show me {v} {t} for {loc}",
    "i want the {t} of {v} {loc}",
    "{loc} {v} {t}",
    "pull up {v} {t} for {loc}",
    "can i see the {v} {t} for {loc}",
]
HISTORY_WORDS = ["history", "historical data", "past records", "past data",
                 "historical records", "history data", "previous records"]

LONG_RANGE_FRAMES = [
    "how much {v} did {loc} get {t}",
    "give me the {v} for {loc} {t}",
    "{v} totals for {loc} {t}",
    "what was the {v} like in {loc} {t}",
    "year by year {v} for {loc} {t}",
    "monthly {v} for {loc} {t}",
    "show me {v} trends in {loc} {t}",
    "average {v} in {loc} {t}",
    "has the {v} in {loc} changed {t}",
    "{v} history for {loc} {t}",
]

AGG_FRAMES = {
    Aggregation.SUM: ["total {v} {loc} {t}", "how much {v} in total {loc} {t}",
                      "cumulative {v} {loc} {t}", "add up the {v} {loc} {t}"],
    Aggregation.AVG: ["average {v} {loc} {t}", "what is the mean {v} {loc} {t}",
                      "typical {v} {loc} {t}", "{v} on average {loc} {t}"],
    Aggregation.MAX: ["highest {v} {loc} {t}", "peak {v} {loc} {t}",
                      "what is the maximum {v} {loc} {t}", "how hot does it get {loc} {t}"],
    Aggregation.MIN: ["lowest {v} {loc} {t}", "minimum {v} {loc} {t}",
                      "how cold does it get {loc} {t}", "coldest {v} {loc} {t}"],
    Aggregation.TREND: ["when does the {v} start dropping {loc} {t}",
                        "is the {v} rising {loc} {t}", "how is the {v} changing {loc} {t}",
                        "when will the {v} pick up {loc} {t}"],
}
AGG_VARIABLES = {
    Aggregation.SUM: [Variable.RAIN, Variable.SUNSHINE],
    Aggregation.AVG: [Variable.TEMPERATURE, Variable.HUMIDITY, Variable.WIND],
    Aggregation.MAX: [Variable.TEMPERATURE, Variable.WIND, Variable.RAIN, Variable.UV],
    Aggregation.MIN: [Variable.TEMPERATURE, Variable.HUMIDITY, Variable.SOIL_MOISTURE],
    Aggregation.TREND: [Variable.TEMPERATURE, Variable.RAIN, Variable.WIND, Variable.CLOUD],
}

# Turns that never reach the weather API. Written out rather than generated: the whole point
# is the phrasings a weather template could never produce.
CHITCHAT_FRAMES = {
    Intent.GREETING: [
        "hi", "hello", "hey", "hey there", "good morning", "good afternoon", "good evening",
        "hii", "helo", "hello there", "yo", "namaste", "hi bot", "hey buddy", "morning",
        "greetings", "hello?", "anyone there", "hi again", "hey, you up", "haan ji hello",
    ],
    Intent.THANKS: [
        "thanks", "thank you", "thanks a lot", "thx", "ty", "thank you so much",
        "that helps", "that helped, thanks", "perfect thanks", "great, thank you",
        "appreciate it", "nice, thanks", "cool thanks", "thanku", "thanks buddy",
        "much appreciated", "thanks for the info", "ok thanks",
    ],
    Intent.GOODBYE: [
        "bye", "goodbye", "see you", "see ya", "bye bye", "cya", "later", "talk later",
        "catch you later", "good night", "gn", "signing off", "that is all", "im done",
        "ok bye", "alright bye", "nothing else, bye", "thats it for now",
    ],
    Intent.SMALL_TALK: [
        "how are you", "how are you doing", "are you a bot", "are you human", "who are you",
        "what is your name", "whats your name", "are you real", "do you sleep",
        "how old are you", "are you an ai", "who made you", "do you like rain",
        "whats up", "how is your day", "are you there", "do you get bored",
        "whats your favourite weather",
    ],
    Intent.CAPABILITY: [
        "what can you do", "help", "what do you do", "how do i use this", "what can i ask",
        "show me what you can do", "what are your features", "can you help me",
        "what questions can i ask", "how does this work", "give me some examples",
        "what all can you tell me", "do you do farming advice", "can you compare places",
        "list your commands", "menu", "options", "what data do you have",
    ],
    Intent.UNSUPPORTED_METRIC: [
        "what is the air quality {loc}", "aqi {loc}", "pollution levels {loc}",
        "is there snow {loc}", "snow depth {loc}", "pollen count {loc}",
        "any earthquakes {loc}", "is there a tsunami warning {loc}",
        "what is the moon phase {loc}", "tide timings {loc}", "lightning strikes {loc}",
        "air pollution index {loc}", "how bad is the smog {loc}", "is there fog {loc}",
        "visibility {loc}", "sea level {loc}", "will there be a cyclone {loc}",
        "pm2.5 reading {loc}", "aurora forecast {loc}", "water level in the river {loc}",
    ],
    Intent.OUT_OF_SCOPE: [
        "book me a flight", "what is the stock price of tcs", "tell me a joke",
        "who won the match yesterday", "order me food", "what is 2 plus 2",
        "translate this to hindi", "play some music", "set an alarm for 6am",
        "what is the capital of france", "send a message to amma", "call my brother",
        "how do i cook biryani", "whats the traffic like", "find me a hotel",
        "what is the gold rate today", "book a cab", "who is the prime minister",
        "recommend a movie", "help me with my homework", "what is bitcoin at",
    ],
    Intent.UNCLEAR: [
        "asdf", "??", "hmm", "the thing", "do it", "maybe", "...", "wait",
        "hello world test", "abc def", "idk", "again", "..?", "hmmm what",
        "you know", "whatever", "test test", "aaaa", "qwerty", "blah",
    ],
    Intent.CHANGE_LOCATION: [
        "can i change my location", "change my location", "set my location",
        "i want to change the place", "use a different location", "switch to another place",
        "update my location", "i moved to a new place", "can you use my current location",
        "use my gps", "change the city", "different village please",
        "not that place, use another one", "set location to {loc}", "my location is {loc}",
        "i am in {loc} now", "change it to {loc}", "use {loc} instead from now on",
        "actually i meant {loc}", "shift to {loc}",
    ],
    Intent.RESET: [
        "start over", "new chat", "clear this", "forget that", "reset", "start fresh",
        "lets start again", "clear the chat", "begin again", "wipe this conversation",
        "start a new conversation", "forget everything", "restart", "clear history",
        "lets start from scratch", "reset the chat", "forget what i said",
        "start again please", "new conversation",
    ],
    Intent.AFFIRM: [
        "yes", "yeah", "yep", "correct", "that one", "right", "ok yes", "sure", "exactly",
        "yes please", "haan", "correct one", "yup", "thats the one", "yes that is right",
        "affirmative", "ya", "of course", "go ahead",
    ],
    Intent.DENY: [
        "no", "nope", "not that", "wrong", "that is wrong", "no thats not it", "nah",
        "incorrect", "not correct", "no not that one", "wrong answer", "thats not right",
        "nope not that", "no thanks", "neither", "none of those", "not what i meant",
    ],
    Intent.EXPLAIN: [
        "why", "why do you say that", "on what basis", "how do you know", "explain that",
        "say that again", "repeat that", "come again", "i didnt get that",
        "can you repeat", "elaborate", "what do you mean", "why not", "how did you decide",
        "where did you get that", "what is that based on", "explain", "why is that",
        "how sure are you", "what made you say that",
    ],
}

DROPPABLE = {"the", "a", "an", "is", "are", "will", "be", "do", "does", "of", "about",
             "any", "there", "me", "to", "it", "in", "for", "and"}
FILLERS = ["pls", "plz", "kindly", "asap", "sir", "bro", "urgent", "quickly"]


# --- text helpers -----------------------------------------------------------

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" ,")


def _misspell(rng: random.Random, word: str) -> str:
    """One realistic keyboard slip in a single word."""
    i = rng.randrange(1, len(word) - 1)
    roll = rng.random()
    if roll < 0.35:
        return word[:i] + word[i + 1:]                            # dropped letter
    if roll < 0.65:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]    # swapped pair
    if roll < 0.85:
        return word[:i] + word[i] + word[i:]                      # doubled letter
    return word[:i] + rng.choice("aeiou") + word[i + 1:]          # wrong vowel


def _free_words(text: str, spans: list[str], pattern: str):
    """Word matches that overlap no annotated span - the only text safe to mangle."""
    protected = []
    for span in spans:
        start = text.find(span)
        if start >= 0:
            protected.append((start, start + len(span)))
    return [m for m in re.finditer(pattern, text)
            if not any(m.start() < end and start < m.end() for start, end in protected)]


def _typo(rng: random.Random, text: str, spans: list[str]) -> str:
    """Misspell one word OUTSIDE every span, so each span stays verbatim in the prompt."""
    words = _free_words(text, spans, r"[A-Za-z]{5,}")
    if not words:
        return text
    word = rng.choice(words)
    return text[:word.start()] + _misspell(rng, word.group()) + text[word.end():]


def _bad_grammar(rng: random.Random, text: str, spans: list[str]) -> str:
    """How people actually type. Spans are never touched."""
    roll = rng.random()
    if roll < 0.45:
        words = [m for m in _free_words(text, spans, r"\b[a-zA-Z]+\b")
                 if m.group().lower() in DROPPABLE]
        if words:
            word = rng.choice(words)
            text = text[:word.start()] + text[word.end():]
    elif roll < 0.70:
        text = text.rstrip("?. ") + rng.choice(["", "", "??", "?!", "..", " ?"])
    elif roll < 0.88:
        filler = rng.choice(FILLERS)
        text = f"{filler} {text}" if rng.random() < 0.4 else f"{text.rstrip('?. ')} {filler}"
    else:
        text = re.sub(r"^(what is|whats|what's|how is|can you tell me|i want to know)\s+", "",
                      text, flags=re.I)
    return _clean(text)


def skeleton(text: str, spans: list[str]) -> str:
    """The text with its spans masked - identifies the template a row came from."""
    for span in sorted(spans, key=len, reverse=True):
        text = text.replace(span, "@")
    return re.sub(r"\s+", " ", text.lower()).strip(" ?.!,")


# --- location vocabulary ----------------------------------------------------

def load_locations(path: Path = LOCATIONS_CSV) -> dict[str, list[dict]]:
    """data/locations.csv -> per-split place records carrying every surface form.

    The codes and misspellings columns come from src/fetch_locations.py - "HYD" and "Hydrabad"
    are how people type.
    """
    if not path.exists():
        raise SystemExit(f"{path} missing - run: python src/fetch_locations.py")
    pools: dict[str, list[dict]] = {"train": [], "eval": []}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            name = " ".join(row["name"].split())
            parents = [p for p in (row.get("parents") or "").split(" | ") if p]
            pools[row["split"]].append({
                "name": name,
                "qualified": f"{name}, {parents[-1]}" if parents else name,
                "codes": [c for c in (row.get("codes") or "").split("|") if c],
                "typos": [t for t in (row.get("misspellings") or "").split("|") if t],
            })
    return pools


ENTITY_SLOTS = {f"{{{kind.value}}}": kind for kind in EntityType}


def fill_entities(rng: random.Random, text: str, activity=Activity.NONE,
                  pool=None) -> tuple[str, dict[str, list[str]]]:
    """Swap {sport} / {crop} / {material} ... for a real term, and record what went in.

    The annotation is exact because the generator chose the entity. `check` confirms the
    gazetteer finds it back, which keeps ENTITY_VOCAB and the frames from drifting apart.
    """
    entities: dict[str, list[str]] = {}
    for slot, kind in ENTITY_SLOTS.items():
        while slot in text:
            choices = (pool or {}).get(kind) or terms_for(activity, kind)
            term = rng.choice(choices)
            text = text.replace(slot, term, 1)
            entities.setdefault(kind.value, []).append(term)
    return text, entities


def _regroup(entities: dict[str, list[str]], spans: list[str]) -> dict[str, list[str]]:
    """Put re-cased entity spans back under their own types, in the order they went out."""
    out, cursor = {}, 0
    for kind, terms in entities.items():
        out[kind] = spans[cursor:cursor + len(terms)]
        cursor += len(terms)
    return out


def surface(rng: random.Random, place: dict) -> str:
    """One way of writing this place - canonical, qualified, misspelt, code, or lowercased."""
    roll = rng.random()
    if place["codes"] and roll < CODE_RATE:
        return rng.choice(place["codes"])
    if place["typos"] and roll < CODE_RATE + MISSPELT_LOC_RATE:
        written = rng.choice(place["typos"])
    elif roll < CODE_RATE + MISSPELT_LOC_RATE + QUALIFIED_RATE:
        written = place["qualified"]
    else:
        written = place["name"]
    return written.lower() if rng.random() < LOWER_RATE else written


# --- row assembly -----------------------------------------------------------

def _row(chat_id, turn, text, *, intent, weather_intent, variables, activity, aggregation,
         locations, times, operation, ctx_locations, ctx_times, split, source, lang="en",
         entities=None, para_group=""):
    normalized = [normalize_time(span) for span in times]
    return {
        "chat_id": chat_id, "turn": turn, "text": text,
        "intent": intent.value, "weather_intent": weather_intent.value,
        "variables": "|".join(v.value for v in variables),
        "activity": activity.value, "action": group_for(activity).value,
        # derived, never annotated: an open string read back off the entities and the text
        "sub_activity": sub_activity_for(activity, entities or {}, text),
        "entities": json.dumps(entities or {}), "para_group": para_group,
        "aggregation": aggregation.value,
        "locations": json.dumps(locations), "times": json.dumps(times),
        "times_normalized": json.dumps(normalized),
        "time_bucket": bucket_for(normalized[0] if normalized else None).value,
        "operation": operation.value,
        "ctx_locations": json.dumps(ctx_locations), "ctx_times": json.dumps(ctx_times),
        "split": split, "source": source, "lang": lang,
    }


class Builder:
    """Accumulates rows while enforcing the two quality rules: unique text, capped template."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.rows: list[dict] = []
        self.seen: set[str] = set()
        self.skeletons: Counter = Counter()
        self.rejected = Counter()

    def add(self, text: str, spans: list[str], **labels) -> bool:
        text = _clean(text)
        key = text.lower()
        if not text or key in self.seen:
            self.rejected["duplicate text"] += 1
            return False
        shape = skeleton(text, spans)
        if self.skeletons[shape] >= MAX_PER_SKELETON:
            self.rejected["template at cap"] += 1
            return False
        if not all(span in text for span in spans):
            self.rejected["span not verbatim"] += 1     # Rule 4.1, checked at write time
            return False
        self.seen.add(key)
        self.skeletons[shape] += 1
        self.rows.append(_row(text=text, **labels))
        return True


def _phrase(rng: random.Random, place: str | None, time_span: str | None,
            frame: str) -> tuple[str, list[str], list[str]]:
    """Fill one frame, returning the text and the spans exactly as they appear in it."""
    locations, times = [], []
    if place:
        rendered = rng.choice(PLACE_FORMS).format(loc=place)
        locations.append(place)
    else:
        rendered = ""
    if time_span:
        times.append(time_span)
    text = frame.replace("{loc}", rendered).replace("{t}", time_span or "")
    return _clean(text), locations, times


def _noise(rng: random.Random, text: str, spans: list[str]) -> str:
    if rng.random() < GRAMMAR_RATE:
        text = _bad_grammar(rng, text, spans)
    if rng.random() < TYPO_RATE:
        text = _typo(rng, text, spans)
    return text


def _capitalise(text: str, spans: list[str]) -> tuple[str, list[str]]:
    """Sentence case, carrying any span that starts the sentence along with it - or the span
    is no longer verbatim in its own prompt (Rule 4.1) and the row is thrown away."""
    if not text or not text[0].islower():
        return text, spans
    head = text[0].upper() + text[1:]
    return head, [s[0].upper() + s[1:] if text.startswith(s) else s for s in spans]


def generate(rng: random.Random, places: list[dict], split: str, per_cell: int) -> Builder:
    """Fill every (source, weather_intent) cell to `per_cell` unique rows, or report short."""
    builder = Builder(rng)
    wheel = list(places)
    rng.shuffle(wheel)
    spin = cycle(wheel)                       # even coverage: every name before any repeat
    counter = defaultdict(int)
    group_id = itertools.count()

    def next_place(bare_ok: bool):
        """A place, or None - advisory and follow-up questions frequently name none."""
        if bare_ok and rng.random() < BARE_RATE:
            return None
        return surface(rng, next(spin))

    def emit(intent, want, frames, *, activity=Activity.NONE, variables=None,
             aggregation=Aggregation.RAW, operation=Operation.SET, source="gen",
             bare_ok=False, second=None, pool=None, times_pool=None):
        """`want` chooses which time wordings to draw from; the label is derived from the
        span actually drawn, so weather_intent and times cannot disagree.

        With probability REDUNDANCY the same slots are realised through two or three of
        `frames`, giving paraphrase groups that share a `para_group`.
        """
        if isinstance(frames, str):
            frames = [frames]
        place = next_place(bare_ok)
        wordings = times_pool or TIMES[want]
        time_span = rng.choice(wordings) if (place or not bare_ok
                                             or rng.random() < 0.7) else None
        weather_intent = weather_intent_for(normalize_time(time_span) if time_span else None)
        # the source is part of the cell key, or the plain information frames fill every
        # INFORMATION cell first and the reduction frames are all refused
        cell = (source, weather_intent)
        if counter[cell] >= round(per_cell * QUOTA.get(source, 1.0)):
            return False
        variables = variables or [Variable.GENERAL]
        word = rng.choice(VARIABLE_WORDS[variables[0]])

        wanted = 1
        if len(frames) > 1 and rng.random() < REDUNDANCY:
            wanted = 2 if rng.random() < 0.75 else 3
        picked = rng.sample(frames, min(wanted, len(frames)))
        group = f"{split[:2]}-p{next(group_id):05d}" if len(picked) > 1 else ""

        made = 0
        for frame in picked:
            text = frame.replace("{v}", word)
            text, entities = fill_entities(rng, text, activity, pool)
            entity_spans = [span for spans in entities.values() for span in spans]
            if second is not None:                          # a comparison names two places
                text = text.replace("{a}", place or "here").replace("{b}", second)
                locations = [p for p in (place, second) if p]
                times = [time_span] if time_span else []
                text = _clean(text.replace("{t}", time_span or ""))
            else:
                text, locations, times = _phrase(rng, place, time_span, text)
            # entity spans are protected text too: a typo inside "cotton" would leave the
            # annotation pointing at a word that is no longer there
            all_spans = locations + times + entity_spans
            text, all_spans = _capitalise(_noise(rng, text, all_spans), all_spans)
            n_places, n_stamps = len(locations), len(times)
            locations = all_spans[:n_places]
            times = all_spans[n_places:n_places + n_stamps]
            entities = _regroup(entities, all_spans[n_places + n_stamps:])
            made += builder.add(
                text, all_spans, chat_id=f"{split[:2]}-{len(builder.rows):06d}", turn=0,
                intent=intent, weather_intent=weather_intent, variables=variables,
                activity=activity, aggregation=aggregation, locations=locations, times=times,
                operation=operation, ctx_locations=locations, ctx_times=times,
                split=split, source=source, entities=entities,
                para_group=group if len(picked) > 1 else "")
        counter[cell] += made
        return made > 0

    # INFORMATION - the plain "what is the X" requests, every variable, every window
    for weather_intent in WINDOWS:
        for _ in range(per_cell * 4):                       # over-offer; the cap does the rest
            # GENERAL is drawn far more often than a uniform pick would give it
            variables = [Variable.GENERAL if rng.random() < 0.25 else rng.choice(list(Variable))]
            if rng.random() < 0.12:                         # "rain and temperature"
                other = rng.choice([v for v in Variable if v != variables[0]])
                variables.append(other)
            emit(Intent.INFORMATION, weather_intent, INFORMATION_FRAMES,
                 variables=variables, source="information")

    # INFORMATION with a reduction - the 'determine' functions
    for weather_intent in WINDOWS:
        for _ in range(per_cell * 2):
            aggregation = rng.choice([a for a in Aggregation if a is not Aggregation.RAW])
            emit(Intent.INFORMATION, weather_intent, AGG_FRAMES[aggregation],
                 variables=[rng.choice(AGG_VARIABLES[aggregation])], aggregation=aggregation,
                 source="aggregation")

    # COMPARISON - always two places
    for weather_intent in WINDOWS:
        for _ in range(per_cell * 4):
            emit(Intent.COMPARISON, weather_intent, COMPARISON_FRAMES,
                 variables=[rng.choice(list(Variable))], operation=Operation.COMPARE,
                 second=surface(rng, next(spin)), source="comparison")

    # ADVICE - one decision per row, variables implied by the activity
    activities = [a for a in Activity if a is not Activity.NONE]
    for weather_intent in WINDOWS:
        for _ in range(per_cell * 6):
            activity = rng.choice(activities)
            emit(Intent.ADVICE, weather_intent, ADVICE_FRAMES[activity],
                 activity=activity, variables=ACTIVITY_VARIABLES[activity] or [Variable.GENERAL],
                 bare_ok=True, source="advice")

    # explicit history: "last 7 days", "past records", "history of rainfall"
    for _ in range(per_cell * 4):
        emit(Intent.INFORMATION, WeatherIntent.HISTORICAL, HISTORY_FRAMES,
             variables=[rng.choice([Variable.RAIN, Variable.TEMPERATURE, Variable.HUMIDITY,
                                    Variable.WIND, Variable.GENERAL, Variable.SOIL_MOISTURE])],
             aggregation=rng.choice([Aggregation.RAW, Aggregation.RAW, Aggregation.SUM,
                                     Aggregation.AVG]),
             source="history")

    # "history of rainfall" / "rainfall history" - the time word sits in a noun slot
    for _ in range(per_cell * 3):
        emit(Intent.INFORMATION, WeatherIntent.HISTORICAL, HISTORY_NOUN_FRAMES,
             variables=[rng.choice([Variable.RAIN, Variable.TEMPERATURE, Variable.HUMIDITY,
                                    Variable.WIND, Variable.SOIL_MOISTURE])],
             source="history", times_pool=HISTORY_WORDS)

    # long spans as ordinary INFORMATION - the planner decides what comes back
    for weather_intent in (WeatherIntent.HISTORICAL, WeatherIntent.FORECAST):
        for _ in range(per_cell * 2):
            aggregation = rng.choice([Aggregation.RAW, Aggregation.SUM, Aggregation.AVG])
            emit(Intent.INFORMATION, weather_intent, LONG_RANGE_FRAMES,
                 variables=[rng.choice([Variable.RAIN, Variable.TEMPERATURE,
                                        Variable.HUMIDITY, Variable.SUNSHINE])],
                 aggregation=aggregation, source="longrange",
                 times_pool=LONG_TIMES[weather_intent])

    # COMPARISON by comparative adjective - "which is cooler" must mean TEMPERATURE
    for weather_intent in WINDOWS:
        for _ in range(per_cell * 3):
            variable = rng.choice(list(COMPARATIVE))
            frame = rng.choice(COMPARATIVE_FRAMES).replace(
                "{c}", rng.choice(COMPARATIVE[variable]))
            emit(Intent.COMPARISON, weather_intent, [frame], variables=[variable],
                 operation=Operation.COMPARE, second=surface(rng, next(spin)),
                 source="comparative")

    # ADVICE where no word gives the label away - the fix for the shortcut rate
    implicit = list(IMPLICIT_FRAMES)
    for weather_intent in WINDOWS:
        for _ in range(per_cell * 3):
            activity = rng.choice(implicit)
            emit(Intent.ADVICE, weather_intent, IMPLICIT_FRAMES[activity],
                 activity=activity, variables=ACTIVITY_VARIABLES[activity] or [Variable.GENERAL],
                 bare_ok=True, source="implicit")

    # ADVICE crossings - the entity deliberately disagrees with the usual activity
    for weather_intent in WINDOWS:
        for _ in range(per_cell * 2):
            frame, activity, pools = rng.choice(CONFUSION_FRAMES)
            emit(Intent.ADVICE, weather_intent, [frame], activity=activity,
                 variables=ACTIVITY_VARIABLES[activity] or [Variable.GENERAL],
                 bare_ok=True, source="confusion",
                 pool={kind: CROSS_POOLS[name] for kind, name in pools.items()})

    return builder


def generate_chitchat(rng: random.Random, places: list[dict], split: str,
                      per_intent: int, builder: Builder) -> None:
    """The turns that never reach the weather API.

    weather_intent is NONE for all of them, OUT_OF_RANGE included: "weather in December" names
    a window, but we decline before resolving one, and FORECAST would promise a fetch.
    """
    for intent, frames in CHITCHAT_FRAMES.items():
        made = 0
        for _ in range(per_intent * 8):
            if made >= per_intent:
                break
            frame = rng.choice(frames)
            if "{loc}" in frame:
                place = surface(rng, rng.choice(places))
                text = _clean(frame.replace("{loc}", rng.choice(PLACE_FORMS).format(loc=place)))
                spans = [place]
            else:
                text, spans = frame, []
            text, spans = _capitalise(_noise(rng, text, spans), spans)
            made += builder.add(
                text, spans, chat_id=f"{split[:2]}-chat-{intent.value.lower()}-{made:04d}",
                turn=0, intent=intent, weather_intent=WeatherIntent.NONE, variables=[],
                activity=Activity.NONE, aggregation=Aggregation.RAW, locations=spans,
                times=[], operation=Operation.SET, ctx_locations=spans, ctx_times=[],
                split=split, source="chitchat")


def generate_conversations(rng: random.Random, places: list[dict], split: str,
                           count: int, builder: Builder) -> None:
    """Multi-turn chats: a full question, then fragments that inherit from it.

    The fragments are where v3 duplicated worst, so they obey MAX_PER_SKELETON like everything
    else.
    """
    for chat in range(count):
        place = surface(rng, rng.choice(places))
        time_span = rng.choice(TIMES[rng.choice(WINDOWS)])
        weather_intent = weather_intent_for(normalize_time(time_span))
        variables = [rng.choice(list(Variable))]
        chat_id = f"{split[:2]}-chat-{chat:05d}"

        opening = rng.choice(INFORMATION_FRAMES).replace(
            "{v}", rng.choice(VARIABLE_WORDS[variables[0]]))
        text, locations, times = _phrase(rng, place, time_span, opening)
        text, spans = _capitalise(_noise(rng, text, locations + times), locations + times)
        locations, times = spans[:len(locations)], spans[len(locations):]
        builder.add(text, locations + times,
                    chat_id=chat_id, turn=0, intent=Intent.INFORMATION,
                    weather_intent=weather_intent, variables=variables,
                    activity=Activity.NONE, aggregation=Aggregation.RAW, locations=locations,
                    times=times, operation=Operation.SET, ctx_locations=locations,
                    ctx_times=times, split=split, source="chats")

        context_places, context_times = list(locations), list(times)
        for turn in range(1, rng.randint(2, 4)):
            operation = rng.choice(list(FOLLOW_FRAMES))
            frame = rng.choice(FOLLOW_FRAMES[operation])
            turn_intent = weather_intent
            if operation is Operation.REPLACE:
                new_place = surface(rng, rng.choice(places))
                fragment, spans = frame.format(loc=new_place), [new_place]
            elif operation is Operation.MODIFY:
                new_time = rng.choice(TIMES[rng.choice(WINDOWS)])
                fragment, spans = frame.format(t=new_time), [new_time]
                # "what about tomorrow?" moves the window; the label has to move with it
                turn_intent = weather_intent_for(normalize_time(new_time))
            else:
                fragment, spans = frame, []
            fragment, spans = _capitalise(fragment + ("?" if rng.random() < 0.6 else ""), spans)
            # context comes from the *final* spans: "{t}" alone becomes "Tomorrow", and a
            # context still holding "tomorrow" would drop the span from the annotation
            if operation is Operation.REPLACE:
                context_places = list(spans)
            elif operation is Operation.MODIFY:
                context_times = list(spans)
            builder.add(
                fragment, spans,
                chat_id=chat_id, turn=turn, intent=Intent.INFORMATION,
                weather_intent=turn_intent, variables=variables, activity=Activity.NONE,
                aggregation=Aggregation.RAW,
                locations=[s for s in spans if s in context_places],
                times=[s for s in spans if s in context_times],
                operation=operation, ctx_locations=context_places, ctx_times=context_times,
                split=split, source="chats")


# Import shim only: the seed files predate the activity column, so it is read back out of the
# sentence. A seed matching no cue stays INFORMATION rather than being guessed at.
ACTIVITY_CUES = {
    Activity.RAIN_PROTECTION: ("umbrella", "raincoat", "rain gear", "get wet", "getting wet"),
    Activity.SUN_PROTECTION: ("sunscreen", "sunblock", "sun protection", "sunburn", "a cap",
                              "shade"),
    Activity.CLOTHING: ("jacket", "sweater", "what to wear", "should i wear", "warm clothes",
                        "woollens", "dress warm"),
    Activity.DRYING: ("clothes dry", "washing", "clothes in", "drying", "laundry",
                      "wash the", "wash my", "clean the", "air out"),
    Activity.OUTDOOR_ACTIVITY: ("go out", "head out", "step out", "outside", "outdoors",
                                "cricket", "football", "match", "game", "walk", "jog",
                                "gardening", "garden", "concrete", "masons", "site"),
    Activity.TRAVEL: ("travel", "trip", "journey", "drive", "commute", "to work", "office",
                      "roads"),
    Activity.FERTILIZE: ("fertiliz", "fertilis", "urea", "manure", "top dressing"),
    Activity.SPRAY: ("spray", "pesticide", "insecticide", "fungicide"),
    Activity.IRRIGATE: ("irrigat", "water the", "watering", "run the pump", "needs water"),
    Activity.HARVEST: ("harvest", "cut it", "cut the"),
    Activity.SOW: ("sow", "sowing", "plant the", "planting", "seed"),
}


def activity_from_seed(text: str, annotated: str = "") -> Activity:
    """The annotated column wins; the cue scan is the fallback for seeds that predate it,
    where the longest cue wins so "wash the car" is not read as a walk."""
    try:
        return Activity(annotated.strip())
    except ValueError:
        pass
    lowered = text.lower()
    best, best_len = Activity.NONE, 0
    for activity, cues in ACTIVITY_CUES.items():
        for cue in cues:
            if cue in lowered and len(cue) > best_len:
                best, best_len = activity, len(cue)
    return best


def load_seeds(builder: Builder, split: str) -> int:
    """The hand-written rows. Templates cannot evaluate templates, and they cannot invent a
    phrasing either - these are the sentences no frame produced."""
    added = 0
    for path, intent in ((SEED_CSV, Intent.INFORMATION), (ADVICE_SEED_CSV, Intent.ADVICE)):
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for record in csv.reader(handle):
                if len(record) < 6 or record[0].lower() == "text":
                    continue
                text = record[0].strip()
                try:
                    locations = [l for l in json.loads(record[4] or "[]") if l in text]
                    times = [t for t in json.loads(record[5] or "[]") if t in text]
                except (ValueError, IndexError):
                    continue
                # "soon and back by evening" has two spans and only the second buckets - take
                # the first that does, so the label is not lost to span order
                spans = [normalize_time(t) for t in times]
                normalized = next((s for s in spans if bucket_for(s) is not TimeBucket.NONE),
                                  spans[0] if spans else None)
                activity = (activity_from_seed(text, record[1]) if intent is Intent.ADVICE
                            else Activity.NONE)
                # an ADVICE seed with no readable activity is still a good INFORMATION row
                row_intent = Intent.INFORMATION if activity is Activity.NONE else intent
                variables = (ACTIVITY_VARIABLES.get(activity) or [Variable.GENERAL])
                added += builder.add(
                    text, locations + times, chat_id=f"seed-{added:05d}", turn=0,
                    intent=row_intent, weather_intent=weather_intent_for(normalized),
                    variables=variables, activity=activity,
                    aggregation=Aggregation.RAW, locations=locations, times=times,
                    operation=Operation.SET, ctx_locations=locations, ctx_times=times,
                    split=split, source="seed")
    return added


def build(path: Path = CSV_PATH, per_cell: int = 700, conversations: int = 500,
          seed: int = 13) -> dict:
    # data/v4_dataset.csv is generated; ./v4_dataset.csv is hand-written and unrecoverable
    if ADVICE_SEED_CSV.exists() and path.resolve() == ADVICE_SEED_CSV.resolve():
        raise SystemExit(f"refusing to overwrite the hand-written seeds at {path}")
    pools = load_locations()
    rng = random.Random(seed)
    all_rows: list[dict] = []
    report: dict = {}

    for split, places, cells, chats in (("train", pools["train"], per_cell, conversations),
                                        ("test", pools["train"], per_cell // 5, conversations // 5),
                                        ("eval", pools["eval"], per_cell // 8, conversations // 10)):
        builder = generate(random.Random(seed + len(split)), places, split, cells)
        generate_chitchat(rng, places, split, max(cells // 4, 20), builder)
        generate_conversations(rng, places, split, chats, builder)
        if split == "train":
            report["seed rows"] = load_seeds(builder, split)
        all_rows += builder.rows
        report[f"{split} rows"] = len(builder.rows)
        report[f"{split} rejected"] = dict(builder.rejected)

    # train and test share a location pool, so a text generated for both must not leak
    seen, deduped = set(), []
    for row in all_rows:
        if row["text"].lower() in seen:
            continue
        seen.add(row["text"].lower())
        deduped.append(row)
    report["cross-split duplicates dropped"] = len(all_rows) - len(deduped)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(deduped)
    report["rows"] = len(deduped)
    return report


def load(path: Path | str = CSV_PATH, split: str | None = None,
         source: str | None = None) -> list[dict]:
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            if (split and record["split"] != split) or (source and record["source"] != source):
                continue
            rows.append({
                **record,
                "turn": int(record["turn"]),
                "variables": [v for v in record["variables"].split("|") if v],
                "entities": json.loads(record.get("entities") or "{}"),
                "sub_activity": record.get("sub_activity", ""),
                "para_group": record.get("para_group", ""),
                "locations": json.loads(record["locations"]),
                "times": json.loads(record["times"]),
                "times_normalized": json.loads(record["times_normalized"]),
                "ctx_locations": json.loads(record["ctx_locations"]),
                "ctx_times": json.loads(record["ctx_times"]),
            })
    return rows


def shortcut_rate(rows: list[dict]) -> tuple[float, dict]:
    """Share of ADVICE rows that name a keyword for their own activity - whether the model can
    skip reading the sentence. One such row is fine; a set of them teaches keyword lookup."""
    advice = [r for r in rows if r["intent"] == "ADVICE"]
    if not advice:
        return 0.0, {}
    per = defaultdict(lambda: [0, 0])
    for row in advice:
        cues = ACTIVITY_CUES.get(Activity(row["activity"]), ())
        hit = any(cue in row["text"].lower() for cue in cues)
        per[row["activity"]][0] += hit
        per[row["activity"]][1] += 1
    total = sum(v[0] for v in per.values()) / len(advice)
    return total, {a: round(h / n, 2) for a, (h, n) in sorted(per.items(),
                                                              key=lambda kv: -kv[1][0] / kv[1][1])}


def entity_leak(rows: list[dict]) -> tuple[float, list[str]]:
    """Entity terms that only ever appear with one activity - a free shortcut to the label."""
    seen, per_type = defaultdict(set), defaultdict(set)
    for row in rows:
        if row["intent"] != "ADVICE":
            continue
        for kind, terms in row["entities"].items():
            per_type[kind].add(row["activity"])
            for term in terms:
                seen[f"{kind}={term.lower()}"].add(row["activity"])
    # A sport only ever belongs to OUTDOOR_ACTIVITY - ontology, not a leak.
    crossable = {kind for kind, acts in per_type.items() if len(acts) > 1}
    scored = {k: v for k, v in seen.items() if k.split("=")[0] in crossable}
    if not scored:
        return 0.0, []
    solo = sorted(k for k, v in scored.items() if len(v) == 1)
    return len(solo) / len(scored), solo


def redundancy(rows: list[dict]) -> tuple[float, int]:
    """Share of rows that belong to a paraphrase group, and how many groups there are."""
    grouped = [r for r in rows if r.get("para_group")]
    return len(grouped) / max(len(rows), 1), len({r["para_group"] for r in grouped})


def stats(rows: list[dict]) -> dict:
    texts = [r["text"].lower() for r in rows]
    shapes = Counter(skeleton(r["text"], r["locations"] + r["times"]) for r in rows)
    names = {name for r in rows for name in r["locations"]}
    # the cap is per split, so this is reported per split too
    per_split = Counter()
    for row in rows:
        per_split[(row["split"], skeleton(row["text"], row["locations"] + row["times"]))] += 1
    return {
        "rows": len(rows),
        "unique texts": f"{len(set(texts))} ({len(set(texts)) / max(len(rows), 1):.1%})",
        "distinct templates": len(shapes),
        "rows per template": round(len(rows) / max(len(shapes), 1), 2),
        "busiest template": (per_split.most_common(1)[0] if per_split else None),
        "distinct location surfaces": len(names),
        "intent": dict(Counter(r["intent"] for r in rows)),
        "weather_intent": dict(Counter(r["weather_intent"] for r in rows)),
        "action": dict(Counter(r["action"] for r in rows)),
        "activity": len({r["activity"] for r in rows}),
        "sub_activity": dict(Counter(r["sub_activity"] for r in rows if r["sub_activity"]).most_common(8)),
        "entity types": dict(Counter(kind for r in rows for kind in r["entities"])),
        "distinct entity terms": len({t for r in rows for ts in r["entities"].values() for t in ts}),
        "variables": dict(Counter(v for r in rows for v in r["variables"]).most_common()),
        "aggregation": dict(Counter(r["aggregation"] for r in rows)),
        "operation": dict(Counter(r["operation"] for r in rows)),
        "splits": dict(Counter(r["split"] for r in rows)),
        "shortcut rate": f"{shortcut_rate(rows)[0]:.1%} name their own cue",
        "thinnest keyword-free": {a: f"{1 - h:.0%}" for a, h in
                                  list(shortcut_rate(rows)[1].items())[:4]},
        "entity leak": f"{entity_leak(rows)[0]:.1%} of terms map to one activity",
        "redundancy": "{:.1%} of rows in {} paraphrase groups".format(*redundancy(rows)),
    }


def _by_group(rows: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for row in rows:
        if row.get("para_group"):
            groups[row["para_group"]].append(row)
    return groups


def check(rows: list[dict]) -> None:
    """The invariants that make this dataset worth training on. Fails loudly, not silently."""
    assert rows, "no rows - run --build first"
    texts = [r["text"].lower() for r in rows]
    assert len(set(texts)) == len(texts), f"{len(texts) - len(set(texts))} duplicate texts"

    # the cap is enforced per split, so that is where it has to hold
    for split in {r["split"] for r in rows}:
        shapes = Counter(skeleton(r["text"], r["locations"] + r["times"])
                         for r in rows if r["split"] == split)
        worst, count = shapes.most_common(1)[0]
        assert count <= MAX_PER_SKELETON, f"{split}: template over cap ({count}): {worst!r}"

    entity_spans = lambda row: [t for terms in row["entities"].values() for t in terms]
    for row in rows:                                    # Rule 4.1: spans verbatim in the text
        for span in row["locations"] + row["times"] + entity_spans(row):
            assert span in row["text"], f"span {span!r} not in {row['text']!r}"

    # the gazetteer has to recover what the generator inserted, or ENTITY_VOCAB and the
    # frames have drifted apart and the runtime will see entities the training data never had
    for row in rows:
        if not row["entities"]:
            continue
        found = extract_entities(row["text"])
        for kind, terms in row["entities"].items():
            got = {t.lower() for t in found.get(kind, [])}
            missing = {t.lower() for t in terms} - got
            assert not missing, f"gazetteer missed {missing} ({kind}) in {row['text']!r}"

    for row in rows:                                    # derived labels must agree
        expected = group_for(row["activity"]).value
        assert row["action"] == expected, f"{row['activity']} -> {row['action']} != {expected}"
        if row["intent"] == "ADVICE":
            assert row["activity"] != "NONE", f"ADVICE with no activity: {row['text']!r}"
        else:
            assert row["activity"] == "NONE", f"{row['intent']} with activity: {row['text']!r}"

    for row in rows:               # a turn we never fetch for must not claim a time window
        if row["intent"] in {i.value for i in NO_DATA_NEEDED}:
            assert row["weather_intent"] == "NONE", f"{row['intent']} with a window: {row['text']!r}"
            assert not row["variables"], f"{row['intent']} with variables: {row['text']!r}"
            assert not row["times"], f"{row['intent']} with a time span: {row['text']!r}"
        else:
            assert row["weather_intent"] != "NONE", f"weather turn with no window: {row['text']!r}"

    for (activity, kind), terms in ENTITY_SUBSETS.items():
        stray = set(terms) - set(ENTITY_VOCAB[kind])
        assert not stray, f"{activity.value}/{kind.value} subset outside ENTITY_VOCAB: {stray}"

    replies = {i.value for i in NO_DATA_NEEDED} | {Intent.GREETING.value}
    for intent in replies:                              # every declined label needs an answer
        assert REPLIES.get(Intent(intent)), f"{intent} has no reply text"

    per_split = defaultdict(set)
    for row in rows:
        per_split[row["split"]].add(row["text"].lower())
    for a, b in (("train", "test"), ("train", "eval"), ("test", "eval")):
        overlap = per_split[a] & per_split[b]
        assert not overlap, f"{len(overlap)} texts shared between {a} and {b}"

    rate, per_activity = shortcut_rate(rows)
    thin = {a: 1 - hit for a, hit in per_activity.items() if 1 - hit < MIN_IMPLICIT_SHARE}
    assert not thin, (
        f"these activities have under {MIN_IMPLICIT_SHARE:.0%} keyword-free ADVICE rows, so "
        f"the model can read their label off one word: "
        f"{ {a: f'{v:.0%}' for a, v in thin.items()} }. Add IMPLICIT_FRAMES for them.")

    leak, solo = entity_leak(rows)
    assert leak <= MAX_ENTITY_LEAK, (
        f"{leak:.1%} of entity terms appear with exactly one activity (ceiling "
        f"{MAX_ENTITY_LEAK:.0%}) - e.g. {solo[:5]}. Add CONFUSION_FRAMES that cross them.")

    share, groups = redundancy(rows)
    assert 0.08 <= share <= 0.14, (
        f"paraphrase redundancy {share:.1%} outside the 8-14% band ({groups} groups) - "
        f"tune REDUNDANCY")
    for group, members in _by_group(rows).items():
        labels = {(r["intent"], r["weather_intent"], r["activity"], r["variables"] and
                   tuple(r["variables"])) for r in members}
        assert len(labels) == 1, f"paraphrase group {group} disagrees on labels: {labels}"

    counts = Counter(r["weather_intent"] for r in rows)
    rarest = min(counts.values()) / len(rows)
    assert rarest >= 0.05, f"weather_intent is unbalanced, rarest cell {rarest:.1%}: {counts}"
    print(f"OK {len(rows)} rows | all texts unique | no template over {MAX_PER_SKELETON} | "
          f"spans verbatim | splits disjoint | rarest weather_intent {rarest:.1%}\n"
          f"   shortcut {rate:.1%} (every activity >={MIN_IMPLICIT_SHARE:.0%} keyword-free) | "
          f"entity leak {leak:.1%} "
          f"(<={MAX_ENTITY_LEAK:.0%}) | redundancy {share:.1%} in {groups} groups")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=str(CSV_PATH))
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--per-cell", type=int, default=700)
    parser.add_argument("--conversations", type=int, default=500)
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--samples", type=int, metavar="N")
    args = parser.parse_args()
    path = Path(args.path)

    if args.build:
        for key, value in build(path, args.per_cell, args.conversations).items():
            print(f"  {key:30s} {value}")
    rows = load(path)
    if args.stats or args.build:
        print()
        for key, value in stats(rows).items():
            print(f"  {key:26s} {value}")
    if args.check or args.build:
        print()
        check(rows)
    if args.samples:
        for row in random.Random(5).sample(rows, min(args.samples, len(rows))):
            print(f"\n  {row['text']}")
            print(f"    intent={row['intent']:12s} weather={row['weather_intent']:10s} "
                  f"activity={row['activity']:12s} action={row['action']}")
            print(f"    variables={row['variables']} loc={row['locations']} "
                  f"time={row['times']} -> {row['times_normalized']} agg={row['aggregation']}")


if __name__ == "__main__":
    main()
