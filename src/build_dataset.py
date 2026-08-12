"""
Builds a balanced WeatherBot NLU training set from the hand-written seed
(data/intents.csv) plus templated generation.

LOCATION vocabulary comes from data/locations.csv - real village / block / district / state
names sampled read-only from the `shapes` schema by src/fetch_locations.py, >=80% of them
inside India. Without that CSV the build falls back to a built-in city list.

Enforces MODEL_RULES.md:
  - labels restricted to the 14 WeatherIntent / 3 Action enums in schema.py
  - LOCATION / TIME are raw text spans that must appear verbatim in the prompt
  - equal rows per (weather_intent, action) cell  (Rule 5.2)
  - varied syntactic order: question-first / location-first / time-first  (Rule 5.3)

Two generated splits, balanced per (intent, action) cell and disjoint by prompt text:
  --split train  data/processed/nlu_dataset.csv  model fitting        (~6300 rows)
  --split test   data/processed/nlu_test.csv     in-distribution held-out; harvest failures,
                                                 feed them back into training, retrain

The evaluation set is NOT generated: data/eval_manual.csv is hand-written, covering typos,
code-mixing, ellipsis and other edge cases no template produces. Templates cannot evaluate
templates, so keep that file authored by hand.

Usage:  python src/build_dataset.py [--split train|test] [--per-cell N] [--out PATH]
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.schema import Action, WeatherIntent

# --- vocabulary -------------------------------------------------------------

# Real place names come from data/locations.csv (villages / blocks / districts / states
# sampled read-only from the `shapes` schema by src/fetch_locations.py). The list below is
# only the fallback for when that CSV is absent, so a build never requires DB access.
# (surface text inserted into the prompt, annotated span inside that text)
FALLBACK_LOCATIONS = [(x, x) for x in [
    "Hyderabad", "Vizag", "Visakhapatnam", "Rajahmundry", "Guntur", "Vijayawada",
    "Warangal", "Tirupati", "Nellore", "Kurnool", "Anantapur", "Karimnagar",
    "Khammam", "Nizamabad", "Ongole", "Kakinada", "Chennai", "Bengaluru",
    "Mumbai", "Delhi", "Kolkata", "Pune", "Coimbatore", "Kochi",
    "AP", "Andhra Pradesh", "Telangana", "Karnataka", "Tamil Nadu", "Kerala",
    "London", "New York", "Tokyo", "Dubai", "Singapore", "Sydney", "Toronto",
]]

# Relative locations (Rule 4.1). Surface carries its own preposition, so these are
# only used in BARE_* frames where {loc} is not preceded by "in"/"for".
RELATIVE = [
    ("near me", "near me"), ("here", "here"), ("around here", "here"),
    ("at my location", "my location"), ("in this area", "this area"),
    ("in my field", "my field"), ("in my farm", "my farm"), ("in my village", "my village"),
    ("in my plot", "my plot"), ("at my place", "my place"), ("nearby", "nearby"),
    ("around my area", "my area"), ("in our village", "our village"),
    # misspelt as users type them; the span stays verbatim, typos included
    ("in my feild", "my feild"), ("near by", "near by"), ("in my vilage", "my vilage"),
]

# Calendar-relative expressions. Wall-clock times are generated instead (see _clock_pool).
TIMES = [
    ("", ""),  # no temporal mention -> time == []
    ("now", "now"), ("right now", "right now"), ("today", "today"),
    ("tonight", "tonight"), ("this morning", "this morning"),
    ("this afternoon", "this afternoon"), ("this evening", "this evening"),
    ("tomorrow", "tomorrow"), ("tomorrow morning", "tomorrow morning"),
    ("tomorrow evening", "tomorrow evening"), ("tomorrow night", "tomorrow night"),
    ("day after tomorrow", "day after tomorrow"), ("this weekend", "this weekend"),
    ("next week", "next week"), ("next month", "next month"),
    ("on Friday", "Friday"), ("on Monday", "Monday"), ("on Sunday", "Sunday"),
    ("on Tuesday", "Tuesday"), ("on Wednesday", "Wednesday"), ("on Saturday", "Saturday"),
    ("for the next 3 days", "next 3 days"),
    ("for the next 7 days", "next 7 days"),
    ("during the sowing week", "sowing week"),
    ("this week", "this week"), ("next weekend", "next weekend"),
    ("later today", "later today"), ("early morning", "early morning"),
    ("by midnight", "midnight"), ("yesterday", "yesterday"), ("last week", "last week"),
    # how people actually type dates and relative days
    ("tomorow", "tomorow"), ("tommorow", "tommorow"), ("tmrw", "tmrw"), ("2day", "2day"),
    ("rn", "rn"), ("2nite", "2nite"), ("nxt week", "nxt week"),
    ("day after tomorow", "day after tomorow"),
]
NOUNS = {
    # Existing intents...

    "CURRENT_CONDITIONS": [
        "weather", "weather conditions", "current conditions", "weather update",
        "weather situation", "weather status", "mausam", "vaatavaranam",
        "wether", "wheather", "climate right now", "conditions right now",
        "outside weather", "outside conditions", "how is the weather",
        "what's the weather like", "weather now"
    ],

    "FORECAST": [
        "forecast", "weather forecast", "weather outlook", "extended forecast",
        "weather prediction", "outlook", "week ahead forecast", "weather report",
        "forcast", "forecaste", "prediction", "predictions",
        "future weather", "upcoming weather", "upcoming forecast",
        "weather for next days", "weather ahead", "weekly weather",
        "daily forecast", "long range forecast"
    ],

    "TEMPERATURE": [
        "temperature", "temp", "air temperature", "heat level", "how hot it is",
        "taapmaan", "temprature", "tempreture", "tempature",
        "degrees", "degree", "how hot", "how cold", "heat",
        "cold", "current temp", "temperature now", "actual temperature",
        "feels like temperature", "what temperature"
    ],

    "TEMPERATURE_MIN": [
        "minimum temperature", "min temp", "lowest temperature", "overnight low",
        "night temperature", "coldest temperature", "low temp",
        "minimum temprature", "min temprature",
        "lowest temp", "lowest temperature today", "night low",
        "nighttime low", "overnight temperature", "coldest",
        "minimum", "min temperature"
    ],

    "TEMPERATURE_MAX": [
        "maximum temperature", "max temp", "highest temperature", "daytime high",
        "peak temperature", "hottest temperature", "high temp",
        "maximum temprature", "max tempreture",
        "highest temp", "highest temperature today", "day high",
        "daytime temperature", "hottest", "maximum", "max temperature"
    ],

    "RAIN": [
        "rain", "rainfall", "precipitation", "chance of rain", "showers",
        "downpour", "rain chance", "barish", "vaana", "rian", "rainfal",
        "precipitaion", "rain probability", "probability of rain",
        "rain forecast", "will it rain", "is it going to rain",
        "raining", "rainy", "drizzle", "thunderstorm", "storm",
        "heavy rain", "light rain", "rain intensity", "rain amount",
        "precipitation amount", "precipitation chance"
    ],

    "HUMIDITY": [
        "humidity", "relative humidity", "humidity level", "humidity percentage",
        "moisture in the air", "rh", "nami", "humidty", "humdity", "humididty",
        "air moisture", "moisture level", "humidity percent",
        "how humid", "humid", "relative moisture"
    ],

    "DEW_POINT": [
        "dew point", "dewpoint", "dew point temperature", "dew", "dew level",
        "os bindu", "dewpoit", "due point",
        "dew temperature", "dewpoint temperature", "dew point temp"
    ],

    "WIND_SPEED": [
        "wind speed", "wind", "wind strength", "wind velocity", "breeze",
        "gusts", "hawa ki raftar", "wnd speed", "wind speeed",
        "wind force", "air speed", "wind rate", "wind intensity",
        "how windy", "windy", "strong wind", "wind speed forecast",
        "gust speed", "gust", "wind gusts"
    ],

    "WIND_DIRECTION": [
        "wind direction", "wind heading", "direction of the wind",
        "which way the wind blows", "wind bearing", "hawa ki disha",
        "wnd direction", "wind directon",
        "wind flow direction", "where wind is coming from",
        "wind comes from", "wind blows from", "wind bearing",
        "direction wind"
    ],

    "SUNSHINE": [
        "sunshine", "sunshine hours", "sunlight", "solar exposure", "sun",
        "hours of sun", "sunny hours", "dhoop", "sunshien", "sunlite",
        "sun hours", "sun duration", "sunlight hours", "solar radiation",
        "solar exposure", "bright sunshine", "sunny", "hours of sunlight"
    ],

    "CLOUD_COVER": [
        "cloud cover", "cloudiness", "cloud percentage", "clouds", "sky cover",
        "overcast level", "badal", "cloud covr", "cloudness",
        "cloud cover percentage", "cloud percent", "cloud amount",
        "how cloudy", "cloudy", "overcast", "sky condition",
        "cloud density", "cloud coverage"
    ],

    "SOIL_MOISTURE": [
        "soil moisture", "soil moisture at 10cm", "moisture in the soil",
        "field moisture", "soil water", "soil wetness", "moisture at 40cm",
        "mitti ki nami", "soil moistur", "soil moisure",
        "soil moisture at 10 m", "soil moisture 10cm",
        "soil moisture 40cm", "soil moisture at 40 m",
        "ground moisture", "soil water content",
        "soil wetness level", "moisture in field"
    ],

    "SOIL_TEMPERATURE": [
        "soil temperature", "soil temp", "ground temperature",
        "soil temperature at 10cm", "earth temperature", "soil heat",
        "mitti ka taapmaan", "soil temprature", "soil tempreture",
        "ground temp", "soil temperature 10cm", "soil temperature at 40cm",
        "soil heat level", "ground heat", "earth temperature",
        "temperature of soil"
    ],
}

# Generic frames. {m}=metric noun phrase, {loc}/{loc2}=location, {t}/{t2}=time.
# Ordering is deliberately mixed (Rule 5.3).
GET_FRAMES = [
    "what is the {m} in {loc} {t}?",
    "what's the {m} in {loc} {t}?",
    "tell me the {m} for {loc} {t}",
    "give me the {m} for {loc} {t}",
    "show me the {m} in {loc} {t}",
    "check the {m} in {loc} {t}",
    "i want to know the {m} in {loc} {t}",
    "can you tell me the {m} in {loc} {t}?",
    "how is the {m} in {loc} {t}?",
    "{m} in {loc} {t}",
    "{loc} {m} {t}",
    "in {loc}, what is the {m} {t}?",
    "{t}, what is the {m} in {loc}?",
    "what will the {m} be in {loc} {t}?",
    "what is the {m} {t} in {loc}?",
    "any idea about the {m} in {loc} {t}?",
    "could you let me know the {m} in {loc} {t}?",
    "i need the {m} reading for {loc} {t}",
    "whats the {m} looking like in {loc} {t}?",
    "do you have the {m} data for {loc} {t}?",
    "update me on the {m} in {loc} {t}",
    "regarding {loc}, what is the {m} {t}?",
    "{t} {m} figures for {loc}",
    "pls tell me the {m} in {loc} {t}",
    "need {m} for {loc} {t}",
    "{m} details for {loc} {t}",
    "how much {m} in {loc} {t}?",
    "kindly share the {m} for {loc} {t}",
    "whats up with the {m} in {loc} {t}?",
    "quick {m} check for {loc} {t}",
    "wat is the {m} in {loc} {t}?",
    "whts {m} in {loc} {t}",
    "tel me the {m} for {loc} {t}",
    "hw is {m} in {loc} {t}",
    "{m} {loc} {t} pls",
    "give {m} {loc} {t}",
    "want to know {m} {loc} {t}",
    "{loc} ka {m} {t}",
    "{loc} {t} {m}?",
    "check {m} {loc} {t}",
]
NOLOC_GET_FRAMES = [
    "what is the {m} {t}?",
    "what's the {m} {t}?",
    "tell me the {m} {t}",
    "how is the {m} {t}?",
    "{m} {t}",
    "{t}, what is the {m}?",
    "check the {m} {t}",
    "i need the {m} reading {t}",
    "whats the {m} looking like {t}?",
    "update me on the {m} {t}",
    "how much {m} {t}?",
    "any idea about the {m} {t}?",
    "{m} details {t}",
    "wat is {m} {t}",
    "{m} {t} pls",
    "tel me {m} {t}",
    "need to know {m} {t}",
]
COMPARE_FRAMES = [
    "compare the {m} between {loc} and {loc2} {t}",
    "compare {m} in {loc} vs {loc2} {t}",
    "{loc} vs {loc2} {m} {t}",
    "difference in {m} between {loc} and {loc2} {t}",
    "show me the {m} for {loc} and {loc2} {t}",
    "{t}, compare the {m} in {loc} and {loc2}",
    "how does the {m} in {loc} compare with {loc2} {t}?",
    "contrast the {m} in {loc} with {loc2} {t}",
    "put the {m} of {loc} against {loc2} {t}",
    "{m} gap between {loc} and {loc2} {t}",
    "for {loc} and {loc2} both, what is the {m} {t}?",
    "{m} in {loc} and {loc2} {t}, which is better?",
    "check the {m} for both {loc} and {loc2} {t}",
    "between {loc} and {loc2}, where is the {m} stronger {t}?",
    "compair the {m} between {loc} and {loc2} {t}",
    "compre {m} in {loc} vs {loc2} {t}",
    "{loc} verses {loc2} {m} {t}",
    "diffrence in {m} between {loc} and {loc2} {t}",
    "how duz the {m} in {loc} compare with {loc2} {t}?",
    "wich has more {m} {loc} or {loc2} {t}",
    "{m} {loc} vs {loc2} {t}",
    "compare {m} {loc} {loc2} {t}",
    "tell diffrence {m} {loc} {loc2} {t}",
    "{loc} or {loc2} better {m} {t}?",
    "{loc} vs {loc2} {m}",
    "{loc} v/s {loc2} {m} {t}",
    "{m} {loc} or {loc2}",
    "compare {loc} and {loc2} {m}",
    "{loc} and {loc2} {m} {t}",
    "which is better for {m}, {loc} or {loc2}?",
    "{loc} versus {loc2} {m} {t}",
]
COMPARE_TIME_FRAMES = [  # same location, two temporal spans
    "compare the {m} in {loc} between {t} and {t2}",
    "how does the {m} in {loc} {t} compare to {t2}?",
    "{loc} {m} {t} vs {t2}",
    "contrast the {m} in {loc} between {t} and {t2}",
    "{m} in {loc}: {t} or {t2}?",
    "is the {m} in {loc} different {t} and {t2}?",
]
COMPARE_SCALAR_FRAMES = [  # "higher/more" only makes sense for measurable quantities
    "which has higher {m}, {loc} or {loc2} {t}?",
    "is the {m} higher in {loc} or in {loc2} {t}?",
    "where is the {m} greater, {loc} or {loc2} {t}?",
    "is the {m} in {loc} higher {t} or {t2}?",
]
NON_SCALAR = {"CURRENT_CONDITIONS", "FORECAST", "WIND_DIRECTION"}
ALERT_FRAMES = [
    "set an alert for {m} in {loc} {t}",
    "notify me about the {m} in {loc} {t}",
    "is there any {m} warning for {loc} {t}?",
    "warn me about the {m} in {loc} {t}",
    "send me an alert if the {m} is unusual in {loc} {t}",
    "any warnings about {m} in {loc} {t}?",
    "let me know if there is a {m} alert in {loc} {t}",
    "{t}, alert me about the {m} in {loc}",
    "remind me to check the {m} in {loc} {t}",
    "ping me about the {m} in {loc} {t}",
    "keep me posted on the {m} in {loc} {t}",
    "raise an alarm if the {m} looks bad in {loc} {t}",
    "is any {m} advisory issued for {loc} {t}?",
    "flag the {m} for {loc} {t}",
    "message me the {m} in {loc} {t}",
    "put a reminder for {m} in {loc} {t}",
    "should i worry about the {m} in {loc} {t}?",
    "alrt me if {m} changes in {loc} {t}",
    "notifiy me about the {m} in {loc} {t}",
    "warrn me about {m} in {loc} {t}",
    "set an alart for {m} in {loc} {t}",
    "plz alert {m} {loc} {t}",
    "send alert {m} in {loc} {t}",
    "{m} alert {loc} {t}",
    "want alert for {m} in {loc} {t}",
    "inform me if {m} bad in {loc} {t}",
    "reminder for {m} {loc} {t}",
    "tell me when {m} changes {loc} {t}",
]
NOLOC_ALERT_FRAMES = [
    "set an alert for {m} {t}",
    "notify me about the {m} {t}",
    "is there any {m} warning {t}?",
    "warn me about the {m} {t}",
    "ping me about the {m} {t}",
    "flag the {m} {t}",
    "is any {m} advisory issued {t}?",
]
# {loc} without a leading preposition -> used with RELATIVE locations.
BARE_GET_FRAMES = [
    "what is the {m} {loc} {t}?",
    "what's the {m} {loc} {t}?",
    "tell me the {m} {loc} {t}",
    "how is the {m} {loc} {t}?",
    "check the {m} {loc} {t}",
    "{m} {loc} {t}",
    "{t}, what is the {m} {loc}?",
    "could you check the {m} {loc} {t}?",
    "update me on the {m} {loc} {t}",
    "{t} {m} figures {loc}",
    "how much {m} {loc} {t}?",
    "wat is {m} {loc} {t}",
    "{m} {loc} {t} pls",
    "tel me {m} {loc} {t}",
]
BARE_ALERT_FRAMES = [
    "set an alert for {m} {loc} {t}",
    "notify me about the {m} {loc} {t}",
    "is there any {m} warning {loc} {t}?",
    "warn me about the {m} {loc} {t}",
    "ping me about the {m} {loc} {t}",
    "flag the {m} {loc} {t}",
]
BARE_COMPARE_FRAMES = [
    "compare the {m} {loc} with {loc2} {t}",
    "how does the {m} {loc} compare with {loc2} {t}?",
    "contrast the {m} {loc} with {loc2} {t}",
    "put the {m} {loc} against {loc2} {t}",
]

# Intent-specific natural phrasings the generic frames cannot express.
EXTRA = {
    "CURRENT_CONDITIONS": {"GET": ["how is it outside in {loc} {t}?", "what is it like in {loc} {t}?"],
                           "ALERT": ["is there a weather warning for {loc} {t}?",
                                     "alert me if the weather turns bad in {loc} {t}"]},
    "FORECAST": {"GET": ["what will the weather be like in {loc} {t}?", "what is coming up weather wise in {loc} {t}?"],
                 "ALERT": ["alert me if the forecast changes for {loc} {t}",
                           "is there a severe weather forecast for {loc} {t}?"]},
    "TEMPERATURE": {"GET": ["how hot is it in {loc} {t}?", "how cold is it in {loc} {t}?"],
                    "ALERT": ["alert me if the temperature crosses 40 degrees in {loc} {t}"]},
    "TEMPERATURE_MIN": {"GET": ["how low will the temperature drop in {loc} {t}?"],
                        "ALERT": ["warn me if the temperature falls below 10 degrees in {loc} {t}"]},
    "TEMPERATURE_MAX": {"GET": ["how high will the temperature go in {loc} {t}?"],
                        "ALERT": ["alert me if the temperature goes above 42 degrees in {loc} {t}"]},
    "RAIN": {"GET": ["will it rain in {loc} {t}?", "is it going to rain in {loc} {t}?",
                     "do i need an umbrella in {loc} {t}?", "how much rain will {loc} get {t}?",
                     "any chance of showers in {loc} {t}?", "should i cover the harvest in {loc} {t}?"],
             "ALERT": ["alert me if it rains in {loc} {t}", "notify me if it starts raining in {loc} {t}",
                       "warn me if rainfall crosses 50 mm in {loc} {t}"]},
    "HUMIDITY": {"GET": ["how humid is it in {loc} {t}?", "is it humid in {loc} {t}?"],
                 "ALERT": ["alert me if the humidity goes above 90 percent in {loc} {t}"]},
    "DEW_POINT": {"GET": ["how high is the dew point in {loc} {t}?"],
                  "ALERT": ["notify me if the dew point rises in {loc} {t}"]},
    "WIND_SPEED": {"GET": ["how windy is it in {loc} {t}?", "how strong is the wind in {loc} {t}?",
                           "is it windy in {loc} {t}?", "is it too windy to spray in {loc} {t}?"],
                   "ALERT": ["alert me if the wind crosses 40 kmph in {loc} {t}",
                             "warn me about strong winds in {loc} {t}"]},
    "WIND_DIRECTION": {"GET": ["which direction is the wind blowing in {loc} {t}?",
                               "which way is the wind going in {loc} {t}?"],
                       "ALERT": ["notify me if the wind direction shifts in {loc} {t}"]},
    "SUNSHINE": {"GET": ["will it be sunny in {loc} {t}?", "how many hours of sun will {loc} get {t}?",
                         "is there enough sunlight in {loc} {t}?",
                         "can i dry the paddy in {loc} {t}?", "will the clothes dry in {loc} {t}?"],
                 "ALERT": ["alert me if sunshine hours drop in {loc} {t}"]},
    "CLOUD_COVER": {"GET": ["how cloudy is it in {loc} {t}?", "will it be cloudy in {loc} {t}?",
                            "is the sky clear in {loc} {t}?"],
                    "ALERT": ["notify me if it gets overcast in {loc} {t}"]},
    "SOIL_MOISTURE": {"GET": ["is the soil wet enough in {loc} {t}?", "do i need to irrigate in {loc} {t}?",
                              "how dry is the soil in {loc} {t}?", "can i skip watering in {loc} {t}?",
                              "is the field too wet to plough in {loc} {t}?"],
                      "ALERT": ["alert me if the soil moisture drops below normal in {loc} {t}",
                                "warn me when the field goes dry in {loc} {t}"]},
    "SOIL_TEMPERATURE": {"GET": ["is the soil warm enough for planting in {loc} {t}?",
                                 "how warm is the ground in {loc} {t}?",
                                 "can i sow in {loc} {t}?", "is the ground too cold in {loc} {t}?"],
                         "ALERT": ["alert me if the soil temperature falls in {loc} {t}"]},
}

# --- location vocabulary from the shapes DB ----------------------------------
# A fifth of the sampled names is reserved (split == "eval" in data/locations.csv) and never
# generated into train/test, so the hand-written evaluation set can use place names this
# model has genuinely never seen.

FALLBACK_HELDOUT_LOCATIONS = [(x, x) for x in [
    "Eluru", "Machilipatnam", "Srikakulam", "Adilabad", "Chittoor", "Bhimavaram",
    "Mancherial", "Siddipet", "Proddatur", "Tenali", "Jaipur", "Ahmedabad",
    "Lucknow", "Bhopal", "Indore", "Colombo", "Nairobi", "Hanoi", "Lisbon",
]]

LOCATIONS_CSV = Path(__file__).resolve().parent.parent / "data/locations.csv"
INDIA_NAMES = set()  # lowercased names the shapes DB places inside the India bbox
ALL_NAMES = set()    # every sampled name, India or not


QUALIFIED_RATE = 0.55  # share of place names that also get an address form


def _chain(name, parents):
    """['Angara', 'Rajahmundry', 'East Godavari', 'Andhra Pradesh'], minus repeats - plenty
    of villages carry their block's or district's own name."""
    out = [name]
    for parent in parents:
        if parent and parent.lower() != out[-1].lower():
            out.append(parent)
    return out


def _qualified(rng, chain):
    """A partial or full address: "Angara, East Godavari" .. "Angara, Rajahmundry,
    East Godavari, Andhra Pradesh". None when the place has no ancestors."""
    if len(chain) < 2:
        return None
    tail = chain[1:]
    if len(tail) >= 2 and rng.random() < 0.6:  # usually skip the block: village, district
        tail = tail[1:]
    keep = rng.choices(range(1, len(tail) + 1), weights=[60, 25, 15][:len(tail)])[0]
    parts = [chain[0]] + tail[:keep]
    # a third of users type "angara andhra pradesh" with no comma at all
    return (" " if rng.random() < 0.33 else ", ").join(parts)


def _load_location_vocab(path=LOCATIONS_CSV):
    """(train pool, eval-only pool) of place names, from the shapes-DB sample when present.

    Each row contributes its bare name and, most of the time, one address form built from
    its village/block/district/state ancestry. The eval pool is held out by
    src/fetch_locations.py so eval keeps unseen entity spans.
    """
    if not path.exists():
        print(f"  ! {path.name} missing - using the built-in city list "
              f"(run: python src/fetch_locations.py)")
        return FALLBACK_LOCATIONS, FALLBACK_HELDOUT_LOCATIONS
    rng = random.Random(23)  # fixed: the vocabulary must not shift between splits
    pools = {"train": [], "eval": []}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            # collapse internal whitespace: _clean() does the same to the prompt, and a
            # "Ghagrapar  Circle" span would then no longer be verbatim in its own sentence
            row = {k: " ".join(v.split()) if isinstance(v, str) else v for k, v in row.items()}
            names = [row["name"]]
            if rng.random() < QUALIFIED_RATE:
                address = _qualified(rng, _chain(row["name"], row["parents"].split(" | ")))
                if address:
                    names.append(address)
            pools[row["split"]].extend((n, n) for n in names)
            ALL_NAMES.update(n.lower() for n in names)
            if row["in_india"] == "1":
                INDIA_NAMES.update(n.lower() for n in names)
    return pools["train"], pools["eval"]


LOCATIONS, HELDOUT_LOCATIONS = _load_location_vocab()

_PUNCT_FIX = [(" ?", "?"), (" ,", ","), (" .", "."), ("? ?", "?")]
NO_TIME_RATE = 0.18  # share of prompts with no temporal mention (time == [])
CLOCK_RATE = 0.28    # share of temporal mentions that are wall-clock times
TYPO_RATE = 0.26     # share of prompts carrying a spelling mistake (outside the spans)
COMPARE_TYPO_BONUS = 0.12  # COMPARE wording is the most template-like, so rough it up harder
GRAMMAR_RATE = 0.28  # share of prompts with dropped function words / chat punctuation
LOCATION_TYPO_RATE = 0.18  # share of prompts where the PLACE NAME itself is misspelt


def _clock(rng):
    """One clock label: '6 PM', '5:30 am', '7:05 PM'. Round hours and half hours dominate,
    because that is how people speak, but odd minutes keep their tail so the model never
    learns that ':00' and ':30' are the only shapes a time can have."""
    hour = rng.randint(1, 12)
    minute = rng.choices([0, 30, 15, 45, None], weights=[52, 24, 7, 7, 10])[0]
    if minute is None:
        minute = rng.randint(1, 59)
    meridiem = rng.choice(["AM", "PM", "am", "pm"])
    label = f"{hour} {meridiem}" if minute == 0 else f"{hour}:{minute:02d} {meridiem}"
    hour24 = (hour % 12) + (12 if meridiem.lower() == "pm" else 0)
    return label, hour24 * 60 + minute


def _clock_phrase(rng):
    """(surface, span) for one wall-clock mention; span is always verbatim in surface."""
    roll = rng.random()
    if roll < 0.55:                                    # single point in time
        label, _ = _clock(rng)
        return rng.choice([f"at {label}", f"by {label}", f"around {label}"]), label
    if roll < 0.78:                                    # explicit range, start before end
        (start, s_min), (end, e_min) = _clock(rng), _clock(rng)
        if s_min >= e_min or start in end or end in start:
            return None
        return f"from {start} to {end}", f"{start} to {end}"
    if roll < 0.90:                                    # offset in hours
        hours = rng.choice([1, 2, 3, 4, 6, 8, 12, rng.randint(1, 18)])
        unit = "hour" if hours == 1 else "hours"
        return f"in the next {hours} {unit}", f"next {hours} {unit}"
    minutes = rng.choices([15, 30, 45, 60, 90, None], weights=[20, 25, 15, 15, 10, 15])[0]
    if minutes is None:
        minutes = rng.randint(5, 115)
    return f"in {minutes} minutes", f"{minutes} minutes"


def _clock_pool(rng, n):
    """n distinct wall-clock (surface, span) pairs."""
    pool, guard = {}, 0
    while len(pool) < n and guard < n * 50:
        guard += 1
        phrase = _clock_phrase(rng)
        if phrase:
            pool.setdefault(phrase[1], phrase)
    return list(pool.values())


# Fixed seed, not the --seed argument: train and test must share one clock vocabulary.
CLOCKS = _clock_pool(random.Random(11), 90)


def _pick_time(rng, times, clocks):
    if rng.random() < NO_TIME_RATE:
        return ("", "")
    if clocks and rng.random() < CLOCK_RATE:
        return rng.choice(clocks)
    return rng.choice([t for t in times if t[1]])


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    for a, b in _PUNCT_FIX:
        text = text.replace(a, b)
    return text


def _render(frame: str, metric: str, locs, times):
    """Fill a frame; return (text, location_spans, time_spans)."""
    values = {"m": metric}
    for key, (surface, _) in zip(("loc", "loc2"), locs):
        values[key] = surface
    for key, (surface, _) in zip(("t", "t2"), times):
        values[key] = surface
    text = _clean(frame.format(**values))
    loc_spans = [span for _, span in locs if span]
    time_spans = [span for _, span in times if span]
    return text, loc_spans, time_spans


def _style(rng, text, loc_spans, time_spans):
    """Casing / punctuation variation, keeping spans verbatim substrings."""
    roll = rng.random()
    if roll < 0.30:  # all lowercase user typing
        lowered = text.lower()
        return lowered, [s.lower() for s in loc_spans], [s.lower() for s in time_spans]
    if roll < 0.45 and text.endswith("?"):  # dropped punctuation
        text = text[:-1]
    if text and text[0].islower():
        cap = lambda s: s[0].upper() + s[1:] if text.startswith(s) else s
        loc_spans, time_spans = [cap(s) for s in loc_spans], [cap(s) for s in time_spans]
        text = text[0].upper() + text[1:]
    return text, loc_spans, time_spans


def _misspell(rng, word):
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


DROPPABLE = {"the", "a", "an", "is", "are", "will", "be", "do", "does", "of", "about",
             "any", "there", "me", "to", "it", "in", "for", "and"}
FILLERS = ["pls", "plz", "kindly", "asap", "sir", "bro", "urgent", "quickly"]


def _free_words(text, spans, pattern):
    """Word matches that do not overlap any annotated span - the only text safe to mangle."""
    protected = []
    for span in spans:
        start = text.find(span)
        if start >= 0:
            protected.append((start, start + len(span)))
    return [m for m in re.finditer(pattern, text)
            if not any(m.start() < end and start < m.end() for start, end in protected)]


def _bad_grammar(rng, text, spans):
    """How people actually type: dropped articles and prepositions, missing question marks,
    doubled punctuation, a trailing "pls". Spans are never touched, so they stay verbatim."""
    roll = rng.random()
    if roll < 0.45:                                   # drop a function word
        words = [m for m in _free_words(text, spans, r"\b[a-zA-Z]+\b")
                 if m.group().lower() in DROPPABLE]
        if words:
            word = rng.choice(words)
            text = text[:word.start()] + text[word.end():]
    elif roll < 0.70:                                 # punctuation as typed in a hurry
        text = text.rstrip("?. ")
        text += rng.choice(["", "", "??", "?!", "..", " ?"])
    elif roll < 0.88:                                 # chat filler
        filler = rng.choice(FILLERS)
        text = f"{filler} {text}" if rng.random() < 0.4 else f"{text.rstrip('?. ')} {filler}"
    else:                                             # drop the leading interrogative
        text = re.sub(r"^(what is|whats|what's|how is|can you tell me|i want to know)\s+", "",
                      text, flags=re.I)
    return _clean(text)


def _typo(rng, text, spans):
    """Misspell one word OUTSIDE every annotated span, so each span stays verbatim in the
    prompt while the model still has to survive "temprature" and "wat is the forcast"."""
    words = _free_words(text, spans, r"[A-Za-z]{5,}")
    if not words:
        return text
    word = rng.choice(words)
    return text[:word.start()] + _misspell(rng, word.group()) + text[word.end():]


def _typo_span(rng, text, spans):
    """Misspell a place name and update its annotation to match.

    _typo() deliberately never touches spans, which left the tagger having seen every
    village spelled perfectly - so "hyderbad" and "angara" went unrecognised. Here the
    corruption is applied to text and annotation together, so the span stays verbatim.
    """
    candidates = [s for s in spans if len(s) >= 6 and s in text]
    if not candidates:
        return text, spans
    span = rng.choice(candidates)
    words = [w for w in re.finditer(r"[A-Za-z]{5,}", span)]
    if not words:
        return text, spans
    word = rng.choice(words)
    broken = span[:word.start()] + _misspell(rng, word.group()) + span[word.end():]
    return text.replace(span, broken, 1), [broken if s == span else s for s in spans]


def _cell_rows(rng, intent: str, action: str, n: int, avoid=()):
    """Generate n prompts for one (intent, action) cell, none of them in `avoid`."""
    metrics = NOUNS[intent]
    extra = EXTRA.get(intent, {}).get(action, [])
    locations, relative, times, clocks = LOCATIONS, RELATIVE, TIMES, CLOCKS
    if action == "GET":
        frames, noloc, bare = GET_FRAMES + extra, NOLOC_GET_FRAMES, BARE_GET_FRAMES
    elif action == "ALERT":
        frames, noloc, bare = ALERT_FRAMES + extra, NOLOC_ALERT_FRAMES, BARE_ALERT_FRAMES
    else:
        frames, noloc, bare = COMPARE_FRAMES + COMPARE_TIME_FRAMES, [], BARE_COMPARE_FRAMES
        if intent not in NON_SCALAR:
            frames = frames + COMPARE_SCALAR_FRAMES

    rows, seen = [], set(avoid)
    guard = 0
    while len(rows) < n and guard < n * 200:
        guard += 1
        roll = rng.random()
        if noloc and roll < 0.15:   # no location at all -> location == [] (Rule 4.1)
            frame, pool = rng.choice(noloc), locations
        elif roll < 0.32:           # relative location ("near me", "in my field")
            frame, pool = rng.choice(bare), relative
        else:
            frame, pool = rng.choice(frames), locations
        n_loc = frame.count("{loc}") + frame.count("{loc2}")
        n_time = frame.count("{t}") + frame.count("{t2}")
        # relative compare frames read as "<relative> with <place>", never relative twice
        if pool is relative and n_loc == 2:
            locs = [rng.choice(relative), rng.choice(locations)]
        else:
            locs = rng.sample(pool, n_loc) if n_loc else []
        if n_time == 2:  # explicit time comparison -> two distinct, non-overlapping spans
            a, b = rng.sample([t for t in times + clocks if t[1]], 2)
            if a[1] in b[1] or b[1] in a[1]:
                continue
            # bare spans here: these frames supply their own "between .. and .."/"compare to",
            # so the surface preposition would read as "compare to by 10 PM".
            picked = [(a[1], a[1]), (b[1], b[1])]
        else:
            picked = [_pick_time(rng, times, clocks)] if n_time else []
        text, loc_spans, time_spans = _render(frame, rng.choice(metrics), locs, picked)
        text, loc_spans, time_spans = _style(rng, text, loc_spans, time_spans)
        # Noise on purpose: a model that only ever sees clean templates scores 100% on more
        # templates and falls over on the first real user. Spans survive untouched.
        typo_rate = TYPO_RATE + (COMPARE_TYPO_BONUS if action == "COMPARE" else 0)
        for _ in range(2):
            if rng.random() < typo_rate:
                text = _typo(rng, text, loc_spans + time_spans)
        if rng.random() < GRAMMAR_RATE:
            text = _bad_grammar(rng, text, loc_spans + time_spans)
        if rng.random() < LOCATION_TYPO_RATE:
            text, loc_spans = _typo_span(rng, text, loc_spans)
        if not all(span in text for span in loc_spans + time_spans):
            continue  # a mangling clipped a span - drop the row rather than mis-annotate it
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, "weather_intent": intent, "action": action,
                     "location": loc_spans, "time": time_spans})
    if len(rows) < n:
        print(f"  ! {intent}/{action}: only {len(rows)}/{n} unique prompts available")
    return rows


# --- validation (MODEL_RULES enforcement) -----------------------------------

INTENTS = [i.value for i in WeatherIntent]
ACTIONS = [a.value for a in Action]
_LEADING_PREP = re.compile(r"^(on|at|in|from|for|during|by|over)\s", re.I)


def validate_row(row) -> str:
    """Returns an error string, or '' if the row satisfies MODEL_RULES.md."""
    text = row["text"]
    if not text or not text.strip():
        return "empty text"
    if row["weather_intent"] not in INTENTS:
        return f"bad weather_intent {row['weather_intent']!r}"
    if row["action"] not in ACTIONS:
        return f"bad action {row['action']!r}"
    for field in ("location", "time"):
        spans = row[field]
        if not isinstance(spans, list):
            return f"{field} is not a list"
        for span in spans:
            if span not in text:
                return f"{field} span {span!r} not verbatim in text"
    # "on Friday" / "at 6 PM" must be annotated as "Friday" / "6 PM": the preposition belongs
    # to the sentence, not the temporal expression. Mixing both conventions teaches the model
    # two different right answers for the same phrase.
    for span in row["time"]:
        if _LEADING_PREP.match(span):
            return f"time span {span!r} includes a leading preposition"
    if row["action"] == "COMPARE" and len(row["location"]) + len(row["time"]) < 2:
        return "COMPARE needs 2 locations or 2 time spans"
    return ""


def india_share(rows) -> float:
    """Share of real place names (relative spans like "near me" excluded) that the shapes DB
    puts inside the India bbox. Names absent from data/locations.csv - the hand-written seed
    cities - count as outside, so this number never flatters itself."""
    relative = {span.lower() for _, span in RELATIVE}
    places = [s.lower() for r in rows for s in r["location"] if s.lower() not in relative]
    # deliberately misspelt names match nothing in the vocabulary; scoring them as
    # "outside India" would measure the typo rate rather than the geography
    known = [s for s in places if s in ALL_NAMES]
    return sum(s in INDIA_NAMES for s in known) / len(known) if known else 0.0


def load_seed(path: Path):
    if not path.exists():
        return []
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            row["location"] = json.loads(row["location"] or "[]")
            row["time"] = json.loads(row["time"] or "[]")
            rows.append(row)
    return rows


# per-cell rows, rng seed, output path, and whether the hand-written seed CSV is folded in
SPLITS = {
    "train": (150, 7, "data/processed/nlu_dataset.csv", True),  # 42 cells x 150 = 6300 rows
    "test": (36, 101, "data/processed/nlu_test.csv", False),    # 42 cells x 36 = 1512 rows
}
# splits generated later must not repeat a prompt from these
UPSTREAM = {"train": [], "test": ["train"]}

# Hand-written evaluation set - never generated, never overwritten by this script.
EVAL_MANUAL = Path(__file__).resolve().parent.parent / "data/eval_manual.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=list(SPLITS), default="train")
    parser.add_argument("--per-cell", type=int, help="rows per (intent, action) pair")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--seed-csv", default="data/intents.csv")
    parser.add_argument("--out")
    args = parser.parse_args()

    per_cell, seed, out_path, use_seed_csv = SPLITS[args.split]
    per_cell = args.per_cell or per_cell
    seed = args.seed if args.seed is not None else seed
    out_path = args.out or out_path

    root = Path(__file__).resolve().parent.parent
    rng = random.Random(seed)

    rows, dropped = [], []
    if use_seed_csv:
        for row in load_seed(root / args.seed_csv):
            (dropped if validate_row(row) else rows).append(row)
        for row in dropped:
            print(f"  ! dropped seed row: {validate_row(row)} -> {row['text']!r}")
        print(f"Seed rows kept: {len(rows)} (dropped {len(dropped)})")

    # Prompts already used by an earlier split - never reuse one across splits.
    seen = set()
    for name in UPSTREAM[args.split]:
        upstream = load_seed(root / SPLITS[name][2])
        assert upstream, f"build the {name} split first: python src/build_dataset.py --split {name}"
        seen.update(r["text"].strip().lower() for r in upstream)
    print(f"Split: {args.split}  (excluding {len(seen)} prompts from {UPSTREAM[args.split] or 'nothing'})")

    # Top each (intent, action) cell up to --per-cell, counting the seed rows it already has.
    have = Counter((r["weather_intent"], r["action"]) for r in rows)
    for intent, action in product(INTENTS, ACTIONS):
        missing = per_cell - have[(intent, action)]
        if missing > 0:
            rows.extend(_cell_rows(rng, intent, action, missing, avoid=seen))

    unique = []
    for row in rows:
        key = row["text"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(row)
    print(f"Deduplicated {len(rows) - len(unique)} prompts")

    for row in unique:
        err = validate_row(row)
        assert not err, f"{err} -> {row}"

    rng.shuffle(unique)
    out = root / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "weather_intent", "action", "location", "time"])
        for row in unique:
            writer.writerow([row["text"], row["weather_intent"], row["action"],
                             json.dumps(row["location"]), json.dumps(row["time"])])

    no_loc = sum(1 for r in unique if not r["location"])
    no_time = sum(1 for r in unique if not r["time"])
    print(f"Wrote {len(unique)} rows -> {out}")
    print(f"  intents: {len(INTENTS)}  actions: {len(ACTIONS)}  "
          f"no-location: {no_loc}  no-time: {no_time}")
    print(f"  location spans inside India: {india_share(unique):.1%}"
          if INDIA_NAMES else "  location source: built-in fallback list")


if __name__ == "__main__":
    main()
