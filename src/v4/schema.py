"""
v4 contracts: one sentence, six independent components.

v3 asked one model "what is this turn?" and scored it with a single `everything` number, which
told you the turn was wrong but never which part. v4 splits the question into pieces that can
be trained and scored separately, so a report reads

    intent 96%   weather_intent 94%   variable 92%   activity 89%   location 97%   time 96%

instead of "overall 91%".

    "I'm biking to work tomorrow morning in Hyderabad, will the weather be okay?"

        intent          ADVICE          what kind of request
        weather_intent  TOMORROW        which temporal operation
        variables       [GENERAL]       which measurement(s)
        activity        BIKE            what the user is actually doing
        action          TRAVEL          the coarse group of that activity
        locations       ["Hyderabad"]           raw spans, resolved downstream
        times           ["tomorrow morning"]    raw spans, normalised downstream

The split that matters is the last one: `intent`/`variables` say what to *fetch*, `activity`
says what to *decide*. A weather engine turns BIKE into rain + wind + temperature; the advice
engine turns those numbers into a verdict. Neither is the classifier's job.
"""

import re
from datetime import date
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """What kind of request this is - the shape of the answer, not its subject.

    Three families. The first fetches data; the other two never touch the weather API, which
    is the point of predicting them: a greeting that goes through location resolution comes
    back asking which city you meant, and "book me a flight" answered with a rainfall table is
    worse than a plain "I only do weather".
    """

    # answered from data
    INFORMATION = "INFORMATION"    # report the numbers
    ADVICE = "ADVICE"              # decide something and say yes / no / careful
    COMPARISON = "COMPARISON"      # two or more places, or two or more times

    # answered from a template, no API call
    GREETING = "GREETING"          # hi, hello, good morning
    THANKS = "THANKS"              # thanks, that helped
    GOODBYE = "GOODBYE"            # bye, see you later
    SMALL_TALK = "SMALL_TALK"      # how are you, are you a bot
    CAPABILITY = "CAPABILITY"      # what can you do, help

    # acting on the conversation itself, not on the weather
    CHANGE_LOCATION = "CHANGE_LOCATION"        # use a different place from here on
    RESET = "RESET"                            # start the chat over
    AFFIRM = "AFFIRM"                          # yes / that one - answers a clarify prompt
    DENY = "DENY"                              # no / not that
    EXPLAIN = "EXPLAIN"                        # why? on what basis? say that again

    # declined, and the reason decides which apology gets said
    UNSUPPORTED_METRIC = "UNSUPPORTED_METRIC"  # weather-shaped, no source has it: AQI, snow
    OUT_OF_SCOPE = "OUT_OF_SCOPE"              # not weather at all
    UNCLEAR = "UNCLEAR"                        # too vague or garbled to act on


# Turns that are answered without calling the weather API.
CONVERSATIONAL = {Intent.GREETING, Intent.THANKS, Intent.GOODBYE, Intent.SMALL_TALK,
                  Intent.CAPABILITY}
DECLINED = {Intent.UNSUPPORTED_METRIC, Intent.OUT_OF_SCOPE, Intent.UNCLEAR}

# Range is NOT an NLU concern. The model extracts whatever span was asked for; how much of it
# is retrievable, from which source, and at what resolution is the query planner's decision -
# it is the layer that knows Zarr serves points and Postgres serves pre-aggregated districts.
# Putting "too long" in the label set would freeze today's database performance into a
# retrained model, and would refuse "rainfall each year from 2010 to 2025", which is 15 cheap
# rows out of a GROUP BY.
FORECAST_HORIZON_DAYS = 10   # /interpolate looks about this far ahead


class Resolution(str, Enum):
    """How finely to return a span. Derived from its length, never predicted."""

    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


# (span in days, finest resolution worth returning, coarsest sensible). The planner picks from
# the allowed set; the point is that a ten-year question becomes ~120 monthly rows or 10 yearly
# ones out of a GROUP BY, not a million observations pulled into the application.
SPAN_LADDER = [
    (10, [Resolution.HOURLY, Resolution.DAILY]),
    (31, [Resolution.DAILY]),
    (366, [Resolution.DAILY, Resolution.WEEKLY, Resolution.MONTHLY]),
    (3653, [Resolution.MONTHLY, Resolution.YEARLY]),
    (10**6, [Resolution.YEARLY]),
]


def resolutions_for(span_days: int) -> List[Resolution]:
    """Resolutions worth offering for a span of this length."""
    for limit, allowed in SPAN_LADDER:
        if span_days <= limit:
            return allowed
    return [Resolution.YEARLY]


def rows_for(span_days: int, resolution: Resolution) -> int:
    """Roughly how many rows come back - the number the query budget actually cares about."""
    per = {Resolution.HOURLY: 24, Resolution.DAILY: 1, Resolution.WEEKLY: 1 / 7,
           Resolution.MONTHLY: 1 / 30.4, Resolution.YEARLY: 1 / 365.25}[resolution]
    return max(int(span_days * per), 1)
# Turns that change session state or reference the previous answer rather than asking a new
# question. They need the conversation, not the weather API - which is why AFFIRM and DENY
# cannot sit in UNCLEAR: "yes" is a perfectly clear answer to a question the bot just asked.
CONTROL = {Intent.CHANGE_LOCATION, Intent.RESET, Intent.AFFIRM, Intent.DENY, Intent.EXPLAIN}
NO_DATA_NEEDED = CONVERSATIONAL | DECLINED | CONTROL

# What to say back. Kept here rather than in the backend because it is part of the contract:
# predicting one of these labels is only useful if there is a reply attached to it. A list per
# intent so the same greeting twice in a row does not read like a machine.
REPLIES = {
    Intent.GREETING: ["Hello. Ask me about the weather anywhere in India.",
                      "Hi there. Which place should I check?",
                      "Hey. What weather do you need?"],
    Intent.THANKS: ["Anytime.", "Glad that helped.", "You are welcome."],
    Intent.GOODBYE: ["Goodbye.", "See you.", "Take care."],
    Intent.SMALL_TALK: ["I am a weather bot - no feelings, but good forecasts.",
                        "I am a bot that reads weather data and answers questions about it."],
    Intent.CAPABILITY: [
        f"I can report temperature, rain, humidity, wind, cloud, sunshine and soil "
        f"conditions for any place in India, forecast about {FORECAST_HORIZON_DAYS} days "
        f"ahead or look up past dates, compare two places, and tell you whether to spray, "
        f"irrigate, fertilise or carry an umbrella."],
    Intent.UNSUPPORTED_METRIC: [
        "I do not have that reading. I cover temperature, rain, humidity, wind, cloud, "
        "sunshine and soil moisture and temperature."],

    Intent.OUT_OF_SCOPE: ["I only handle weather questions."],
    Intent.UNCLEAR: ["I did not follow that. Name a place and what you want to know."],
    # CHANGE_LOCATION and RESET are handled by the backend - it changes state, then replies.
    # The text here is the fallback when there is nothing to change.
    Intent.CHANGE_LOCATION: ["Sure - which place should I use from now on?"],
    Intent.RESET: ["Starting fresh. What would you like to know?"],
    Intent.AFFIRM: ["Got it."],
    Intent.DENY: ["Understood - tell me what it should have been."],
    Intent.EXPLAIN: ["Here is what that answer was based on."],
}


class WeatherIntent(str, Enum):
    """Which temporal operation the question needs.

    TOMORROW is a special case of FORECAST, kept as its own class because it is far and away
    the most asked-for window and the backend fetches it differently (one day, not a horizon).
    It is decided by the time slot, so it stays consistent with `times` by construction.

    NONE is for the turns that ask for no weather at all - a greeting has no time window.
    """

    NONE = "NONE"
    CURRENT = "CURRENT"
    FORECAST = "FORECAST"
    TOMORROW = "TOMORROW"
    HISTORICAL = "HISTORICAL"


class Variable(str, Enum):
    """Which measurement(s). Multi-label: "rain and temperature" is normal, not an edge case.

    Ten, down from v3's thirteen. TEMPERATURE_MIN / TEMPERATURE_MAX and WIND_DIRECTION are
    gone: "how hot does it get" is TEMPERATURE with aggregation MAX, which is the same answer
    reached with one fewer class to confuse. DEW_POINT folds into HUMIDITY. UV has no field in
    the WeatherSnap feed and is served by a sunshine-and-cloud proxy - the label is kept
    because users ask for it, and the answer says it is a proxy.
    """

    GENERAL = "GENERAL"
    TEMPERATURE = "TEMPERATURE"
    RAIN = "RAIN"
    HUMIDITY = "HUMIDITY"
    WIND = "WIND"
    CLOUD = "CLOUD"
    SUNSHINE = "SUNSHINE"
    UV = "UV"
    SOIL_MOISTURE = "SOIL_MOISTURE"
    SOIL_TEMPERATURE = "SOIL_TEMPERATURE"


class Activity(str, Enum):
    """The general weather-dependent thing the user wants to do - the classifier target.

    Twelve, and the test for membership is strict: a label exists only when the distinction
    changes which fields are fetched or which threshold is applied. "Can I walk" and "can I
    play cricket" read the same fields against the same thresholds, so they are one label with
    different `sub_activity`; spraying and fertilising read different fields and return
    opposite verdicts on the same day, so they are two.

    That is also why there is no single AGRICULTURE label. Grouping the five farming actions
    would put the deciding distinction in `sub_activity`, and `sub_activity` is a gazetteer -
    it would find nothing in "should I protect my cotton from insects", leaving the advice
    engine with no rule to run and no safe default (fertilise wants rain coming, harvest wants
    none). Where a missing sub_activity IS survivable - walk vs sport differ only in tolerance
    - the merge is taken. The coarse grouping still exists, as the derived `Action` below.
    """

    NONE = "NONE"
    OUTDOOR_ACTIVITY = "OUTDOOR_ACTIVITY"    # walk, sport, event, gardening, construction
    TRAVEL = "TRAVEL"                        # any journey, any vehicle
    RAIN_PROTECTION = "RAIN_PROTECTION"      # umbrella, raincoat
    SUN_PROTECTION = "SUN_PROTECTION"        # sunscreen, shade
    CLOTHING = "CLOTHING"                    # warmth: jacket, layers
    DRYING = "DRYING"                        # laundry, washing a vehicle, cleaning
    SOW = "SOW"
    IRRIGATE = "IRRIGATE"
    FERTILIZE = "FERTILIZE"
    SPRAY = "SPRAY"
    HARVEST = "HARVEST"


class Action(str, Enum):
    """The coarse group - derived from the activity by `group_for`, never annotated or
    predicted separately, so the two can never disagree."""

    NONE = "NONE"
    OUTDOOR_ACTIVITY = "OUTDOOR_ACTIVITY"
    TRAVEL = "TRAVEL"
    CLOTHING = "CLOTHING"
    HOUSEHOLD = "HOUSEHOLD"
    AGRICULTURE = "AGRICULTURE"


ACTIVITY_GROUP = {
    Activity.NONE: Action.NONE,
    Activity.OUTDOOR_ACTIVITY: Action.OUTDOOR_ACTIVITY,
    Activity.TRAVEL: Action.TRAVEL,
    Activity.RAIN_PROTECTION: Action.CLOTHING,
    Activity.SUN_PROTECTION: Action.CLOTHING,
    Activity.CLOTHING: Action.CLOTHING,
    Activity.DRYING: Action.HOUSEHOLD,
    Activity.SOW: Action.AGRICULTURE,
    Activity.IRRIGATE: Action.AGRICULTURE,
    Activity.FERTILIZE: Action.AGRICULTURE,
    Activity.SPRAY: Action.AGRICULTURE,
    Activity.HARVEST: Action.AGRICULTURE,
}


# --- sub-activity ------------------------------------------------------------
# Deliberately NOT an enum and NOT a classifier target: an open string, derived at read time.
#
# It is descriptive. It may shift a threshold - clothes need the humidity term, a car does not
# - but every activity that can carry one has a safe default, so an empty sub_activity
# degrades the answer rather than breaking it. Anything that cannot degrade safely is an
# Activity instead, which is why the five farming actions are labels and none of these are.
#
# Because it is open, a new sport or vehicle costs a line in a list and never a regenerated
# dataset or a retrained head. Most values come free from entities that are already extracted;
# SUB_KEYWORDS covers only the handful with no entity behind them.

SUB_KEYWORDS = {
    "concrete": "construction", "masons": "construction", "construction": "construction",
    "gardening": "gardening", "garden": "gardening", "lawn": "gardening", "prune": "gardening",
    "jog": "run", "jogging": "run", "workout": "run", "running": "run",
    "walk": "walk", "stroll": "walk",
    "sunscreen": "sunscreen", "sunblock": "sunscreen",
    "shade": "shade", "cap": "shade", "hat": "shade",
    "laundry": "laundry", "washing": "laundry",
}

# Synonyms folded only where the advice engine would otherwise need to know all of them.
SUB_SYNONYMS = {
    "bicycle": "bike", "cycle": "bike", "motorcycle": "bike", "motorbike": "bike",
    "scooter": "bike", "scooty": "bike", "two wheeler": "bike",
    "jeep": "car", "lorry": "car", "truck": "car", "auto": "car", "autorickshaw": "car",
    "rickshaw": "car",
}

# Activities that can carry one at all, with the value assumed when nothing specific is named.
SUB_DEFAULTS = {
    Activity.TRAVEL: "car",
    Activity.OUTDOOR_ACTIVITY: "",
    Activity.DRYING: "laundry",
    Activity.RAIN_PROTECTION: "umbrella",
    Activity.SUN_PROTECTION: "sunscreen",
    Activity.CLOTHING: "jacket",
}


def sub_activity_for(activity, entities: Optional[Dict[str, List[str]]] = None,
                     text: str = "") -> str:
    """The specific thing named, as a plain lowercase string. "" when nothing was.

    The farming activities always return "" - they are the leaf already, and giving them a
    sub-activity would re-create the distinction the Activity label exists to carry.
    """
    activity = Activity(activity)
    if activity not in SUB_DEFAULTS:
        return ""
    entities = entities or {}
    fold = lambda term: SUB_SYNONYMS.get(term.lower(), term.lower())

    # DRYING is the one case where the entity *type* decides rather than the term: washing a
    # bike and washing a shirt are both DRYING, and "bike" there means a vehicle, not a ride.
    if activity is Activity.DRYING:
        if entities.get(EntityType.TRANSPORT.value):
            return "vehicle"
        if entities.get(EntityType.CLOTHING_ITEM.value):
            return "laundry"

    if activity is Activity.OUTDOOR_ACTIVITY:
        for kind in (EntityType.SPORT, EntityType.EVENT):
            if terms := entities.get(kind.value):
                return fold(terms[0])            # the sport or event itself: "cricket"

    for kind in (EntityType.TRANSPORT, EntityType.CLOTHING_ITEM):
        if terms := entities.get(kind.value):
            return fold(terms[0])

    lowered = text.lower()
    for word in sorted(SUB_KEYWORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return SUB_KEYWORDS[word]
    return SUB_DEFAULTS[activity]


class EntityType(str, Enum):
    """The specific thing involved, extracted alongside the activity.

    These are closed vocabularies - there are a few dozen sports and a few dozen crops - so
    they are matched by gazetteer in src/v4/entities.py rather than predicted. A lookup over a
    closed list cannot be 87% right about whether "cotton" is a crop.
    """

    SPORT = "sport"
    TRANSPORT = "transport"
    CROP = "crop"
    MATERIAL = "material"          # what is being applied: urea, pesticide, manure
    CLOTHING_ITEM = "clothing"
    EVENT = "event"


ENTITY_VOCAB = {
    EntityType.SPORT: [
        "cricket", "football", "soccer", "tennis", "badminton", "volleyball", "hockey",
        "kabaddi", "basketball", "golf", "kho kho", "throwball", "athletics", "marathon",
    ],
    EntityType.TRANSPORT: [
        "bike", "bicycle", "cycle", "motorcycle", "motorbike", "scooter", "scooty", "car",
        "bus", "train", "auto", "autorickshaw", "rickshaw", "tractor", "two wheeler",
        "lorry", "truck", "jeep", "boat", "flight",
    ],
    EntityType.CROP: [
        "paddy", "rice", "cotton", "wheat", "maize", "corn", "groundnut", "peanut",
        "sugarcane", "chilli", "chili", "tomato", "onion", "potato", "soybean", "soyabean",
        "mustard", "bajra", "jowar", "ragi", "millet", "turmeric", "banana", "mango",
        "grapes", "pulses", "gram", "bengal gram", "red gram", "black gram", "castor",
        "sunflower", "sesame", "tobacco", "coconut", "coffee", "tea", "brinjal", "okra",
        "cabbage", "cauliflower", "watermelon", "papaya", "guava", "pomegranate",
    ],
    EntityType.MATERIAL: [
        "pesticide", "insecticide", "fungicide", "herbicide", "weedicide", "urea", "dap",
        "npk", "potash", "fertilizer", "fertiliser", "manure", "compost", "micronutrients",
        "zinc", "gypsum", "vermicompost", "neem oil", "sulphur",
    ],
    EntityType.CLOTHING_ITEM: [
        "jacket", "sweater", "raincoat", "umbrella", "woollens", "sweatshirt", "hoodie",
        "blanket", "blankets", "bedsheet", "bedsheets", "saree", "sarees", "shoes",
        "white clothes", "woollen clothes", "heavy clothes", "winter clothes", "shawl",
    ],
    EntityType.EVENT: [
        "wedding", "marriage", "function", "party", "picnic", "festival", "procession",
        "outdoor shoot", "photo shoot", "farmers market", "exhibition", "rally", "fair",
        "housewarming", "get together", "reunion", "concert",
    ],
}


def group_for(activity) -> Action:
    """Specific activity -> coarse action group."""
    return ACTIVITY_GROUP.get(Activity(activity), Action.NONE)


class Aggregation(str, Enum):
    """The 'determine' functions: which reduction the wording asks for over the range."""

    RAW = "RAW"
    SUM = "SUM"
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"
    TREND = "TREND"


class Operation(str, Enum):
    """How this turn folds into the conversation - the multi-turn half of the contract."""

    SET = "SET"                # a fresh, self-contained question
    REPLACE = "REPLACE"        # same question, new place or time
    MODIFY = "MODIFY"          # same subject, one slot changed
    INHERIT = "INHERIT"        # a fragment leaning entirely on the previous turn
    COMPARE = "COMPARE"        # explicitly against what came before


class TimeBucket(str, Enum):
    """A coarse label for the time span, derived from its normalised form.

    Derived, not predicted: `times` already carries the raw span and the resolver already
    turns it into absolute datetimes. A classifier here would be a third opinion on a question
    two components have already answered.
    """

    NONE = "NONE"
    NOW = "NOW"
    TODAY = "TODAY"
    TONIGHT = "TONIGHT"
    MORNING = "MORNING"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"
    TOMORROW = "TOMORROW"
    DAY_AFTER = "DAY_AFTER"
    WEEKDAY = "WEEKDAY"
    WEEKEND = "WEEKEND"
    WEEK = "WEEK"
    MONTH = "MONTH"
    PAST = "PAST"
    CLOCK = "CLOCK"            # "18:45" or "07:00-11:00"
    DATE = "DATE"              # an explicit calendar date - "15 august 2023", "2023-08-15"


# The year a build treats as "now" when deciding whether a dated turn is past.
REFERENCE_YEAR = date.today().year

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
# Checked first and as substrings: "last sunday" is the past, not a weekday, and "the past
# week" is the past, not a week. Matching those on the bare noun gets the tense backwards.
PAST_MARKERS = ("yesterday", "last ", "past ", " ago", "previous")


MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august",
          "september", "october", "november", "december")
# "15 august 2023", "august 2023", "2023-08-15", "15/08/2023" - anything naming a real date.
DATE_PATTERN = re.compile(
    rf"\b(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}}|"
    rf"(?:{'|'.join(MONTHS)})\b[^,]{{0,12}}\b\d{{4}}|\b\d{{4}}\b(?=\s|$))")
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


_MONTH_INDEX = {name: i for i, name in enumerate(MONTHS, start=1)}
_MONTH_INDEX.update({name[:3]: i for i, name in enumerate(MONTHS, start=1)})
_MONTH_INDEX["sept"] = 9
_DATE_FORMS = (
    (r"\b(?P<y>(?:19|20)\d{2})-(?P<m>\d{1,2})-(?P<d>\d{1,2})\b", None),
    (r"\b(?P<d>\d{1,2})[/-](?P<m>\d{1,2})[/-](?P<y>(?:19|20)\d{2})\b", None),
    (r"\b(?P<d>\d{1,2})\s+(?P<M>[a-z]+)\.?\s+(?P<y>(?:19|20)\d{2})\b", "M"),
    (r"\b(?P<M>[a-z]+)\.?\s+(?P<d>\d{1,2}),?\s+(?P<y>(?:19|20)\d{2})\b", "M"),
)


def first_date_in(text: str):
    """The earliest calendar date named, as a date. None when none is."""
    found = []
    for pattern, named in _DATE_FORMS:
        for match in re.finditer(pattern, (text or "").lower()):
            parts = match.groupdict()
            month = _MONTH_INDEX.get(parts["M"].rstrip(".")) if named else int(parts["m"])
            if not month:
                continue
            try:
                found.append(date(int(parts["y"]), month, int(parts["d"])))
            except ValueError:
                continue
    return min(found) if found else None


def year_in(text: str) -> Optional[int]:
    match = YEAR_PATTERN.search(text or "")
    return int(match.group()) if match else None


def bucket_for(normalized: Optional[str]) -> TimeBucket:
    """Canonical time wording -> coarse bucket. Pure lookup, never a model.

    Order is the whole algorithm: the first needle that matches wins, so the more specific
    wording has to be listed above the more general one it contains - "day after tomorrow"
    before "tomorrow", "tomorrow morning" before "morning".
    """
    text = (normalized or "").strip().lower()
    if not text:
        return TimeBucket.NONE
    if DATE_PATTERN.search(text):
        return TimeBucket.DATE
    if any(marker in text for marker in PAST_MARKERS):
        return TimeBucket.PAST
    if ":" in text:
        return TimeBucket.CLOCK
    if text in WEEKDAYS:
        return TimeBucket.WEEKDAY
    for needle, bucket in (
        ("day after tomorrow", TimeBucket.DAY_AFTER), ("tomorrow", TimeBucket.TOMORROW),
        ("tonight", TimeBucket.TONIGHT), ("weekend", TimeBucket.WEEKEND),
        ("morning", TimeBucket.MORNING), ("afternoon", TimeBucket.AFTERNOON),
        ("evening", TimeBucket.EVENING),
        ("hour", TimeBucket.NOW), ("moment", TimeBucket.NOW), ("currently", TimeBucket.NOW),
        ("now", TimeBucket.NOW), ("noon", TimeBucket.TODAY), ("today", TimeBucket.TODAY),
        ("week", TimeBucket.WEEK), ("month", TimeBucket.MONTH),
    ):
        if needle in text:
            return bucket
    return TimeBucket.NONE


def weather_intent_for(normalized: Optional[str]) -> WeatherIntent:
    """Time bucket -> temporal operation, so the two can never contradict each other."""
    bucket = bucket_for(normalized)
    if bucket is TimeBucket.DATE:
        # Compare the whole date, not just the year. "11 jan 2026" asked in August 2026 is the
        # past, and a year-only test called it a forecast - the archive was never consulted.
        when = first_date_in(normalized or "")
        if when:
            return (WeatherIntent.HISTORICAL if when < date.today() else WeatherIntent.FORECAST)
        year = year_in(normalized or "")
        return (WeatherIntent.HISTORICAL if year and year < REFERENCE_YEAR
                else WeatherIntent.FORECAST)
    if bucket is TimeBucket.PAST:
        return WeatherIntent.HISTORICAL
    if bucket is TimeBucket.TOMORROW:
        return WeatherIntent.TOMORROW
    if bucket in {TimeBucket.NOW, TimeBucket.TODAY, TimeBucket.TONIGHT, TimeBucket.MORNING,
                  TimeBucket.AFTERNOON, TimeBucket.EVENING, TimeBucket.CLOCK}:
        return WeatherIntent.CURRENT
    return WeatherIntent.FORECAST


class Slots(BaseModel):
    """Every slot is a list: "Guntur and Vizag" is normal, not an edge case."""

    variables: List[Variable] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)       # raw spans, verbatim in the text
    times: List[str] = Field(default_factory=list)           # raw spans, verbatim in the text
    times_normalized: List[str] = Field(default_factory=list)
    # {"sport": ["cricket"], "crop": ["cotton"]} - the specific things the activity involves
    entities: Dict[str, List[str]] = Field(default_factory=dict)


class V4Result(BaseModel):
    """One turn, as six components plus the confidence of each."""

    text: str
    intent: Intent = Intent.INFORMATION
    weather_intent: WeatherIntent = WeatherIntent.FORECAST
    activity: Activity = Activity.NONE
    aggregation: Aggregation = Aggregation.RAW
    operation: Operation = Operation.SET
    slots: Slots = Field(default_factory=Slots)
    confidence: Dict[str, float] = Field(default_factory=dict)     # one per component
    scores: Dict[str, float] = Field(default_factory=dict)
    model_version: str = "v4"

    @property
    def action(self) -> Action:
        return group_for(self.activity)

    @property
    def time_bucket(self) -> TimeBucket:
        return bucket_for(self.slots.times_normalized[0] if self.slots.times_normalized else None)


# Which measurements an activity actually depends on. The advice engine reads this; the
# dataset builder uses it so a generated ADVICE row carries variables consistent with its
# activity instead of whatever the frame happened to name.
# Which terms make sense for which activity. Without this the generator writes "spray
# fertiliser" and "carry a white clothes" - grammatical noise, and worse, it teaches the model
# that SPRAY and FERTILIZE take the same inputs when the whole reason they are separate labels
# is that they do not. Every list here must stay a subset of ENTITY_VOCAB or the gazetteer
# will not find back what was inserted; `check` asserts exactly that.
ENTITY_SUBSETS = {
    (Activity.SPRAY, EntityType.MATERIAL): [
        "pesticide", "insecticide", "fungicide", "herbicide", "weedicide", "neem oil",
        "sulphur", "micronutrients"],
    (Activity.FERTILIZE, EntityType.MATERIAL): [
        "urea", "dap", "npk", "potash", "fertilizer", "fertiliser", "manure", "compost",
        "vermicompost", "zinc", "gypsum"],
    (Activity.RAIN_PROTECTION, EntityType.CLOTHING_ITEM): ["raincoat", "umbrella", "shawl"],
    (Activity.CLOTHING, EntityType.CLOTHING_ITEM): [
        "jacket", "sweater", "sweatshirt", "hoodie", "woollens", "shawl", "winter clothes",
        "heavy clothes"],
    (Activity.DRYING, EntityType.CLOTHING_ITEM): [
        "white clothes", "woollen clothes", "bedsheets", "sarees", "blankets", "shoes",
        "woollens", "heavy clothes"],
    (Activity.DRYING, EntityType.TRANSPORT): [
        "car", "bike", "scooter", "scooty", "motorcycle", "truck", "tractor", "jeep", "lorry"],
}


def terms_for(activity, kind) -> list:
    """The terms this activity may use for this entity type - the subset if one is declared.

    Generator-side only. The runtime gazetteer matches the whole vocabulary regardless of
    activity, and CONFUSION_FRAMES deliberately break these pairings, so no model ever learns
    that a material implies an activity.
    """
    return ENTITY_SUBSETS.get((activity, kind)) or ENTITY_VOCAB[kind]


ACTIVITY_VARIABLES = {
    Activity.NONE: [],
    Activity.OUTDOOR_ACTIVITY: [Variable.RAIN, Variable.TEMPERATURE, Variable.WIND],
    Activity.TRAVEL: [Variable.RAIN, Variable.WIND],
    Activity.RAIN_PROTECTION: [Variable.RAIN],
    Activity.SUN_PROTECTION: [Variable.UV, Variable.SUNSHINE, Variable.CLOUD],
    Activity.CLOTHING: [Variable.TEMPERATURE, Variable.RAIN],
    Activity.DRYING: [Variable.RAIN, Variable.HUMIDITY, Variable.SUNSHINE],
    Activity.SOW: [Variable.SOIL_MOISTURE, Variable.SOIL_TEMPERATURE, Variable.RAIN],
    Activity.IRRIGATE: [Variable.SOIL_MOISTURE, Variable.RAIN, Variable.TEMPERATURE],
    Activity.FERTILIZE: [Variable.RAIN, Variable.SOIL_MOISTURE],
    Activity.SPRAY: [Variable.WIND, Variable.RAIN],
    Activity.HARVEST: [Variable.RAIN, Variable.HUMIDITY],
}


# --- API field mapping -------------------------------------------------------
# Which WeatherSnap columns a variable means. v3 kept this keyed on its own 13-value enum;
# v4 has 10, so it needs its own table rather than a translation layer.
#
# UV has no column anywhere. It is served by a sunshine-and-cloud proxy and the answer has to
# say so - see UV_IS_A_PROXY.
FIELD_SETS = {
    Variable.GENERAL: ["Tavg", "Rainfall", "RH"],
    Variable.TEMPERATURE: ["Tmin", "Tmax", "Tavg"],
    Variable.RAIN: ["Rainfall"],
    Variable.HUMIDITY: ["RH", "RH_max", "RH_min"],
    Variable.WIND: ["Wind_Speed", "Wind_max", "Wind_Direction"],
    Variable.CLOUD: ["Lowcloud"],
    Variable.SUNSHINE: ["SunSD", "DayLength"],
    Variable.UV: ["SunSD", "DayLength", "Lowcloud"],
    Variable.SOIL_MOISTURE: ["Soilm10", "Soilm40"],
    Variable.SOIL_TEMPERATURE: ["Soilt10"],
}
UV_IS_A_PROXY = ("no UV index in the feed - judged from sunshine hours and cloud cover")

# How wide the table should be. v3 trained a head for this and scored 100% on it - which is
# the tell that it never needed a model: "in detail", "full", "just the" is a closed set of
# cue words, so it is read off the text like the time qualifiers are.
FULL_CUES = ("in detail", "detailed", "full", "everything", "all the", "complete",
             "in depth", "expand", "elaborate", "more detail", "breakdown", "comprehensive",
             "every reading", "all readings", "whole picture")
MINIMAL_CUES = ("just the", "only the", "one number", "quick", "briefly", "in short",
                "short answer", "tl;dr", "summary only", "one line")

# The columns each variable means at each width. NORMAL is what a question without a cue gets.
DETAIL_FIELDS = {
    Variable.GENERAL: {
        "MINIMAL": ["Tavg"],
        "NORMAL": ["Tavg", "Rainfall", "RH"],
        "FULL": ["Tmin", "Tmax", "Tavg", "Rainfall", "RH", "Wind_Speed", "Lowcloud"],
    },
    Variable.TEMPERATURE: {"MINIMAL": ["Tavg"], "NORMAL": ["Tmin", "Tmax", "Tavg"],
                           "FULL": ["Tmin", "Tmax", "Tavg", "DPT"]},
    Variable.RAIN: {"MINIMAL": ["Rainfall"], "NORMAL": ["Rainfall"],
                    "FULL": ["Rainfall", "RH", "Lowcloud"]},
    Variable.HUMIDITY: {"MINIMAL": ["RH"], "NORMAL": ["RH", "RH_max", "RH_min"],
                        "FULL": ["RH", "RH_max", "RH_min", "DPT"]},
    Variable.WIND: {"MINIMAL": ["Wind_Speed"], "NORMAL": ["Wind_Speed", "Wind_max",
                                                          "Wind_Direction"],
                    "FULL": ["Wind_Speed", "Wind_max", "Wind_Direction"]},
    Variable.CLOUD: {"MINIMAL": ["Lowcloud"], "NORMAL": ["Lowcloud"],
                     "FULL": ["Lowcloud", "SunSD"]},
    Variable.SUNSHINE: {"MINIMAL": ["SunSD"], "NORMAL": ["SunSD", "DayLength"],
                        "FULL": ["SunSD", "DayLength", "Lowcloud"]},
    Variable.UV: {"MINIMAL": ["SunSD"], "NORMAL": ["SunSD", "DayLength", "Lowcloud"],
                  "FULL": ["SunSD", "DayLength", "Lowcloud"]},
    Variable.SOIL_MOISTURE: {"MINIMAL": ["Soilm10"], "NORMAL": ["Soilm10", "Soilm40"],
                             "FULL": ["Soilm10", "Soilm40", "Rainfall"]},
    Variable.SOIL_TEMPERATURE: {"MINIMAL": ["Soilt10"], "NORMAL": ["Soilt10"],
                                "FULL": ["Soilt10", "Tavg"]},
}


def detail_from_text(text: str) -> str:
    """MINIMAL / NORMAL / FULL, read off the wording. Explicit cues win; everything else NORMAL."""
    lowered = (text or "").lower()
    if any(cue in lowered for cue in FULL_CUES):
        return "FULL"
    if any(cue in lowered for cue in MINIMAL_CUES):
        return "MINIMAL"
    return "NORMAL"


def fields_for(variables, detail: str = "NORMAL", limit: int = 8) -> List[str]:
    """Columns to fetch, in the order the user named their variables, at the asked-for width."""
    chosen: List[str] = []
    for variable in variables:
        key = variable if isinstance(variable, Variable) else Variable(variable)
        widths = DETAIL_FIELDS.get(key) or {"NORMAL": FIELD_SETS.get(key, [])}
        for field in widths.get(detail) or widths.get("NORMAL") or []:
            if field not in chosen:
                chosen.append(field)
    return chosen[:limit] or ["Tavg"]
