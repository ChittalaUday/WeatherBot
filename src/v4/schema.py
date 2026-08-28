"""
v4 contracts: schema enums, single sources of truth, and slots.
"""

import re
from datetime import date
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from src.dates import MONTH_NAMES as MONTHS
from src.dates import dates_in


class Intent(str, Enum):
    """Request intents classification."""

    # Data queries
    INFORMATION = "INFORMATION"
    ADVICE = "ADVICE"
    COMPARISON = "COMPARISON"

    # Conversational templates (no API call)
    GREETING = "GREETING"
    THANKS = "THANKS"
    GOODBYE = "GOODBYE"
    SMALL_TALK = "SMALL_TALK"
    CAPABILITY = "CAPABILITY"

    # Session control
    CHANGE_LOCATION = "CHANGE_LOCATION"
    RESET = "RESET"
    AFFIRM = "AFFIRM"
    DENY = "DENY"
    EXPLAIN = "EXPLAIN"

    # Declined requests
    UNSUPPORTED_METRIC = "UNSUPPORTED_METRIC"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    UNCLEAR = "UNCLEAR"


FORECAST_HORIZON_DAYS = 10

# Single source of truth for no-data-needed intents and their default replies
REPLIES = {
    Intent.GREETING: "Hello. Ask me about the weather anywhere in India.",
    Intent.THANKS: "Glad that helped.",
    Intent.GOODBYE: "Goodbye. Take care.",
    Intent.SMALL_TALK: "I am a weather bot that reads weather data and answers questions about it.",
    Intent.CAPABILITY: (
        f"I can report temperature, rain, humidity, wind, cloud, sunshine and soil "
        f"conditions for any place in India, forecast about {FORECAST_HORIZON_DAYS} days "
        f"ahead or look up past dates, compare two places, and tell you whether to spray, "
        f"irrigate, fertilise or carry an umbrella."
    ),
    Intent.UNSUPPORTED_METRIC: "I cover temperature, rain, humidity, wind, cloud, sunshine and soil moisture.",
    Intent.OUT_OF_SCOPE: "I handle weather questions across India.",
    Intent.UNCLEAR: "I did not follow that. Name a place and what you want to know.",
    Intent.CHANGE_LOCATION: "Sure - which place should I use from now on?",
    Intent.RESET: "Starting fresh. What would you like to know?",
    Intent.AFFIRM: "Got it.",
    Intent.DENY: "Understood - tell me what it should have been.",
    Intent.EXPLAIN: "Here is what that answer was based on.",
}

CONVERSATIONAL = {Intent.GREETING, Intent.THANKS, Intent.GOODBYE, Intent.SMALL_TALK, Intent.CAPABILITY}
DECLINED = {Intent.UNSUPPORTED_METRIC, Intent.OUT_OF_SCOPE, Intent.UNCLEAR}
CONTROL = {Intent.CHANGE_LOCATION, Intent.RESET, Intent.AFFIRM, Intent.DENY, Intent.EXPLAIN}
NO_DATA_NEEDED = set(REPLIES.keys())


class Resolution(str, Enum):
    """Resolution for query responses."""

    HOURLY = "HOURLY"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


RESOLUTION_RATES = {
    Resolution.HOURLY: 24,
    Resolution.DAILY: 1,
    Resolution.WEEKLY: 1 / 7,
    Resolution.MONTHLY: 1 / 30.4,
    Resolution.YEARLY: 1 / 365.25,
}

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
    """Roughly how many rows come back for a query budget."""
    return max(int(span_days * RESOLUTION_RATES[resolution]), 1)


class WeatherIntent(str, Enum):
    """Temporal operation for weather queries."""

    NONE = "NONE"
    CURRENT = "CURRENT"
    FORECAST = "FORECAST"
    TOMORROW = "TOMORROW"
    HISTORICAL = "HISTORICAL"


class Variable(str, Enum):
    """Weather measurements requested by the user."""

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
    """Weather-dependent activities."""

    NONE = "NONE"
    OUTDOOR_ACTIVITY = "OUTDOOR_ACTIVITY"
    TRAVEL = "TRAVEL"
    RAIN_PROTECTION = "RAIN_PROTECTION"
    SUN_PROTECTION = "SUN_PROTECTION"
    CLOTHING = "CLOTHING"
    DRYING = "DRYING"
    SOW = "SOW"
    IRRIGATE = "IRRIGATE"
    FERTILIZE = "FERTILIZE"
    SPRAY = "SPRAY"
    HARVEST = "HARVEST"


class Action(str, Enum):
    """Coarse action group derived from Activity."""

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


class EntityType(str, Enum):
    """Entity types extracted alongside activities."""

    SPORT = "sport"
    TRANSPORT = "transport"
    CROP = "crop"
    MATERIAL = "material"
    CLOTHING_ITEM = "clothing"
    EVENT = "event"


# Single source of truth for entity vocabularies
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

# Derived subsets for activity-entity pairings from ENTITY_VOCAB
_SPRAY_MATERIALS = [
    "pesticide", "insecticide", "fungicide", "herbicide", "weedicide", "neem oil",
    "sulphur", "micronutrients"
]
_FERTILIZE_MATERIALS = [m for m in ENTITY_VOCAB[EntityType.MATERIAL] if m not in _SPRAY_MATERIALS]

ENTITY_SUBSETS = {
    (Activity.SPRAY, EntityType.MATERIAL): _SPRAY_MATERIALS,
    (Activity.FERTILIZE, EntityType.MATERIAL): _FERTILIZE_MATERIALS,
    (Activity.RAIN_PROTECTION, EntityType.CLOTHING_ITEM): ["raincoat", "umbrella", "shawl"],
    (Activity.CLOTHING, EntityType.CLOTHING_ITEM): [
        "jacket", "sweater", "sweatshirt", "hoodie", "woollens", "shawl", "winter clothes", "heavy clothes"
    ],
    (Activity.DRYING, EntityType.CLOTHING_ITEM): [
        "white clothes", "woollen clothes", "bedsheets", "sarees", "blankets", "shoes", "woollens", "heavy clothes"
    ],
    (Activity.DRYING, EntityType.TRANSPORT): [
        "car", "bike", "scooter", "scooty", "motorcycle", "truck", "tractor", "jeep", "lorry"
    ],
}


def terms_for(activity, kind) -> list:
    """The terms this activity may use for this entity type."""
    return ENTITY_SUBSETS.get((activity, kind)) or ENTITY_VOCAB[kind]


SUB_KEYWORDS = {
    "concrete": "construction", "masons": "construction", "construction": "construction",
    "gardening": "gardening", "garden": "gardening", "lawn": "gardening", "prune": "gardening",
    "jog": "run", "jogging": "run", "workout": "run", "running": "run",
    "walk": "walk", "stroll": "walk",
    "sunscreen": "sunscreen", "sunblock": "sunscreen",
    "shade": "shade", "cap": "shade", "hat": "shade",
    "laundry": "laundry", "washing": "laundry",
}

SUB_SYNONYMS = {
    "bicycle": "bike", "cycle": "bike", "motorcycle": "bike", "motorbike": "bike",
    "scooter": "bike", "scooty": "bike", "two wheeler": "bike",
    "jeep": "car", "lorry": "car", "truck": "car", "auto": "car", "autorickshaw": "car",
    "rickshaw": "car",
}

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
    """Derive specific sub-activity string from entities or text."""
    activity = Activity(activity)
    if activity not in SUB_DEFAULTS:
        return ""
    entities = entities or {}
    fold = lambda term: SUB_SYNONYMS.get(term.lower(), term.lower())

    if activity is Activity.DRYING:
        if entities.get(EntityType.TRANSPORT.value):
            return "vehicle"
        if entities.get(EntityType.CLOTHING_ITEM.value):
            return "laundry"

    if activity is Activity.OUTDOOR_ACTIVITY:
        for kind in (EntityType.SPORT, EntityType.EVENT):
            if terms := entities.get(kind.value):
                return fold(terms[0]) or ""

    for kind in (EntityType.TRANSPORT, EntityType.CLOTHING_ITEM):
        if terms := entities.get(kind.value):
            return fold(terms[0]) or ""

    lowered = text.lower()
    for word in sorted(SUB_KEYWORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return SUB_KEYWORDS[word]
    return SUB_DEFAULTS[activity]


INDOOR_SPORTS = frozenset({
    "badminton", "table tennis", "squash", "chess", "carrom", "billiards", "snooker",
    "bowling", "gym", "yoga", "swimming pool", "basketball court", "boxing", "wrestling",
})

VENUE_WORDS = {
    "indoor": "indoor", "indoors": "indoor", "inside": "indoor", "covered": "indoor",
    "under a roof": "indoor", "at home": "indoor",
    "outdoor": "outdoor", "outdoors": "outdoor", "outside": "outdoor",
    "open ground": "outdoor", "in the open": "outdoor", "terrace": "outdoor",
    "ground": "outdoor", "field": "outdoor",
}

OUTDOOR, INDOOR = "outdoor", "indoor"


def venue_for(activity, sub_activity: str = "", text: str = "") -> str:
    """Determine venue: 'outdoor' or 'indoor'."""
    lowered = (text or "").lower()
    for word in sorted(VENUE_WORDS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            return VENUE_WORDS[word]
    if Activity(activity) is Activity.OUTDOOR_ACTIVITY and sub_activity:
        return INDOOR if sub_activity.lower() in INDOOR_SPORTS else OUTDOOR
    return OUTDOOR


def group_for(activity) -> Action:
    """Map specific activity to coarse action group."""
    return ACTIVITY_GROUP.get(Activity(activity), Action.NONE)


class Aggregation(str, Enum):
    """Reduction requested over the range."""

    RAW = "RAW"
    SUM = "SUM"
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"
    TREND = "TREND"


class Operation(str, Enum):
    """Multi-turn operation mode."""

    SET = "SET"
    REPLACE = "REPLACE"
    MODIFY = "MODIFY"
    INHERIT = "INHERIT"
    COMPARE = "COMPARE"


class TimeBucket(str, Enum):
    """Coarse time span buckets."""

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
    CLOCK = "CLOCK"
    DATE = "DATE"


REFERENCE_YEAR = date.today().year
WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
PAST_MARKERS = ("yesterday", "last ", "past ", " ago", "previous")

DATE_PATTERN = re.compile(
    rf"\b(\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[/-]\d{{1,2}}[/-]\d{{4}}|"
    rf"(?:{'|'.join(MONTHS)})\b[^,]{{0,12}}\b\d{{4}}|\b\d{{4}}\b(?=\s|$))"
)
YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")


def first_date_in(text: str):
    """Earliest calendar date named in text."""
    found = dates_in((text or "").lower())
    return min(found) if found else None


def year_in(text: str) -> Optional[int]:
    """Year named in text."""
    match = YEAR_PATTERN.search(text or "")
    return int(match.group()) if match else None


def bucket_for(normalized: Optional[str]) -> TimeBucket:
    """Map canonical time wording to TimeBucket."""
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
    """Map time bucket to WeatherIntent."""
    bucket = bucket_for(normalized)
    if bucket is TimeBucket.DATE:
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
    """Slot extractions."""

    variables: List[Variable] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    times: List[str] = Field(default_factory=list)
    times_normalized: List[str] = Field(default_factory=list)
    entities: Dict[str, List[str]] = Field(default_factory=dict)


class V4Result(BaseModel):
    """Structured v4 parsing result for one turn."""

    text: str
    intent: Intent = Intent.INFORMATION
    weather_intent: WeatherIntent = WeatherIntent.FORECAST
    activity: Activity = Activity.NONE
    aggregation: Aggregation = Aggregation.RAW
    operation: Operation = Operation.SET
    slots: Slots = Field(default_factory=Slots)
    confidence: Dict[str, float] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)
    model_version: str = "v4"

    @property
    def action(self) -> Action:
        return group_for(self.activity)

    @property
    def time_bucket(self) -> TimeBucket:
        return bucket_for(self.slots.times_normalized[0] if self.slots.times_normalized else None)


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


# Single source of truth for field detail mapping
DETAIL_FIELDS = {
    Variable.GENERAL: {
        "MINIMAL": ["Tavg"],
        "NORMAL": ["Tavg", "Rainfall", "RH"],
        "FULL": ["Tmin", "Tmax", "Tavg", "Rainfall", "RH", "Wind_Speed", "Lowcloud"],
    },
    Variable.TEMPERATURE: {
        "MINIMAL": ["Tavg"],
        "NORMAL": ["Tmin", "Tmax", "Tavg"],
        "FULL": ["Tmin", "Tmax", "Tavg", "DPT"],
    },
    Variable.RAIN: {
        "MINIMAL": ["Rainfall"],
        "NORMAL": ["Rainfall"],
        "FULL": ["Rainfall", "RH", "Lowcloud"],
    },
    Variable.HUMIDITY: {
        "MINIMAL": ["RH"],
        "NORMAL": ["RH", "RH_max", "RH_min"],
        "FULL": ["RH", "RH_max", "RH_min", "DPT"],
    },
    Variable.WIND: {
        "MINIMAL": ["Wind_Speed"],
        "NORMAL": ["Wind_Speed", "Wind_max", "Wind_Direction"],
        "FULL": ["Wind_Speed", "Wind_max", "Wind_Direction"],
    },
    Variable.CLOUD: {
        "MINIMAL": ["Lowcloud"],
        "NORMAL": ["Lowcloud"],
        "FULL": ["Lowcloud", "SunSD"],
    },
    Variable.SUNSHINE: {
        "MINIMAL": ["SunSD"],
        "NORMAL": ["SunSD", "DayLength"],
        "FULL": ["SunSD", "DayLength", "Lowcloud"],
    },
    Variable.UV: {
        "MINIMAL": ["SunSD"],
        "NORMAL": ["SunSD", "DayLength", "Lowcloud"],
        "FULL": ["SunSD", "DayLength", "Lowcloud"],
    },
    Variable.SOIL_MOISTURE: {
        "MINIMAL": ["Soilm10"],
        "NORMAL": ["Soilm10", "Soilm40"],
        "FULL": ["Soilm10", "Soilm40", "Rainfall"],
    },
    Variable.SOIL_TEMPERATURE: {
        "MINIMAL": ["Soilt10"],
        "NORMAL": ["Soilt10"],
        "FULL": ["Soilt10", "Tavg"],
    },
}

# Derived from DETAIL_FIELDS SSOT for backward compatibility
FIELD_SETS = {var: detail["NORMAL"] for var, detail in DETAIL_FIELDS.items()}

UV_IS_A_PROXY = "no UV index in the feed - judged from sunshine hours and cloud cover"

FULL_CUES = (
    "in detail", "detailed", "full", "everything", "all the", "complete",
    "in depth", "expand", "elaborate", "more detail", "breakdown", "comprehensive",
    "every reading", "all readings", "whole picture"
)
MINIMAL_CUES = (
    "just the", "only the", "one number", "quick", "briefly", "in short",
    "short answer", "tl;dr", "summary only", "one line"
)


def detail_from_text(text: str) -> str:
    """MINIMAL / NORMAL / FULL detail level from text."""
    lowered = (text or "").lower()
    if any(cue in lowered for cue in FULL_CUES):
        return "FULL"
    if any(cue in lowered for cue in MINIMAL_CUES):
        return "MINIMAL"
    return "NORMAL"


def fields_for(variables, detail: str = "NORMAL", limit: int = 8) -> List[str]:
    """Columns to fetch for named variables at requested detail level."""
    chosen: List[str] = []
    for variable in variables:
        key = variable if isinstance(variable, Variable) else Variable(variable)
        widths = DETAIL_FIELDS.get(key) or {"NORMAL": FIELD_SETS.get(key, [])}
        for field in widths.get(detail) or widths.get("NORMAL") or []:
            if field not in chosen:
                chosen.append(field)
    return chosen[:limit] or ["Tavg"]
