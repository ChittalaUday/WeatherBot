"""
v2 contracts: one coarse intent, everything else a slot - and slots can hold several values.

v1 folded the weather variable into the intent, which meant 14 classes and no way to say
"rain and temperature". v2 splits them:

    "rain and temperature in Guntur and Vizag tomorrow"
      intent      FORECAST
      variables   [RAIN, TEMPERATURE]        <- multi-label
      locations   ["Guntur", "Vizag"]        <- multi-span
      times       ["tomorrow"]

That shape is what an AI context builder wants: one call returns every parameter it needs
to assemble a query, instead of one intent per variable.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Intent(str, Enum):
    """What the user wants done. Deliberately small - the variable is a slot now."""

    CURRENT = "CURRENT"          # what is it doing right now
    FORECAST = "FORECAST"        # what will it do
    HISTORICAL = "HISTORICAL"    # what did it do
    COMPARE = "COMPARE"          # two or more places, or two or more times
    ALERT = "ALERT"              # tell me when / warn me
    UNKNOWN = "UNKNOWN"          # nothing weather-shaped was said


class Variable(str, Enum):
    """Which measurement(s) the question is about. A query may carry several."""

    GENERAL = "GENERAL"                    # "weather" with no specific measure
    TEMPERATURE = "TEMPERATURE"
    TEMPERATURE_MIN = "TEMPERATURE_MIN"
    TEMPERATURE_MAX = "TEMPERATURE_MAX"
    RAIN = "RAIN"
    HUMIDITY = "HUMIDITY"
    DEW_POINT = "DEW_POINT"
    WIND_SPEED = "WIND_SPEED"
    WIND_DIRECTION = "WIND_DIRECTION"
    SUNSHINE = "SUNSHINE"
    CLOUD_COVER = "CLOUD_COVER"
    SOIL_MOISTURE = "SOIL_MOISTURE"
    SOIL_TEMPERATURE = "SOIL_TEMPERATURE"


class Aggregation(str, Enum):
    RAW = "RAW"
    SUM = "SUM"
    AVG = "AVG"
    MAX = "MAX"
    MIN = "MIN"
    TREND = "TREND"


class Slots(BaseModel):
    """Every slot is a list: "Guntur and Vizag" is normal, not an edge case."""

    variables: List[Variable] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)      # raw spans, resolved downstream
    times: List[str] = Field(default_factory=list)          # raw spans
    times_normalized: List[str] = Field(default_factory=list)


class V2Result(BaseModel):
    """One turn, as v2 reports it. Confidence is per head, not one number for everything."""

    text: str
    intent: Intent
    aggregation: Aggregation = Aggregation.RAW
    slots: Slots = Field(default_factory=Slots)
    confidence: Dict[str, float] = Field(default_factory=dict)   # intent / variables / aggregation
    scores: Dict[str, float] = Field(default_factory=dict)       # full intent vector
    model_version: str = "v2"


# v1 weather_intent -> v2 variable. Used to carry the existing dataset over rather than
# re-annotating 7800 rows by hand.
V1_TO_VARIABLE = {
    "CURRENT_CONDITIONS": Variable.GENERAL,
    "FORECAST": Variable.GENERAL,
    "TEMPERATURE": Variable.TEMPERATURE,
    "TEMPERATURE_MIN": Variable.TEMPERATURE_MIN,
    "TEMPERATURE_MAX": Variable.TEMPERATURE_MAX,
    "RAIN": Variable.RAIN,
    "HUMIDITY": Variable.HUMIDITY,
    "DEW_POINT": Variable.DEW_POINT,
    "WIND_SPEED": Variable.WIND_SPEED,
    "WIND_DIRECTION": Variable.WIND_DIRECTION,
    "SUNSHINE": Variable.SUNSHINE,
    "CLOUD_COVER": Variable.CLOUD_COVER,
    "SOIL_MOISTURE": Variable.SOIL_MOISTURE,
    "SOIL_TEMPERATURE": Variable.SOIL_TEMPERATURE,
}

# Canonical times that mean "already happened" / "happening now" - used to split v1's GET
# into v2's CURRENT / FORECAST / HISTORICAL.
PAST_TIMES = {"yesterday", "last week", "last month"}
PRESENT_TIMES = {"now", "today", "this morning", "this afternoon", "this evening", "tonight"}
