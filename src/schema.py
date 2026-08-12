"""
Schema definitions for WeatherBot NLU system.
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, Field


class WeatherIntent(str, Enum):
    CURRENT_CONDITIONS = "CURRENT_CONDITIONS"
    FORECAST = "FORECAST"
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


class Action(str, Enum):
    GET = "GET"
    COMPARE = "COMPARE"
    ALERT = "ALERT"


class Aggregation(str, Enum):
    """How to reduce the selected rows before answering (Rule 2.3).

    Orthogonal to weather_intent: TEMPERATURE_MAX picks the Tmax *field*, while MAX picks the
    largest value across the chosen time range. "total rain this week" is RAIN + SUM.
    """

    RAW = "RAW"        # show the values as they come
    SUM = "SUM"        # total over the range - rainfall, sunshine hours
    AVG = "AVG"        # mean over the range
    MAX = "MAX"        # peak, hottest, strongest
    MIN = "MIN"        # lowest, coldest, weakest
    TREND = "TREND"    # when does it rise/fall/start/stop


class Entities(BaseModel):
    location: List[str] = Field(default_factory=list, description="Raw location entity text tokens extracted from prompt")
    time: List[str] = Field(default_factory=list, description="Raw temporal entity text tokens extracted from prompt")
    time_normalized: List[str] = Field(
        default_factory=list,
        description=(
            "Canonical form of each `time` span, positionally aligned: spelling and chat "
            "shorthand folded away ('tommorrow' -> 'tomorrow'), clock times as 24h HH:MM, "
            "ranges as HH:MM-HH:MM, durations as 'next N <unit>'. Query on this; `time` "
            "stays raw per Rule 4.2. Resolving to an actual datetime remains downstream."
        ),
    )


class NLUOutput(BaseModel):
    weather_intent: WeatherIntent
    action: Action
    aggregation: Aggregation = Aggregation.RAW
    entities: Entities
