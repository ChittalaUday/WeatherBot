"""
Shared contracts: the conversation state, and the v1-era NLU output `src/nlu.py` still emits.

`backend/nlu/context.py` reads `ConversationState`, `Operation` and `Reference`; nothing else
in the serving path touches this file. The served taxonomy lives in `src/v4/schema.py` - this
is not the place for a new label.
"""

from enum import Enum
from typing import List, Optional

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


# --- pipeline contracts -----------------------------------------------------
# Frozen on purpose: TF-IDF, fastText or a MiniLM encoder can each be swapped in behind
# NLUResult without touching the context engine, the resolvers or the response builder.


class Operation(str, Enum):
    """What a turn does to the conversation state."""

    SET = "SET"            # first mention: location and time both arrive
    REPLACE = "REPLACE"    # "what about Rajahmundry?" - same question, new place
    MODIFY = "MODIFY"      # "what about tomorrow?" - same place, new time
    INHERIT = "INHERIT"    # "there?" - nothing new, reuse everything
    COMPARE = "COMPARE"    # two places or two times in one turn
    CLEAR = "CLEAR"        # start again


class Reference(str, Enum):
    """Phrases that point at the previous turn instead of naming anything."""

    LOCATION = "LOCATION_REFERENCE"    # there, that place, same place
    DATE = "DATE_REFERENCE"            # same day, that day
    TIME = "TIME_REFERENCE"            # at that time
    NONE = "NONE"


class Normalized(BaseModel):
    """Text normalizer output. The original is kept so every rewrite stays auditable."""

    original: str
    normalized: str
    replacements: List[List[str]] = Field(default_factory=list)   # [[before, after], ...]


class ConversationState(BaseModel):
    """Slots carried between turns. Deterministic - no model touches this."""

    location: List[str] = Field(default_factory=list)   # raw spans, resolved separately
    resolved: List[dict] = Field(default_factory=list)  # what the location engine returned
    time_raw: Optional[str] = None
    time_normalized: Optional[str] = None
    weather_intent: Optional[str] = None      # v1 weather_intent or v2 coarse intent
    action: Optional[str] = None
    variables: List[str] = Field(default_factory=list)   # v2 can carry several
    aggregation: Aggregation = Aggregation.RAW
    coords: Optional[dict] = None                       # browser geolocation, once given
    turns: int = 0
