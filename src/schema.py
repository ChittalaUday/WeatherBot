"""
Schema definitions for WeatherBot NLU system.
"""

from enum import Enum
from typing import Dict, List, Optional
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


class Verdict(str, Enum):
    """Query validator outcome - the only three ways a turn can end."""

    READY = "READY"
    CLARIFY = "CLARIFY"
    REJECT = "REJECT"


class Normalized(BaseModel):
    """Text normalizer output. The original is kept so every rewrite stays auditable."""

    original: str
    normalized: str
    replacements: List[List[str]] = Field(default_factory=list)   # [[before, after], ...]


class NLUResult(BaseModel):
    """One turn as the models see it - no resolution, no context, no coordinates."""

    text: Normalized
    weather_intent: WeatherIntent
    action: Action
    aggregation: Aggregation = Aggregation.RAW
    entities: Entities = Field(default_factory=Entities)
    reference: Reference = Reference.NONE
    confidence: float = 0.0
    scores: Dict[str, float] = Field(default_factory=dict)   # full vector, for error analysis


class ConversationState(BaseModel):
    """Slots carried between turns. Deterministic - no model touches this."""

    location: List[str] = Field(default_factory=list)   # raw spans, resolved separately
    resolved: List[dict] = Field(default_factory=list)  # what the location engine returned
    time_raw: Optional[str] = None
    time_normalized: Optional[str] = None
    weather_intent: Optional[WeatherIntent] = None
    action: Optional[Action] = None
    aggregation: Aggregation = Aggregation.RAW
    coords: Optional[dict] = None                       # browser geolocation, once given
    turns: int = 0


class ResolvedQuery(BaseModel):
    """The canonical query handed to the weather service, after every resolver has run."""

    weather_intent: WeatherIntent
    action: Action
    aggregation: Aggregation
    places: List[dict] = Field(default_factory=list)    # resolved, with lat/lon
    start: Optional[str] = None                         # ISO date/datetime, inclusive
    end: Optional[str] = None
    granularity: str = "daily"                          # daily | hourly
    time_label: str = ""                                # what to print above the table
    operation: Operation = Operation.SET
    verdict: Verdict = Verdict.READY
    missing: List[str] = Field(default_factory=list)    # slots that stopped it being READY
