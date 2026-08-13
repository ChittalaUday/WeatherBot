"""
v3 contracts: everything v2 predicts, plus the three decisions Python used to make.

v2 told the backend *what* was asked and a lookup table decided how to answer it: which
columns to show (INTENT_FIELDS), which chart to draw (a branch on series count), which
insights to compute (all of them, every time). Those choices are in the wording, and the
wording is exactly what a model can read:

    "temp in Guntur tomorrow"                -> detail NORMAL   chart STAT    insights RANGE
    "temperature in Guntur in detail"        -> detail FULL      chart LINE    insights RANGE, PEAK, LOW
    "compare rain in Guntur and Vizag"       -> detail NORMAL    chart GROUPED_BAR
                                                insights TOTAL, COMPARISON
    "just the rainfall total this week"      -> detail MINIMAL   chart NONE    insights TOTAL

v3 also never asks a question: it commits to a reading and reports what it assumed.
"""

from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field

from src.v2.schema import Aggregation, Intent, Slots, Variable


class Detail(str, Enum):
    """How much of a variable to show.

    MINIMAL a single number, no table
    NORMAL  the headline field for each variable
    FULL    every related field - "temperature in detail" means min, max and average
    """

    MINIMAL = "MINIMAL"
    NORMAL = "NORMAL"
    FULL = "FULL"


class ChartKind(str, Enum):
    """Which picture actually helps, decided from the question rather than from row counts."""

    NONE = "NONE"                  # one value, or a question a table answers better
    STAT = "STAT"                  # a single big number
    LINE = "LINE"                  # one series over time
    MULTI_LINE = "MULTI_LINE"      # several variables over time, one place
    GROUPED_BAR = "GROUPED_BAR"    # places or periods side by side


class Insight(str, Enum):
    """The observations worth making. Multi-label: a comparison over a week wants several."""

    TOTAL = "TOTAL"                # summed over the range
    AVERAGE = "AVERAGE"
    RANGE = "RANGE"                # min-max spread
    PEAK = "PEAK"                  # highest value and when
    LOW = "LOW"                    # lowest value and when
    TREND = "TREND"                # rising, falling, turning point
    THRESHOLD = "THRESHOLD"        # crossings worth a warning
    COMPARISON = "COMPARISON"      # which place or period wins
    DRY_SPELL = "DRY_SPELL"        # consecutive days under a rain threshold


class Presentation(BaseModel):
    """The three decisions v3 makes that v2 left to Python."""

    detail: Detail = Detail.NORMAL
    chart: ChartKind = ChartKind.NONE
    insights: List[Insight] = Field(default_factory=list)


class V3Result(BaseModel):
    """One turn, fully decided: what was asked and how to present it."""

    text: str
    intent: Intent
    aggregation: Aggregation = Aggregation.RAW
    slots: Slots = Field(default_factory=Slots)
    presentation: Presentation = Field(default_factory=Presentation)
    confidence: Dict[str, float] = Field(default_factory=dict)
    scores: Dict[str, float] = Field(default_factory=dict)
    assumed: List[str] = Field(default_factory=list)   # what it committed to instead of asking
    model_version: str = "v3"


# Which API fields a variable means at each detail level. The mapping itself stays
# deterministic (MODEL_RULES Section 1); the model only chooses the level.
FIELD_SETS = {
    Variable.TEMPERATURE: {"MINIMAL": ["Tavg"], "NORMAL": ["Tavg"],
                           "FULL": ["Tmin", "Tmax", "Tavg"]},
    Variable.TEMPERATURE_MIN: {"MINIMAL": ["Tmin"], "NORMAL": ["Tmin"],
                               "FULL": ["Tmin", "Tavg"]},
    Variable.TEMPERATURE_MAX: {"MINIMAL": ["Tmax"], "NORMAL": ["Tmax"],
                               "FULL": ["Tmax", "Tavg"]},
    Variable.RAIN: {"MINIMAL": ["Rainfall"], "NORMAL": ["Rainfall"],
                    "FULL": ["Rainfall", "RH"]},
    Variable.HUMIDITY: {"MINIMAL": ["RH"], "NORMAL": ["RH"],
                        "FULL": ["RH", "RH_max", "RH_min"]},
    Variable.DEW_POINT: {"MINIMAL": ["DPT"], "NORMAL": ["DPT"], "FULL": ["DPT", "RH"]},
    Variable.WIND_SPEED: {"MINIMAL": ["Wind_Speed"], "NORMAL": ["Wind_Speed"],
                          "FULL": ["Wind_Speed", "Wind_max", "Wind_Direction"]},
    Variable.WIND_DIRECTION: {"MINIMAL": ["Wind_Direction"], "NORMAL": ["Wind_Direction"],
                              "FULL": ["Wind_Direction", "Wind_Speed"]},
    Variable.SUNSHINE: {"MINIMAL": ["SunSD"], "NORMAL": ["SunSD"],
                        "FULL": ["SunSD", "DayLength", "Lowcloud"]},
    Variable.CLOUD_COVER: {"MINIMAL": ["Lowcloud"], "NORMAL": ["Lowcloud"],
                           "FULL": ["Lowcloud", "SunSD"]},
    Variable.SOIL_MOISTURE: {"MINIMAL": ["Soilm10"], "NORMAL": ["Soilm10", "Soilm40"],
                             "FULL": ["Soilm10", "Soilm40", "Rainfall"]},
    Variable.SOIL_TEMPERATURE: {"MINIMAL": ["Soilt10"], "NORMAL": ["Soilt10"],
                                "FULL": ["Soilt10", "Tavg"]},
    Variable.GENERAL: {"MINIMAL": ["Tavg"], "NORMAL": ["Tavg", "Rainfall", "RH"],
                       "FULL": ["Tmin", "Tmax", "Rainfall", "RH", "Wind_Speed", "Lowcloud"]},
}


def fields_for(variables, detail: Detail, limit: int = 6) -> list[str]:
    """Columns to fetch, in the order the user named their variables."""
    chosen: list[str] = []
    for variable in variables:
        key = variable if isinstance(variable, Variable) else Variable(variable)
        for field in FIELD_SETS.get(key, {}).get(detail.value, []):
            if field not in chosen:
                chosen.append(field)
    return chosen[:limit] or ["Tavg"]
