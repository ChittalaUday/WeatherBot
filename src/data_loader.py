"""
Data loader and dataset utilities for WeatherBot NLU training.
"""

import json
import pandas as pd
from typing import Tuple, List, Dict, Any
from pathlib import Path


def load_intents_csv(csv_path: str = "data/intents.csv") -> pd.DataFrame:
    """
    Loads intents CSV dataset and parses location/time JSON list strings.
    """
    df = pd.read_csv(csv_path)
    df["location"] = df["location"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    df["time"] = df["time"].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
    if "aggregation" not in df.columns:            # older CSVs predate Rule 2.3
        df["aggregation"] = "RAW"
    df["aggregation"] = df["aggregation"].fillna("RAW")
    return df


if __name__ == "__main__":
    df = load_intents_csv()
    print(f"Loaded {len(df)} prompts across {df['weather_intent'].nunique()} weather intents and {df['action'].nunique()} actions.")
    print("\nSample records:")
    print(df.head(5).to_string())
