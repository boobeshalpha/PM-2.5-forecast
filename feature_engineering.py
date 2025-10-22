# ml/feature_engineering.py
"""
Feature engineering:
- interpolation + forward/backwards fill
- lag features
- rolling statistics
- time features
"""
import pandas as pd
import numpy as np
from typing import List

def create_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
    df = df.copy()
    # Create lagged pm25 features
    for lag in lags:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)
    # Rolling statistics (window=3)
    df["pm25_roll_mean_3"] = df["pm25"].rolling(window=3).mean()
    df["pm25_roll_std_3"] = df["pm25"].rolling(window=3).std().fillna(0)
    # Time features
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    # Interaction example
    df["temp_humidity"] = df["temperature"] * (df["humidity"] / 100.0)
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean + feature engineering pipeline:
    - time-based interpolation for missing weather
    - create lag/rolling/time features
    - drop rows with insufficient lag history for target
    """
    df = df.copy()
    # Prefer time interpolation, then forward-fill/backfill
    df = df.interpolate(method="time").ffill().bfill()
    df = create_lag_features(df)
    # After lags introduced, drop rows where pm25 is missing (target)
    df = df.dropna(subset=["pm25"])
    return df
