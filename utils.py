# ml/utils.py
"""Utility helpers: timestamps and index normalizers."""
import datetime as dt
import pandas as pd
from typing import Any

def now_utc() -> dt.datetime:
    """Return current UTC datetime (naive)."""
    return dt.datetime.utcnow()

def ensure_hourly_index(df: pd.DataFrame, time_col: str = "timestamp_utc") -> pd.DataFrame:
    """
    Ensure the given dataframe has a clean hourly UTC DatetimeIndex.
    - Converts `time_col` to UTC datetimes
    - Sets it as index and resamples to hourly mean
    """
    df = df.copy()
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True)
        df = df.set_index(time_col).sort_index()
    else:
        # assume index is already datetime-like
        df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
    # resample to hourly and take mean (handles multiple measurements per hour)
    df = df.resample("H").mean()
    return df
