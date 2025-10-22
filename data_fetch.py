# ml/data_fetch.py
"""
Data ingestion module:
- fetch_pm25_openaq: OpenAQ (primary) for PM2.5
- fetch_openmeteo: Open-Meteo combined air quality + weather (fallback / enrichment)
- make_synthetic_data: deterministic fallback for demos
"""
import numpy as np
import pandas as pd
import requests
from typing import Optional
from .utils import now_utc, ensure_hourly_index

OPENAQ_URL = "https://api.openaq.org/v2/measurements"
OPENMETEO_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
OPENMETEO_WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

def fetch_pm25_openaq(lat: float, lon: float, radius_km: int = 25, hours: int = 72) -> Optional[pd.DataFrame]:
    """
    Fetch PM2.5 measurements from OpenAQ within a radius and aggregate hourly.
    Returns DataFrame with index as UTC hourly timestamps and column 'pm25'
    """
    date_to = now_utc()
    date_from = date_to - pd.Timedelta(hours=hours)
    params = {
        "parameter": "pm25",
        "date_from": date_from.isoformat(),
        "date_to": date_to.isoformat(),
        "limit": 10000,
        "radius": int(radius_km * 1000),
        "coordinates": f"{lat},{lon}",
    }
    try:
        resp = requests.get(OPENAQ_URL, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        results = payload.get("results", [])
        if not results:
            return None
        rows = []
        for r in results:
            ts = r.get("date", {}).get("utc")
            val = r.get("value")
            if ts is None or val is None:
                continue
            rows.append({"timestamp_utc": ts, "pm25": val})
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df = ensure_hourly_index(df)
        return df[["pm25"]]
    except Exception:
        return None

def fetch_openmeteo(lat: float, lon: float, hours: int = 72) -> Optional[pd.DataFrame]:
    """
    Fetch hourly PM2.5 and weather features from Open-Meteo (air-quality endpoint).
    Returns DataFrame indexed by UTC hourly timestamps with columns:
      ['pm25','temperature','humidity','wind_speed','wind_dir','pressure']
    """
    end = now_utc()
    start = end - pd.Timedelta(hours=hours)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,surface_pressure",
        "start": start.strftime("%Y-%m-%dT%H:00"),
        "end": end.strftime("%Y-%m-%dT%H:00"),
        "timezone": "UTC"
    }
    try:
        resp = requests.get(OPENMETEO_AQ_URL, params=params, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        hourly = payload.get("hourly", {})
        if not hourly:
            return None
        df = pd.DataFrame(hourly)
        if "time" not in df.columns:
            return None
        df["timestamp_utc"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("timestamp_utc").sort_index()
        rename_map = {
            "pm2_5": "pm25",
            "temperature_2m": "temperature",
            "relativehumidity_2m": "humidity",
            "windspeed_10m": "wind_speed",
            "winddirection_10m": "wind_dir",
            "surface_pressure": "pressure"
        }
        df = df.rename(columns=rename_map)
        for c in ["pm25", "temperature", "humidity", "wind_speed", "wind_dir", "pressure"]:
            if c not in df.columns:
                df[c] = np.nan
        return df[["pm25", "temperature", "humidity", "wind_speed", "wind_dir", "pressure"]]
    except Exception:
        return None

def make_synthetic_data(hours: int = 120, seed: int = 42) -> pd.DataFrame:
    """
    Generate deterministic synthetic hourly PM2.5 + weather dataset for demos.
    """
    rng = pd.date_range(end=now_utc(), periods=hours, freq="H", tz="UTC")
    np.random.seed(seed)
    base = 30 + 10 * np.sin(np.linspace(0, 6 * np.pi, hours))
    noise = np.random.normal(scale=6, size=hours)
    pm25 = np.clip(base + noise + np.linspace(0, 3, hours), 1, None)
    temperature = 22 + 6 * np.sin(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 1, hours)
    humidity = np.clip(55 + 12 * np.cos(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 4, hours), 5, 100)
    wind_speed = np.abs(np.random.normal(2, 0.8, hours))
    wind_dir = np.random.uniform(0, 360, hours)
    pressure = 1013 + np.random.normal(0, 2, hours)
    df = pd.DataFrame({
        "timestamp_utc": rng,
        "pm25": pm25,
        "temperature": temperature,
        "humidity": humidity,
        "wind_speed": wind_speed,
        "wind_dir": wind_dir,
        "pressure": pressure
    }).set_index("timestamp_utc")
    return df
