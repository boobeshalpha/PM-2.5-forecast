# ml/anomaly_detection.py
"""
IsolationForest-based anomaly detection for multivariate AQ readings.
"""
from sklearn.ensemble import IsolationForest
import pandas as pd
from typing import Tuple

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> Tuple[pd.Series, IsolationForest]:
    """
    Fit IsolationForest on small feature set and return boolean mask of anomalies.
    - contamination: expected proportion of anomalies
    """
    features = ["pm25", "temperature", "humidity", "wind_speed", "pressure"]
    X = df[features].fillna(method="ffill").fillna(0).values
    iso = IsolationForest(contamination=contamination, random_state=0)
    iso.fit(X)
    preds = iso.predict(X)  # -1 anomaly, 1 normal
    anomaly_mask = pd.Series(preds == -1, index=df.index)
    return anomaly_mask, iso
