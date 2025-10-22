import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from typing import Tuple, List

def build_model(df: pd.DataFrame, target: str = "pm25") -> Tuple[Pipeline, float, float]:
    # choose features: all columns except target
    features = [c for c in df.columns if c != target]
    df_train = df.dropna(subset=features + [target]).copy()
    X = df_train[features]
    y = df_train[target]
    # Time-aware split (no shuffle)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(n_estimators=200, max_depth=4, random_state=0))
    ])
    pipe.fit(X_train, y_train)
    # Save feature names for later reference
    pipe.feature_names_ = features
    train_mae = float(mean_absolute_error(y_train, pipe.predict(X_train)))
    val_mae = float(mean_absolute_error(y_val, pipe.predict(X_val)))
    return pipe, train_mae, val_mae

def _build_feature_row_for_next(df: pd.DataFrame, features: List[str], next_time: pd.Timestamp) -> pd.Series:
    """
    Construct a single feature vector for the next_time using persistence rules:
    - pm25 lags updated using the last known pm25 value(s)
    - rolling stats computed from last available pm25s
    - hour/dayofweek set from next_time
    - weather features persisted from last observed
    """
    last = df.iloc[-1].copy()
    row = {}
    # pm25 lags: find all pm25_lag_N columns and update
    lag_cols = [c for c in features if c.startswith("pm25_lag_")]
    # build mapping n->col name
    lag_map = {}
    for col in lag_cols:
        try:
            n = int(col.split("_")[-1])
            lag_map[n] = col
        except Exception:
            continue
    # For lag n, new value becomes previous lag n-1; lag_1 becomes last pm25
    for n, col in lag_map.items():
        if n == 1:
            row[col] = float(df["pm25"].iloc[-1])
        else:
            prev = lag_map.get(n-1)
            if prev and prev in df.columns:
                row[col] = float(df[prev].iloc[-1])
            else:
                row[col] = float(df["pm25"].iloc[-1])
    # rolling features
    if "pm25_roll_mean_3" in features:
        row["pm25_roll_mean_3"] = float(df["pm25"].iloc[-3:].mean())
    if "pm25_roll_std_3" in features:
        row["pm25_roll_std_3"] = float(df["pm25"].iloc[-3:].std())
    # time features
    if "hour" in features:
        row["hour"] = int(next_time.hour)
    if "dayofweek" in features:
        row["dayofweek"] = int(next_time.dayofweek)
    # temp_humidity
    if "temp_humidity" in features and ("temperature" in df.columns and "humidity" in df.columns):
        t = float(df["temperature"].iloc[-1])
        h = float(df["humidity"].iloc[-1])
        row["temp_humidity"] = t * (h / 100.0)
    # weather persistence for other weather features
    weather_cols = ["temperature", "humidity", "wind_speed", "wind_dir", "pressure"]
    for c in weather_cols:
        if c in features:
            row[c] = float(df[c].iloc[-1])
    # ensure all features present
    for f in features:
        if f not in row:
            # fallback to last known value if available, else zero
            if f in df.columns:
                row[f] = float(df[f].iloc[-1])
            else:
                row[f] = 0.0
    return pd.Series(row)

def recursive_forecast(model: Pipeline, recent_df: pd.DataFrame, steps: int = 24) -> pd.Series:
    """
    Recursive multi-step forecasting for `steps` hours.
    - model: trained pipeline with .feature_names_ property
    - recent_df: dataframe with features and target for last hours (used to seed lags)
    Returns pandas Series indexed by UTC timestamps.
    """
    df = recent_df.copy().sort_index()
    last_time = df.index.max()
    freq = pd.Timedelta("1H")
    preds = []
    idxs = []
    features = list(getattr(model, "feature_names_", []))
    # If model has no stored feature names, attempt to infer from pipeline after imputer
    if not features:
        # try to infer from recent_df columns minus target
        features = [c for c in recent_df.columns if c != "pm25"]
    for step in range(1, steps + 1):
        next_time = last_time + freq
        # build feature vector
        row = _build_feature_row_for_next(df, features, next_time)
        # model expects a DataFrame
        X_next = row.to_frame().T
        try:
            y_hat = float(model.predict(X_next)[0])
        except Exception:
            # fallback: use mean of last 3 pm25
            y_hat = float(df["pm25"].iloc[-3:].mean())
        # append predicted pm25 to df to update lags for next iteration
        new_row = X_next.copy()
        new_row["pm25"] = y_hat
        new_row.index = [next_time]
        df = pd.concat([df, new_row], axis=0)
        preds.append(y_hat)
        idxs.append(next_time)
        last_time = next_time
    return pd.Series(preds, index=pd.DatetimeIndex(idxs, tz="UTC"))
