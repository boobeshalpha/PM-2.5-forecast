"""
Live PM2.5 Nowcast & Forecast Streamlit App - Enhanced UI
Features modern design with improved visualizations and user experience
"""

from typing import Tuple, Optional, List
import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import joblib
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ---- Config / Constants ----
CITIES = {
    "Salem": (11.6643, 78.1460),
    "Erode": (11.3410, 77.7172),
    "Namakkal": (11.2187, 78.1653),
    "Coimbatore": (11.0168, 76.9558),
    "Chennai": (13.0827, 80.2707),
    "Mallasamudram": (11.4910, 78.0229)
}

DEFAULT_RADIUS_KM = 25
OPENAQ_URL = "https://api.openaq.org/v2/measurements"
OPENMETEO_HIST_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# AQI Categories
AQI_LEVELS = [
    (0, 12, "Good", "#00E400", "Air quality is satisfactory"),
    (12.1, 35.4, "Moderate", "#FFFF00", "Acceptable for most people"),
    (35.5, 55.4, "Unhealthy for Sensitive", "#FF7E00", "Sensitive groups may experience effects"),
    (55.5, 150.4, "Unhealthy", "#FF0000", "Everyone may experience effects"),
    (150.5, 250.4, "Very Unhealthy", "#8F3F97", "Health alert: everyone may experience serious effects"),
    (250.5, 500, "Hazardous", "#7E0023", "Emergency conditions: entire population affected")
]

def get_aqi_info(pm25):
    for low, high, category, color, description in AQI_LEVELS:
        if low <= pm25 <= high:
            return category, color, description
    return "Hazardous", "#7E0023", "Extremely dangerous conditions"

# ---- Utilities ----
def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def ensure_hourly_index(df: pd.DataFrame, time_col: str = "timestamp_utc") -> pd.DataFrame:
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.set_index(time_col).sort_index()
    df = df.resample("h").mean()
    return df

# ---- Data Fetching ----
@st.cache_data(ttl=600)
def fetch_pm25_openaq(lat: float, lon: float, radius_km: int = 25, hours: int = 72) -> Optional[pd.DataFrame]:
    params = {
        "parameter": "pm25",
        "date_from": (now_utc() - dt.timedelta(hours=hours)).isoformat(),
        "date_to": now_utc().isoformat(),
        "limit": 10000,
        "radius": int(radius_km * 1000),
        "coordinates": f"{lat},{lon}"
    }
    try:
        r = requests.get(OPENAQ_URL, params=params, timeout=10)
        r.raise_for_status()
        payload = r.json()
        results = payload.get("results", [])
        if not results:
            return None
        rows = [{"timestamp_utc": rec.get("date", {}).get("utc"), "pm25": rec.get("value")} for rec in results]
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        df = ensure_hourly_index(df)
        return df[["pm25"]]
    except Exception:
        return None

@st.cache_data(ttl=600)
def fetch_weather_and_pm_open_meteo(lat: float, lon: float, hours: int = 72) -> Optional[pd.DataFrame]:
    end = now_utc()
    start = end - dt.timedelta(hours=hours)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm2_5,temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,surface_pressure",
        "start": start.strftime("%Y-%m-%dT%H:00"),
        "end": end.strftime("%Y-%m-%dT%H:00"),
        "timezone": "UTC"
    }
    try:
        r = requests.get(OPENMETEO_HIST_URL, params=params, timeout=10)
        r.raise_for_status()
        payload = r.json()
        hourly = payload.get("hourly", {})
        if not hourly:
            return None
        df = pd.DataFrame(hourly)
        if "time" in df.columns:
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
            for c in ["pm25","temperature","humidity","wind_speed","wind_dir","pressure"]:
                if c not in df.columns:
                    df[c] = np.nan
            return df[["pm25","temperature","humidity","wind_speed","wind_dir","pressure"]]
        return None
    except Exception:
        return None

def make_synthetic_data(hours: int = 120, seed: int = 42) -> pd.DataFrame:
    rng = pd.date_range(end=now_utc(), periods=hours, freq="h", tz="UTC")
    np.random.seed(seed)
    base = 30 + 10 * np.sin(np.linspace(0, 6 * np.pi, hours))
    noise = np.random.normal(scale=6, size=hours)
    pm25 = np.clip(base + noise + np.linspace(0, 5, hours), 1, None)
    temperature = 20 + 6 * np.sin(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 1, hours)
    humidity = np.clip(50 + 20 * np.cos(np.linspace(0, 2 * np.pi, hours)) + np.random.normal(0, 5, hours), 5, 100)
    wind_speed = np.abs(np.random.normal(2, 0.8, hours))
    wind_dir = np.random.uniform(0, 360, hours)
    pressure = 1013 + np.random.normal(0, 3, hours)
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

# ---- Feature Engineering ----
def create_lag_features(df: pd.DataFrame, lags: List[int] = [1,2,3,6,12,24]) -> pd.DataFrame:
    df = df.copy()
    for lag in lags:
        df[f"pm25_lag_{lag}"] = df["pm25"].shift(lag)
    df["pm25_roll_mean_3"] = df["pm25"].rolling(3).mean()
    df["pm25_roll_std_3"] = df["pm25"].rolling(3).std().fillna(0)
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["temp_humidity"] = df["temperature"] * (df["humidity"]/100.0)
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.interpolate(method="time").ffill().bfill()
    df = create_lag_features(df)
    df = df.dropna(subset=["pm25"])
    return df

# ---- Anomaly Detection ----
def detect_anomalies(df: pd.DataFrame, contamination: float = 0.02) -> Tuple[pd.Series, IsolationForest]:
    features = ["pm25","temperature","humidity","wind_speed","pressure"]
    X = df[features].ffill().fillna(0).values
    iso = IsolationForest(contamination=contamination, random_state=0)
    iso.fit(X)
    preds = iso.predict(X)
    anomaly_mask = pd.Series(preds == -1, index=df.index)
    return anomaly_mask, iso

# ---- Model Building & Forecasting ----
def build_and_train_model(df: pd.DataFrame, target_col: str="pm25") -> Tuple[Pipeline,float,float]:
    exclude = [target_col]
    features = [c for c in df.columns if c not in exclude and c != "anomaly"]
    data = df.dropna(subset=features+[target_col])
    X = data[features]
    y = data[target_col]
    X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,shuffle=False)
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("gbr", GradientBoostingRegressor(n_estimators=200,max_depth=4,random_state=0))
    ])
    pipeline.fit(X_train,y_train)
    train_mae = float(mean_absolute_error(y_train, pipeline.predict(X_train)))
    val_mae = float(mean_absolute_error(y_val, pipeline.predict(X_val)))
    return pipeline, train_mae, val_mae

def recursive_forecast(model: Pipeline, recent_df: pd.DataFrame, steps: int = 24) -> pd.Series:
    df = recent_df.copy().sort_index()
    last_time = df.index.max()
    freq = pd.Timedelta("1h")
    preds = []
    idxs = []
    for step in range(1, steps+1):
        next_time = last_time + freq
        row = {}
        last_pm25 = df["pm25"].iloc[-1]
        for col in df.columns:
            if col.startswith("pm25_lag_"):
                n = int(col.split("_")[-1])
                row[col] = last_pm25 if n==1 else df[f"pm25_lag_{n-1}"].iloc[-1] if f"pm25_lag_{n-1}" in df.columns else last_pm25
            elif col in ["pm25_roll_mean_3","pm25_roll_std_3"]:
                recent_pm = df["pm25"].iloc[-3:].values
                row[col] = float(np.mean(recent_pm)) if recent_pm.size>0 else float(last_pm25)
            elif col=="hour":
                row[col] = next_time.hour
            elif col=="dayofweek":
                row[col] = next_time.dayofweek
            elif col=="temp_humidity":
                t,h = df["temperature"].iloc[-1], df["humidity"].iloc[-1]
                row[col] = t*(h/100.0)
            else:
                row[col] = df[col].iloc[-1] if col in df.columns else 0.0
        X_next = pd.DataFrame([row], index=[next_time])
        try:
            y_hat = model.predict(X_next)[0]
        except:
            y_hat = float(df["pm25"].iloc[-3:].mean())
        new_row = X_next.copy()
        new_row["pm25"] = y_hat
        df = pd.concat([df,new_row])
        last_time = next_time
        preds.append(y_hat)
        idxs.append(next_time)
    return pd.Series(preds,index=pd.DatetimeIndex(idxs,tz="UTC"))

# ---- Streamlit UI ----
st.set_page_config(page_title="PM2.5 Air Quality Monitor", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 24px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🌍 PM2.5 Air Quality Monitor & Forecast</h1>', unsafe_allow_html=True)
st.markdown("Real-time air quality monitoring with machine learning predictions")

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=AirQuality+AI", use_container_width=True)
    st.markdown("### ⚙️ Configuration")
    
    city = st.selectbox("📍 Select City", list(CITIES.keys()), index=0)
    lat, lon = CITIES[city]
    
    with st.expander("🔧 Advanced Settings"):
        radius_km = st.number_input("Search radius (km)", min_value=1, max_value=100, value=DEFAULT_RADIUS_KM)
        history_hours = st.slider("Historical data (hours)", 48, 240, 120, step=24)
        forecast_hours = st.slider("Forecast horizon (hours)", 6, 48, 24, step=6)
        contamination = st.slider("Anomaly sensitivity", 0.0, 0.2, 0.02, 0.01)
        retrain = st.checkbox("Retrain model", value=True)
    
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info("""
    This app uses machine learning to predict PM2.5 levels:
    - **Data**: OpenAQ & Open-Meteo APIs
    - **Model**: Gradient Boosting
    - **Detection**: Isolation Forest
    """)

# Data fetching
with st.spinner("🔄 Fetching latest data..."):
    df_pm = fetch_pm25_openaq(lat, lon, radius_km, history_hours)
    df_om = None
    if df_pm is None:
        df_om = fetch_weather_and_pm_open_meteo(lat, lon, history_hours)
    if df_pm is not None:
        df_weather = fetch_weather_and_pm_open_meteo(lat, lon, history_hours)
        if df_weather is not None:
            merged = df_weather.copy()
            merged["pm25"] = df_pm["pm25"].reindex(merged.index).fillna(merged["pm25"])
            df = merged
        else:
            df = df_pm.copy()
            for col in ["temperature","humidity","wind_speed","wind_dir","pressure"]:
                df[col] = np.nan
    elif df_om is not None:
        df = df_om.copy()
    else:
        st.warning("⚠️ Using synthetic data for demonstration")
        df = make_synthetic_data(history_hours)

# Prepare features
df_features = prepare_features(df)
anomaly_mask, iso_model = detect_anomalies(df_features, contamination=contamination)
df_features["anomaly"] = anomaly_mask

# Current conditions
current_pm25 = df_features["pm25"].iloc[-1]
category, color, description = get_aqi_info(current_pm25)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}99 0%, {color} 100%); padding: 1.5rem; border-radius: 12px; color: white;">
        <h3 style="margin:0; font-size:1rem;">Current PM2.5</h3>
        <h1 style="margin:0.5rem 0; font-size:2.5rem;">{current_pm25:.1f}</h1>
        <p style="margin:0;">µg/m³</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #667eea99 0%, #667eea 100%); padding: 1.5rem; border-radius: 12px; color: white;">
        <h3 style="margin:0; font-size:1rem;">Air Quality</h3>
        <h1 style="margin:0.5rem 0; font-size:2rem;">{category}</h1>
        <p style="margin:0; font-size:0.9rem;">{description}</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    temp = df_features["temperature"].iloc[-1]
    humidity = df_features["humidity"].iloc[-1]
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #f093fb99 0%, #f5576c 100%); padding: 1.5rem; border-radius: 12px; color: white;">
        <h3 style="margin:0; font-size:1rem;">Temperature</h3>
        <h1 style="margin:0.5rem 0; font-size:2.5rem;">{temp:.1f}°C</h1>
        <p style="margin:0;">Humidity: {humidity:.0f}%</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    wind = df_features["wind_speed"].iloc[-1]
    anomaly_count = int(anomaly_mask.sum())
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4facfe99 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 12px; color: white;">
        <h3 style="margin:0; font-size:1rem;">Wind Speed</h3>
        <h1 style="margin:0.5rem 0; font-size:2.5rem;">{wind:.1f}</h1>
        <p style="margin:0;">m/s • {anomaly_count} anomalies</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📈 Forecast", "📊 Historical Data", "🔍 Analysis", "⚙️ Model Details"])

with tab1:
    st.markdown("### 🔮 PM2.5 Forecast")
    
    # Train/load model
    model = None
    if retrain:
        with st.spinner("🤖 Training model..."):
            try:
                model, train_mae, val_mae = build_and_train_model(df_features)
                joblib.dump(model, "gbr_pipeline.joblib")
                st.success(f"✅ Model trained | Train MAE: {train_mae:.3f} | Validation MAE: {val_mae:.3f}")
            except Exception as e:
                st.error(f"❌ Training failed: {e}")
    else:
        try:
            model = joblib.load("gbr_pipeline.joblib")
            st.info("📦 Using cached model")
        except:
            st.warning("⚠️ No cached model found")
    
    if model is not None:
        # Nowcast
        recent_row = df_features.tail(1)
        nowcast = float(model.predict(recent_row.drop(columns=["pm25","anomaly"],errors="ignore"))[0])
        nowcast_cat, nowcast_color, _ = get_aqi_info(nowcast)
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {nowcast_color}99 0%, {nowcast_color} 100%); 
                        padding: 2rem; border-radius: 12px; color: white; text-align: center;">
                <h3 style="margin:0;">Next Hour Prediction</h3>
                <h1 style="margin:1rem 0; font-size:3rem;">{nowcast:.1f}</h1>
                <h2 style="margin:0; font-size:1.5rem;">{nowcast_cat}</h2>
                <p style="margin:0.5rem 0;">µg/m³ PM2.5</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Forecast
        forecast_series = recursive_forecast(model, df_features.tail(48), steps=forecast_hours)
        
        # Create combined chart
        fig = go.Figure()
        
        # Historical data
        hist_data = df_features["pm25"].tail(72)
        fig.add_trace(go.Scatter(
            x=hist_data.index,
            y=hist_data.values,
            mode='lines',
            name='Historical',
            line=dict(color='#667eea', width=2),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)'
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_series.index,
            y=forecast_series.values,
            mode='lines',
            name='Forecast',
            line=dict(color='#f5576c', width=2, dash='dash'),
            fill='tozeroy',
            fillcolor='rgba(245, 87, 108, 0.1)'
        ))
        
        # Add AQI reference lines
        for low, high, category, clr, _ in AQI_LEVELS[:3]:
            fig.add_hline(y=high, line_dash="dot", line_color=clr, opacity=0.3,
                         annotation_text=category, annotation_position="right")
        
        fig.update_layout(
            title="PM2.5 Historical & Forecast",
            xaxis_title="Time (UTC)",
            yaxis_title="PM2.5 (µg/m³)",
            hovermode='x unified',
            height=500,
            template="plotly_white",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        with col2:
            st.markdown("#### 📅 Hourly Forecast")
            fc_df = forecast_series.head(12).to_frame()
            fc_df.columns = ["PM2.5"]
            fc_df["Time"] = fc_df.index.strftime("%H:%M %d-%b")
            fc_df["Category"] = fc_df["PM2.5"].apply(lambda x: get_aqi_info(x)[0])
            fc_df = fc_df[["Time", "PM2.5", "Category"]]
            st.dataframe(fc_df.style.format({"PM2.5": "{:.2f}"}), use_container_width=True, height=300)

with tab2:
    st.markdown("### 📊 Historical Trends")
    
    # Multi-variable chart
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=("PM2.5 Levels", "Temperature & Humidity", "Wind Speed"),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    recent = df_features.tail(168)  # Last week
    
    # PM2.5
    fig.add_trace(go.Scatter(x=recent.index, y=recent["pm25"], name="PM2.5",
                            line=dict(color='#667eea', width=2)), row=1, col=1)
    
    # Anomalies
    anomalies = recent[recent["anomaly"]]
    if len(anomalies) > 0:
        fig.add_trace(go.Scatter(x=anomalies.index, y=anomalies["pm25"], mode='markers',
                                name="Anomalies", marker=dict(color='red', size=8, symbol='x')), row=1, col=1)
    
    # Temperature & Humidity
    fig.add_trace(go.Scatter(x=recent.index, y=recent["temperature"], name="Temperature",
                            line=dict(color='#f5576c', width=2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=recent.index, y=recent["humidity"], name="Humidity",
                            line=dict(color='#4facfe', width=2)), row=2, col=1)
    
    # Wind
    fig.add_trace(go.Scatter(x=recent.index, y=recent["wind_speed"], name="Wind Speed",
                            line=dict(color='#00f2fe', width=2), fill='tozeroy'), row=3, col=1)
    
    fig.update_xaxes(title_text="Time (UTC)", row=3, col=1)
    fig.update_yaxes(title_text="µg/m³", row=1, col=1)
    fig.update_yaxes(title_text="°C / %", row=2, col=1)
    fig.update_yaxes(title_text="m/s", row=3, col=1)
    
    fig.update_layout(height=800, showlegend=True, template="plotly_white", hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average PM2.5 (24h)", f"{recent['pm25'].tail(24).mean():.2f} µg/m³")
    with col2:
        st.metric("Max PM2.5 (7d)", f"{recent['pm25'].max():.2f} µg/m³")
    with col3:
        st.metric("Min PM2.5 (7d)", f"{recent['pm25'].min():.2f} µg/m³")

with tab3:
    st.markdown("### 🔍 Deep Dive Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🕐 Hourly Patterns")
        hourly_avg = df_features.groupby("hour")["pm25"].mean().reset_index()
        fig = px.bar(hourly_avg, x="hour", y="pm25", 
                    labels={"hour": "Hour of Day", "pm25": "Average PM2.5"},
                    color="pm25", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 📅 Daily Patterns")
        daily_avg = df_features.groupby("dayofweek")["pm25"].mean().reset_index()
        daily_avg["day"] = daily_avg["dayofweek"].map({0:"Mon", 1:"Tue", 2:"Wed", 3:"Thu", 4:"Fri", 5:"Sat", 6:"Sun"})
        fig = px.bar(daily_avg, x="day", y="pm25",
                    labels={"day": "Day of Week", "pm25": "Average PM2.5"},
                    color="pm25", color_continuous_scale="RdYlGn_r")
        fig.update_layout(height=350, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### 🌡️ Correlation Analysis")
    corr_vars = ["pm25", "temperature", "humidity", "wind_speed", "pressure"]
    corr_data = df_features[corr_vars].corr()
    fig = px.imshow(corr_data, text_auto=".2f", aspect="auto",
                    color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    if anomaly_mask.sum() > 0:
        st.markdown("#### ⚠️ Detected Anomalies")
        anom_df = df_features[df_features["anomaly"]][["pm25", "temperature", "humidity", "wind_speed"]].tail(10)
        st.dataframe(anom_df.style.format("{:.2f}"), use_container_width=True)

with tab4:
    st.markdown("### ⚙️ Model Performance & Details")
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Model Metrics")
            if retrain and train_mae and val_mae:
                metrics_df = pd.DataFrame({
                    "Metric": ["Training MAE", "Validation MAE", "Improvement"],
                    "Value": [f"{train_mae:.3f}", f"{val_mae:.3f}", f"{((train_mae-val_mae)/train_mae*100):.1f}%"]
                })
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            st.markdown("#### 🔧 Model Configuration")
            config_df = pd.DataFrame({
                "Parameter": ["Algorithm", "Estimators", "Max Depth", "Preprocessing"],
                "Value": ["Gradient Boosting", "200", "4", "StandardScaler + Imputer"]
            })
            st.dataframe(config_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("#### 📊 Feature Importance (Top 10)")
            try:
                gbr = model.named_steps["gbr"]
                fi = gbr.feature_importances_
                feature_names = model.named_steps["scaler"].get_feature_names_out(
                    model.named_steps["imputer"].feature_names_in_
                )
                fi_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": fi
                }).sort_values("Importance", ascending=False).head(10)
                
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h',
                            color="Importance", color_continuous_scale="Viridis")
                fig.update_layout(height=400, template="plotly_white", showlegend=False)
                fig.update_yaxes(categoryorder='total ascending')
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not generate feature importance: {e}")
        
        # Model predictions vs actual
        st.markdown("#### 📈 Model Predictions vs Actual")
        try:
            recent_for_pred = df_features.tail(100).dropna()
            X_test = recent_for_pred.drop(columns=["pm25", "anomaly"], errors="ignore")
            y_test = recent_for_pred["pm25"]
            predictions = model.predict(X_test)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test.index, y=y_test.values, 
                                    mode='lines', name='Actual',
                                    line=dict(color='#667eea', width=2)))
            fig.add_trace(go.Scatter(x=y_test.index, y=predictions, 
                                    mode='lines', name='Predicted',
                                    line=dict(color='#f5576c', width=2, dash='dot')))
            
            fig.update_layout(
                title="Model Performance on Recent Data",
                xaxis_title="Time (UTC)",
                yaxis_title="PM2.5 (µg/m³)",
                height=400,
                template="plotly_white",
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Residuals
            residuals = y_test.values - predictions
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Absolute Error", f"{np.mean(np.abs(residuals)):.3f}")
            with col2:
                st.metric("Root Mean Squared Error", f"{np.sqrt(np.mean(residuals**2)):.3f}")
            with col3:
                st.metric("R² Score", f"{1 - (np.sum(residuals**2) / np.sum((y_test.values - np.mean(y_test.values))**2)):.3f}")
        except Exception as e:
            st.warning(f"Could not generate predictions: {e}")
    else:
        st.warning("⚠️ No model available. Please enable 'Retrain model' in settings.")
    
    st.markdown("#### 📚 Dataset Information")
    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Total Records", len(df_features))
    with info_cols[1]:
        st.metric("Features", df_features.shape[1] - 2)  # Excluding pm25 and anomaly
    with info_cols[2]:
        st.metric("Time Range", f"{(df_features.index.max() - df_features.index.min()).days} days")
    with info_cols[3]:
        st.metric("Anomalies Detected", int(anomaly_mask.sum()))

# Footer
st.markdown("---")
current_time = now_utc().strftime("%Y-%m-%d %H:%M UTC")
footer_html = f"""
<div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 12px; margin-top: 2rem;">
    <h3 style="margin: 0 0 1rem 0;">🌟 About This Application</h3>
    <p style="margin: 0.5rem 0; color: #666;">
        This air quality monitoring system uses advanced machine learning to provide accurate PM2.5 forecasts.
        Data is sourced from OpenAQ and Open-Meteo APIs, with updates every 10 minutes.
    </p>
    <p style="margin: 0.5rem 0; color: #666;">
        <strong>Technologies:</strong> Gradient Boosting Regressor • Isolation Forest • Real-time Data Processing
    </p>
    <p style="margin: 1rem 0 0 0; font-size: 0.9rem; color: #999;">
        🔄 Last updated: {current_time} | 
        📍 Location: {city} ({lat:.4f}, {lon:.4f})
    </p>
</div>
"""
st.markdown(footer_html, unsafe_allow_html=True)