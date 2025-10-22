# 🌫️ PM2.5 Forecasting and Air Quality Analysis

## 📘 Overview

This project focuses on **forecasting PM2.5 (Particulate Matter ≤ 2.5 micrometers)** levels using historical air quality data. PM2.5 is a major air pollutant that affects human health, and this project aims to help visualize, analyze, and predict air quality trends using data-driven techniques.

The system integrates **data fetching, preprocessing, time-series analysis, and forecasting models** (such as ARIMA, LSTM, or Prophet) and displays the results using **Streamlit dashboards**.

---

## 🎯 Objectives

* Fetch real-time or historical PM2.5 data from APIs (e.g., OpenAQ, AQICN).
* Clean and preprocess the raw data (handling missing values, timestamp conversion, etc.).
* Visualize trends in PM2.5 levels over time.
* Forecast future PM2.5 values using machine learning or statistical models.
* Provide an interactive dashboard for users to explore air quality insights.

---

## 🧠 Key Features

✅ **Live Data Fetching** – Retrieve PM2.5 data automatically from public APIs.
✅ **Data Cleaning & Preprocessing** – Convert timestamps, remove nulls, and standardize values.
✅ **Visualization** – Line charts, bar graphs, and heatmaps for time-based PM2.5 analysis.
✅ **Forecasting Models** – Predict future PM2.5 levels using ARIMA, LSTM, or Prophet.
✅ **Streamlit Dashboard** – Interactive UI to display analysis and forecasts.

---

## 🧩 Project Structure

```
pm25_forecast/
│
├── data/                       # Raw and cleaned data files
├── outputs/                    # Saved charts, model outputs, and logs
├── app.py                      # Streamlit dashboard
├── data_fetch.py               # PM2.5 data collection script
├── data_cleaning.py            # Cleaning and preprocessing
├── model_train.py              # Forecasting model training
├── model_predict.py            # Prediction and evaluation
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pm25-forecast.git
cd pm25-forecast
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Fetch or load PM2.5 data

You can use live fetching (if API key is available):

```bash
python data_fetch.py
```

Or load historical data from CSV in `/data/`.

### 4. Run data preprocessing

```bash
python data_cleaning.py
```

### 5. Train the forecasting model

```bash
python model_train.py
```

### 6. Run the Streamlit dashboard

```bash
streamlit run app.py
```

---

## 📊 Visual Outputs

* **Trend Analysis:** Daily/Hourly PM2.5 variation plots.
* **Forecasting Plot:** Predicted vs Actual PM2.5 graph.
* **AQI Heatmap:** Geographical or temporal distribution of PM2.5.

All visual outputs are saved in `/outputs/`.

---

## 🧮 Technologies Used

* **Python 3.10+**
* **Pandas, NumPy** – Data processing
* **Matplotlib, Seaborn, Plotly** – Visualization
* **Scikit-learn / Statsmodels / Prophet / TensorFlow** – Forecasting models
* **Streamlit** – Dashboard and visualization interface
* **OpenAQ / AQICN APIs** – Data source for PM2.5 readings

---

## 🌍 Real-world Applications

* Urban air quality monitoring
* Pollution forecasting for environmental authorities
* Health risk assessment and early warnings
* Smart city environmental dashboards

---

## 📈 Example Insights

* Identify the most polluted hours/days.
* Detect seasonal trends (e.g., higher PM2.5 in winter).
* Forecast PM2.5 concentration for the next 24–48 hours.

---

## 🧑‍💻 Author

**Boobesh S**
PM2.5 Forecasting Project | Streamlit Dashboard Developer
📍 Tamil Nadu, India

---

## 🪶 License

This project is open-source under the **MIT License**.

---

## 💬 Future Enhancements

* Integrate multiple pollutants (NO₂, CO, SO₂, O₃).
* Incorporate weather data for better forecasts.
* Deploy on cloud (e.g., Streamlit Cloud, AWS).
* Add alert system for hazardous levels.
