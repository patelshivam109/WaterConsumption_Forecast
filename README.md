# 💧 Water Consumption Forecasting & Anomaly Detection System

## 📌 Project Overview
This project is an end-to-end machine learning system designed to forecast household and district-level water consumption using historical usage records, seasonal patterns, and temporal trends. 

The system predicts future water demand, identifies unusual consumption spikes/drops, and provides a fully interactive UI. This enables utility providers to optimize water distribution, perform resource planning, and prevent infrastructure damage (e.g., detecting pipe leaks via consumption spikes).

## ✨ Features
1. **Time-Series Forecasting**: Evaluates 7 distinct algorithms (Prophet, SARIMA, LSTM, GRU, Random Forest, LightGBM, XGBoost) to predict daily consumption.
2. **Anomaly Detection System**: A rolling Z-Score statistical module that detects critical usage surges (Spikes) and sudden drops.
3. **Demand Categorization**: Automatically bands consumption into `Low`, `Medium`, and `High` zones.
4. **Interactive Dashboard**: A modernized, multi-page Streamlit application featuring glassmorphism design, real-time metric tiles, plotly area charts, and dynamic CSV data exports.

---

## 📂 Project Structure

```text
WaterConsumption_Forecast/
│
├── Dashboard/
│   ├── app.py                         # Main Streamlit dashboard application
│   └── pages/
│       ├── 01_EDA_Analytics.py        # Analytics & Distributions
│       └── 02_Model_Evaluation.py     # Algorithm comparison charts
│
├── Dataset/
│   ├── raw/                           # Original Kaggle dataset
│   └── processed/                     # Train/Val/Test splits & feature-engineered data
│
├── models/
│   └── random_forest_model.pkl         # Serialized winning tree-based model (Random Forest)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA and visualizations
│   ├── 02_feature_engineering.ipynb   # Rolling windows, lags, time splits
│   ├── 03_model_training.ipynb        # 7-Algorithm training & evaluation
│   └── 04_anomaly_detection.ipynb     # Statistical spike/drop detection
│
├── reports/
│   ├── EDA_Report.md
│   ├── Forecast_Evaluation_Report.md
│   └── Water_Demand_Report.md
│
├── src/
│   ├── anomaly_detector.py            # Core logic for Z-score anomaly detection
│   ├── data_preprocessing.py          # Data cleaning & pipeline logic
│   └── forecasting_model.py           # Model loading & inference wrapper
│
├── requirements.txt                   # Python dependencies
└── generate_notebooks.py              # Script to programmatically generate Jupyter notebooks
```

---

## 🚀 Setup & Installation

**1. Clone the repository**
```bash
git clone https://github.com/patelshivam109/WaterConsumption_Forecast.git
cd WaterConsumption_Forecast
```

**2. Create a virtual environment**
```bash
python -m venv .venv
# Activate on Windows:
.\.venv\Scripts\activate
# Activate on Mac/Linux:
source .venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Running the Dashboard

To launch the multi-page analytics dashboard, run the following command from the root directory:

```bash
streamlit run Dashboard/app.py
```
*The dashboard will automatically open in your browser at `http://localhost:8501`.*

---

## 🤖 Model Evaluation Results

We evaluated 7 distinct models on our chronologically-split test set. The statistical time-series models performed the best due to the highly cyclical nature of human water consumption.

| Rank | Model | RMSE (Liters) | MAPE (%) |
|---|---|---|---|
| 🥇 | **Prophet** | 2,047 | 14.19% |
| 🥈 | **SARIMA** | 2,363 | 16.22% |
| 🥉 | **Random Forest** | 4,128 | 34.99% |

*(For full evaluation details, including deep learning (LSTM/GRU) performance, refer to `reports/Forecast_Evaluation_Report.md`)*

---

## 🚨 Anomaly Detection Logic
The system uses a **7-Day Rolling Z-Score** to flag anomalies.
- If the consumption deviates by `> +2 Standard Deviations` from the rolling mean, a **Spike** alert is generated (potential leak or extreme demand).
- If it deviates by `< -2 Standard Deviations`, a **Drop** alert is generated (potential outage or meter failure).
