import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.forecasting_model import WaterDemandForecaster
from src.anomaly_detector import AnomalyDetector

st.set_page_config(page_title="Water Consumption Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/processed/water_consumption_processed.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

@st.cache_resource
def load_model_and_detector():
    forecaster = WaterDemandForecaster()
    try:
        forecaster.load_model()
    except FileNotFoundError:
        st.warning("Model not found. Please train the model first.")
    
    detector = AnomalyDetector()
    return forecaster, detector

st.title("💧 Water Utility Monitoring & Forecasting Dashboard")

df = load_data()
forecaster, detector = load_model_and_detector()

# Process data for anomalies and categories
df_processed = detector.detect_anomalies(df)
df_processed = detector.categorize_demand(df_processed)

# --- Sidebar ---
st.sidebar.header("Filters")
selected_region = st.sidebar.selectbox("Select Region", df_processed['region'].unique())

region_data = df_processed[df_processed['region'] == selected_region].copy()

# --- Main Dashboard ---
col1, col2, col3 = st.columns(3)
with col1:
    avg_consumption = region_data['consumption_liters'].mean()
    st.metric("Avg Daily Consumption", f"{avg_consumption:,.0f} L")
with col2:
    total_anomalies = region_data['is_anomaly'].sum()
    st.metric("Total Anomalies Detected", total_anomalies)
with col3:
    latest_cat = region_data.iloc[-1]['demand_category']
    st.metric("Current Demand Level", latest_cat)

st.markdown("### Historical Consumption & Anomalies")
fig = go.Figure()

# Plot actual consumption
fig.add_trace(go.Scatter(x=region_data['date'], y=region_data['consumption_liters'], 
                         mode='lines', name='Actual Consumption', line=dict(color='blue')))

# Plot rolling mean
fig.add_trace(go.Scatter(x=region_data['date'], y=region_data['rolling_mean_7'], 
                         mode='lines', name='7-Day Rolling Mean', line=dict(color='gray', dash='dash')))

# Highlight anomalies
anomalies = region_data[region_data['is_anomaly'] == 1]
if not anomalies.empty:
    fig.add_trace(go.Scatter(x=anomalies['date'], y=anomalies['consumption_liters'], 
                             mode='markers', name='Anomaly', 
                             marker=dict(color='red', size=8, symbol='x')))

fig.update_layout(height=400, xaxis_title="Date", yaxis_title="Consumption (Liters)")
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Forecasting vs Actual (Last 30 Days)")
# Get last 30 days for evaluation visualization
eval_data = region_data.tail(30).copy()
if hasattr(forecaster.model, 'predict'):
    eval_data['prediction'] = forecaster.predict(eval_data)
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=eval_data['date'], y=eval_data['consumption_liters'], 
                             mode='lines+markers', name='Actual', line=dict(color='blue')))
    fig2.add_trace(go.Scatter(x=eval_data['date'], y=eval_data['prediction'], 
                             mode='lines+markers', name='Predicted', line=dict(color='orange')))
    fig2.update_layout(height=350, xaxis_title="Date", yaxis_title="Consumption (Liters)")
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("Forecasting model not loaded. Train the model to view predictions.")

# Early Warning Alerts
st.markdown("### 🚨 Early Warning Alerts")
recent_anomalies = anomalies.tail(5)
if not recent_anomalies.empty:
    for _, row in recent_anomalies.iterrows():
        color = "red" if row['anomaly_type'] == 'Spike' else "orange"
        st.markdown(f"**{row['date'].strftime('%Y-%m-%d')}**: {row['anomaly_type']} detected! "
                    f"Consumption was {row['consumption_liters']:,.0f} L "
                    f"(Expected ~{row['rolling_mean_7']:,.0f} L)")
else:
    st.success("No recent anomalies detected.")
