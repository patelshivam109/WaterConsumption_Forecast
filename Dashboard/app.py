import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os
import io

# Add src to path so we can import our modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.forecasting_model import WaterDemandForecaster
from src.anomaly_detector import AnomalyDetector

st.set_page_config(page_title="Water Demand Forecast & Analytics", layout="wide", page_icon="🌊")

# --- Custom CSS for Glassmorphism & Animations ---
st.markdown("""
<style>
/* Base Theme Adjustments */
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
    font-family: 'Inter', 'Roboto', sans-serif;
}

/* Glassmorphism Containers */
.glass-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    margin-bottom: 20px;
    transition: transform 0.3s ease;
}
.glass-container:hover {
    transform: translateY(-5px);
}

/* Titles and Headers */
h1, h2, h3, h4 {
    color: #00d2ff;
    font-weight: 700;
}
.main-title {
    font-size: 3rem;
    background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    animation: fadeInDown 1s ease-out;
}

/* Animations */
@keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes pulseGlow {
    0% { box-shadow: 0 0 10px rgba(255, 75, 75, 0.5); }
    50% { box-shadow: 0 0 20px rgba(255, 75, 75, 0.9); }
    100% { box-shadow: 0 0 10px rgba(255, 75, 75, 0.5); }
}

/* Alert Boxes */
.alert-box-spike {
    background: rgba(255, 75, 75, 0.15);
    border-left: 5px solid #ff4b4b;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
    animation: pulseGlow 2s infinite;
}
.alert-box-drop {
    background: rgba(255, 165, 0, 0.15);
    border-left: 5px solid #ffa500;
    padding: 15px;
    border-radius: 5px;
    margin-bottom: 10px;
}

/* Override Streamlit elements to match dark theme */
.stSelectbox label, .stDateInput label {
    color: #00d2ff !important;
}
</style>
""", unsafe_allow_html=True)

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

# --- Title ---
st.markdown('<div class="main-title">💧 Advanced Water Intelligence Hub</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #a0aec0; margin-bottom: 2rem;'>AI-Driven Forecasting & Anomaly Detection Dashboard</p>", unsafe_allow_html=True)

df = load_data()
forecaster, detector = load_model_and_detector()

df_processed = detector.detect_anomalies(df)
df_processed = detector.categorize_demand(df_processed)

# --- Sidebar Controls ---
with st.sidebar:
    st.markdown("### 🎛️ Control Panel")
    selected_region = st.selectbox("🌍 Select Region", df_processed['region'].unique())
    
    min_date = df_processed['date'].min().date()
    max_date = df_processed['date'].max().date()
    
    date_range = st.date_input(
        "📅 Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    st.markdown("---")
    st.markdown("### 📥 Export Data")
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        export_df = df_processed[(df_processed['region'] == selected_region) & 
                                 (df_processed['date'].dt.date >= start_date) & 
                                 (df_processed['date'].dt.date <= end_date)]
        
        csv = export_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"water_data_{selected_region}_{start_date}.csv",
            mime="text/csv",
        )

# Filter Data
if len(date_range) == 2:
    start_date, end_date = date_range
    region_data = df_processed[(df_processed['region'] == selected_region) & 
                               (df_processed['date'].dt.date >= start_date) & 
                               (df_processed['date'].dt.date <= end_date)].copy()
else:
    region_data = df_processed[df_processed['region'] == selected_region].copy()

# --- KPI Section ---
st.markdown("### 📊 Key Performance Indicators")
col1, col2, col3, col4 = st.columns(4)

avg_consumption = region_data['consumption_liters'].mean()
peak_consumption = region_data['consumption_liters'].max()
total_anomalies = region_data['is_anomaly'].sum()
latest_cat = region_data.iloc[-1]['demand_category'] if not region_data.empty else "N/A"

with col1:
    st.markdown(f"""
    <div class="glass-container" style="text-align: center;">
        <h4 style="margin:0; color:#a0aec0;">Avg Daily Demand</h4>
        <h2 style="margin:10px 0 0 0; color:#00d2ff;">{avg_consumption:,.0f} L</h2>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="glass-container" style="text-align: center;">
        <h4 style="margin:0; color:#a0aec0;">Peak Demand</h4>
        <h2 style="margin:10px 0 0 0; color:#ff7675;">{peak_consumption:,.0f} L</h2>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="glass-container" style="text-align: center;">
        <h4 style="margin:0; color:#a0aec0;">Anomalies Detected</h4>
        <h2 style="margin:10px 0 0 0; color:#ffeaa7;">{total_anomalies}</h2>
    </div>
    """, unsafe_allow_html=True)

with col4:
    cat_color = "#55efc4" if latest_cat == "Low" else "#ffeaa7" if latest_cat == "Medium" else "#ff7675"
    st.markdown(f"""
    <div class="glass-container" style="text-align: center;">
        <h4 style="margin:0; color:#a0aec0;">Current Status</h4>
        <h2 style="margin:10px 0 0 0; color:{cat_color};">{latest_cat}</h2>
    </div>
    """, unsafe_allow_html=True)

# --- Charts Section ---
col_chart, col_alerts = st.columns([2.5, 1])

with col_chart:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Historical & Predicted Consumption")
    
    fig = go.Figure()

    # Actual Data
    fig.add_trace(go.Scatter(x=region_data['date'], y=region_data['consumption_liters'], 
                             mode='lines', name='Actual Consumption', 
                             line=dict(color='#00d2ff', width=2), fill='tozeroy', fillcolor='rgba(0, 210, 255, 0.1)'))
                             
    # Model Predictions (if model exists)
    if hasattr(forecaster.model, 'predict'):
        eval_data = region_data.copy()
        eval_data['prediction'] = forecaster.predict(eval_data)
        fig.add_trace(go.Scatter(x=eval_data['date'], y=eval_data['prediction'], 
                                 mode='lines', name='AI Forecast', 
                                 line=dict(color='#ff9f43', width=2, dash='dash')))

    # Anomalies
    anomalies = region_data[region_data['is_anomaly'] == 1]
    spikes = anomalies[anomalies['anomaly_type'] == 'Spike']
    drops = anomalies[anomalies['anomaly_type'] == 'Drop']
    
    if not spikes.empty:
        fig.add_trace(go.Scatter(x=spikes['date'], y=spikes['consumption_liters'], 
                                 mode='markers', name='Spikes', 
                                 marker=dict(color='#ff4b4b', size=10, symbol='triangle-up', line=dict(color='white', width=1))))
    if not drops.empty:
        fig.add_trace(go.Scatter(x=drops['date'], y=drops['consumption_liters'], 
                                 mode='markers', name='Drops', 
                                 marker=dict(color='#feca57', size=10, symbol='triangle-down', line=dict(color='white', width=1))))

    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ffffff'),
        xaxis=dict(showgrid=False, linecolor='rgba(255,255,255,0.2)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', linecolor='rgba(255,255,255,0.2)'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_alerts:
    st.markdown('<div class="glass-container" style="height: 575px; overflow-y: auto;">', unsafe_allow_html=True)
    st.markdown("### 🚨 Active Alerts")
    
    recent_anomalies = anomalies.sort_values('date', ascending=False).head(8)
    
    if not recent_anomalies.empty:
        for _, row in recent_anomalies.iterrows():
            if row['anomaly_type'] == 'Spike':
                st.markdown(f"""
                <div class="alert-box-spike">
                    <strong>⚠️ {row['date'].strftime('%b %d, %Y')}</strong><br>
                    <span style="font-size:0.9em">Critical Spike: {row['consumption_liters']:,.0f} L</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="alert-box-drop">
                    <strong>📉 {row['date'].strftime('%b %d, %Y')}</strong><br>
                    <span style="font-size:0.9em">Sudden Drop: {row['consumption_liters']:,.0f} L</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 40px 20px; color: #55efc4;">
            <h3>✅ System Stable</h3>
            <p>No anomalies detected in the selected timeframe.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)
