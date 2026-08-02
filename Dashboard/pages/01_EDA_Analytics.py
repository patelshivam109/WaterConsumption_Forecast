import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="EDA & Analytics", layout="wide", page_icon="📈")

# Inherit CSS from main app
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: #ffffff;
}
.glass-container {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 20px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 20px;
}
h1, h2, h3 { color: #00d2ff; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("Dataset/raw/water_consumption_forecasting.csv")
    df['date'] = pd.to_datetime(df['date'])
    return df

st.title("📈 Exploratory Data Analysis")
df = load_data()

st.markdown('<div class="glass-container">', unsafe_allow_html=True)
st.markdown("### Consumption Distribution")
fig_hist = px.histogram(df, x="consumption_liters", nbins=40, color_discrete_sequence=['#00d2ff'])
fig_hist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
st.plotly_chart(fig_hist, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### Region Boxplot")
    fig_box = px.box(df, x="region", y="consumption_liters", color="region")
    fig_box.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_box, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-container">', unsafe_allow_html=True)
    st.markdown("### Day of Week Trends")
    df['day_name'] = df['date'].dt.day_name()
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    fig_day = px.box(df, x="day_name", y="consumption_liters", category_orders={"day_name": order})
    fig_day.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
    st.plotly_chart(fig_day, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
