import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Model Evaluation", layout="wide", page_icon="🤖")

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

st.title("🤖 Model Comparison & Evaluation")

st.markdown('<div class="glass-container">', unsafe_allow_html=True)
st.markdown("### Performance Metrics Across 7 Algorithms (Test Set)")
st.markdown("We trained and compared traditional ML, statistical time-series, and deep learning architectures.")

# Hardcoded from our notebook results for display in dashboard
results = {
    "Model": ["Prophet", "SARIMA", "LSTM", "Random Forest", "LightGBM", "XGBoost", "GRU"],
    "RMSE": [2047.43, 2363.29, 4053.67, 4128.54, 4423.02, 4580.85, 4685.76],
    "MAE": [1636.42, 1793.68, 3441.26, 3455.92, 3645.82, 3728.25, 3847.73],
    "MAPE (%)": [14.19, 16.22, 34.23, 34.99, 35.62, 37.73, 30.97]
}
df_results = pd.DataFrame(results).sort_values("RMSE")

st.dataframe(df_results.style.format({"RMSE": "{:,.2f}", "MAE": "{:,.2f}", "MAPE (%)": "{:.2f}%"}))

fig = go.Figure()
fig.add_trace(go.Bar(
    y=df_results['Model'],
    x=df_results['RMSE'],
    name='RMSE',
    orientation='h',
    marker=dict(color='#00d2ff')
))
fig.add_trace(go.Bar(
    y=df_results['Model'],
    x=df_results['MAE'],
    name='MAE',
    orientation='h',
    marker=dict(color='#ff9f43')
))

fig.update_layout(
    barmode='group',
    title="RMSE & MAE Comparison (Lower is Better)",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis={'categoryorder':'total descending'}
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
