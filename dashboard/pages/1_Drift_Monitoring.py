import os
import sys
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# Ensure we can import from config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

st.set_page_config(page_title="Drift Monitoring", page_icon="📈", layout="wide")

# Auto-refresh every 3 seconds
st_autorefresh(interval=3000, limit=10000, key="drift_refresh")

st.title("📈 Concept Drift Monitoring")
st.markdown("Real-time monitoring of statistical distribution changes using **ADWIN**, **DDM**, and **KSWIN**.")

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "drift_events.csv")

def load_drift_data():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # Convert string boolean representations if necessary
                df['drift_status'] = df['drift_status'].astype(str).str.lower() == 'true'
                df['warning_status'] = df['warning_status'].astype(str).str.lower() == 'true'
            return df
        except Exception as e:
            st.error(f"Error loading drift data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

df = load_drift_data()

# --- Status KPIs ---
st.subheader("System Status")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("ADWIN Detector", "🟢 Active" if settings.ADWIN_ENABLED else "🔴 Disabled")
with col2:
    st.metric("DDM Detector", "🟢 Active" if settings.DDM_ENABLED else "🔴 Disabled")
with col3:
    st.metric("KSWIN Detector", "🟢 Active" if settings.KSWIN_ENABLED else "🔴 Disabled")
with col4:
    total_drifts = len(df[df['drift_status'] == True]) if not df.empty else 0
    st.metric("Total Drifts Detected", total_drifts)
    
st.divider()

# --- Visualizations ---
if not df.empty:
    drift_df = df[df['drift_status'] == True]
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Detector Activity")
        if not drift_df.empty:
            detector_counts = drift_df['detector'].value_counts().reset_index()
            detector_counts.columns = ['detector', 'count']
            fig_bar = px.bar(detector_counts, x='detector', y='count', color='detector', 
                             title="Drifts Caught per Algorithm")
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No full drift events triggered yet. Monitoring in progress...")

    with col_chart2:
        st.subheader("Drift Events Timeline")
        if not drift_df.empty:
            fig_scatter = px.scatter(drift_df, x='timestamp', y='detector', color='detector', 
                                     title="When Drifts Occurred")
            fig_scatter.update_traces(marker=dict(size=12, symbol="x"))
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Awaiting drift data...")
            
    st.subheader("Historical Drift Events Log")
    st.dataframe(df.sort_values(by="timestamp", ascending=False), use_container_width=True, hide_index=True)

else:
    st.info("Drift detection storage is empty. Start the `drift_simulator.py` and `consumer.py` to generate real-time events.")
