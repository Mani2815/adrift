import os
import sys
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# Ensure we can import from config and data
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.history_db import SessionLocal, ClassifiedDrift

st.set_page_config(page_title="Drift Intelligence", page_icon="🧠", layout="wide")

# Auto-refresh every 3 seconds
st_autorefresh(interval=3000, limit=10000, key="intel_refresh")

st.title("🧠 Drift Intelligence Dashboard")
st.markdown("Advanced classification and severity assessment of concept drift.")

def load_intelligence_data():
    db = SessionLocal()
    try:
        # Load from the new Phase 3 repository
        records = db.query(ClassifiedDrift).order_by(ClassifiedDrift.id.desc()).all()
        if not records:
            return pd.DataFrame()
        
        data = [{
            "id": r.id,
            "timestamp": r.timestamp,
            "drift_type": r.drift_type,
            "confidence": r.confidence,
            "severity_score": r.severity_score,
            "severity_level": r.severity_level,
            "detectors_triggered": r.detectors_triggered,
            "magnitude_shift": r.magnitude_shift
        } for r in records]
        
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        st.error(f"Error loading intelligence data: {e}")
        return pd.DataFrame()
    finally:
        db.close()

df = load_intelligence_data()

if df.empty:
    st.info("No intelligence records found. Awaiting drift classification events...")
else:
    latest = df.iloc[0]
    
    # --- KPI Row ---
    st.subheader("Latest Drift Event Analysis")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Drift Type", latest['drift_type'])
    with col2:
        st.metric("Confidence Score", f"{latest['confidence']*100:.1f}%")
    with col3:
        st.metric("Severity Level", latest['severity_level'])
    with col4:
        st.metric("Severity Score (0-100)", f"{latest['severity_score']:.1f}")
        
    st.divider()
    
    # --- Charts Row ---
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Severity Over Time")
        # Line chart showing how severe the drifts are over time
        fig_sev = px.line(df, x='timestamp', y='severity_score', color='drift_type', 
                          markers=True, title="Severity Trend of Concept Drifts")
        
        # Add severity zones in the background
        fig_sev.add_hrect(y0=76, y1=100, fillcolor="red", opacity=0.1, line_width=0, annotation_text="Critical")
        fig_sev.add_hrect(y0=51, y1=75, fillcolor="orange", opacity=0.1, line_width=0, annotation_text="High")
        fig_sev.add_hrect(y0=26, y1=50, fillcolor="yellow", opacity=0.1, line_width=0, annotation_text="Moderate")
        fig_sev.add_hrect(y0=0, y1=25, fillcolor="green", opacity=0.1, line_width=0, annotation_text="Low")
        
        fig_sev.update_layout(yaxis_range=[0, 105])
        st.plotly_chart(fig_sev, use_container_width=True)
        
    with col_chart2:
        st.subheader("Drift Type Frequency")
        # Pie chart of classified types
        type_counts = df['drift_type'].value_counts().reset_index()
        type_counts.columns = ['drift_type', 'count']
        fig_pie = px.pie(type_counts, names='drift_type', values='count', 
                         title="Distribution of Classified Drift Types", hole=0.4,
                         color='drift_type',
                         color_discrete_map={
                             'Sudden Drift': 'red', 
                             'Gradual Drift': 'orange', 
                             'Incremental Drift': 'yellow',
                             'Recurring Drift': 'blue'
                         })
        st.plotly_chart(fig_pie, use_container_width=True)
        
    st.subheader("Historical Intelligence Log")
    # Show the raw dataframe
    st.dataframe(df, use_container_width=True, hide_index=True)
