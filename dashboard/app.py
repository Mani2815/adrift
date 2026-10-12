import sys
import os
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

# Ensure we can import from data
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.database import SessionLocal
from data.models import Event, PipelineMetrics
from sqlalchemy import desc

# Configure page
st.set_page_config(page_title="ADRIFT Pipeline Dashboard", page_icon="🌊", layout="wide")

# Auto-refresh every 2 seconds
count = st_autorefresh(interval=2000, limit=10000, key="data_refresh")

st.title("🌊 ADRIFT Dashboard")
st.markdown("**Adaptive Drift-Aware Recovery Framework for Intelligent Streaming Pipelines (Phase 1)**")

def get_metrics():
    db = SessionLocal()
    try:
        metrics = db.query(PipelineMetrics).first()
        return metrics
    finally:
        db.close()

def get_recent_events(limit=100):
    db = SessionLocal()
    try:
        events = db.query(Event).order_by(desc(Event.id)).limit(limit).all()
        return [{
            "id": e.id,
            "timestamp": e.timestamp,
            "event_id": e.event_id,
            "value": e.value,
            "source": e.source,
            "status": e.status,
            "processed_at": e.processed_at
        } for e in events]
    finally:
        db.close()

metrics = get_metrics()

# Top Row: KPIs
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Total Events Processed", value=metrics.total_processed if metrics else 0)

with col2:
    st.metric(label="Processing Rate (evt/min)", value=f"{metrics.processing_rate:.2f}" if metrics else "0.00")

with col3:
    kafka_status = "🟢 Connected" if metrics and metrics.kafka_connected else "🔴 Disconnected"
    st.metric(label="Kafka Status", value=kafka_status)

with col4:
    pipeline_status = "🟢 Active" if metrics and metrics.processing_rate > 0 else "🟡 Idle/Stopped"
    st.metric(label="Consumer Status", value=pipeline_status)

st.divider()

events_data = get_recent_events(limit=200)

if events_data:
    df = pd.DataFrame(events_data)
    # Convert string timestamp to datetime for better plotting
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Event Values Over Time")
        fig_line = px.line(df, x='timestamp', y='value', color='source', 
                           title="Sensor/Node Values (Last 200 Events)")
        st.plotly_chart(fig_line, use_container_width=True)
        
    with col_chart2:
        st.subheader("Event Distribution by Status")
        status_counts = df['status'].value_counts().reset_index()
        status_counts.columns = ['status', 'count']
        fig_pie = px.pie(status_counts, names='status', values='count', 
                         title="Status Breakdown", hole=0.4,
                         color='status',
                         color_discrete_map={'OK': 'green', 'WARN': 'orange', 'ERROR': 'red'})
        st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("Recent Events Log")
    # Show top 20 latest events in the table
    st.dataframe(df[['id', 'timestamp', 'event_id', 'source', 'value', 'status']].head(20), 
                 use_container_width=True, hide_index=True)
else:
    st.info("No events processed yet. Ensure the Producer and Consumer are running.")
