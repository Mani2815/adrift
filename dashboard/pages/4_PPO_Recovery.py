import os
import sys
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config.settings import settings

st.set_page_config(page_title="PPO Recovery Agent", page_icon="🤖", layout="wide")
st_autorefresh(interval=3000, limit=10000, key="ppo_refresh")

st.title("🤖 PPO Adaptive Recovery Dashboard")
st.markdown("Monitoring the Phase 5 Reinforcement Learning policy decisions in real-time.")

if not settings.PPO_ENABLED:
    st.warning("⚠️ PPO Agent is currently DISABLED in `.env`. The system is falling back to the baseline Rule Engine.")

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "recovery_history.csv")

def load_data():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['success'] = df['success'].astype(str).str.lower() == 'true'
            return df
        except Exception as e:
            return pd.DataFrame()
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.info("Awaiting PPO agent actions...")
else:
    latest = df.iloc[-1]
    
    st.subheader("Current Policy Output")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest PPO Action", latest['recovery_action'])
    with col2:
        st.metric("Trigger Condition", latest['trigger_type'])
    with col3:
        st.metric("Severity Addressed", latest['severity'])
    with col4:
        st.metric("Action Success", "✅ Yes" if latest['success'] else "❌ No")
        
    st.divider()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("RL Policy: Action Distribution")
        action_counts = df['recovery_action'].value_counts().reset_index()
        action_counts.columns = ['recovery_action', 'count']
        fig_pie = px.pie(action_counts, names='recovery_action', values='count', hole=0.4,
                         title="Frequency of Chosen Actions")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("PPO Agent: Success Rate Trend")
        df['rolling_success'] = df['success'].rolling(window=10, min_periods=1).mean() * 100
        fig_line = px.line(df, x='timestamp', y='rolling_success', 
                           title="Rolling 10-Event Success Rate (%)", markers=True)
        fig_line.update_layout(yaxis_range=[0, 105])
        st.plotly_chart(fig_line, use_container_width=True)
        
    st.subheader("PPO Action Log")
    st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True, hide_index=True)
