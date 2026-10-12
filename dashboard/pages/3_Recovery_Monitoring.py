import os
import sys
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.express as px

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

st.set_page_config(page_title="Recovery Monitoring", page_icon="⚙️", layout="wide")
st_autorefresh(interval=3000, limit=10000, key="recovery_refresh")

st.title("⚙️ Auto-Recovery Monitoring Dashboard")
st.markdown("Monitoring the Baseline Static Rule-Based Recovery Engine (Phase 4).")

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "recovery_history.csv")

def load_recovery_data():
    if os.path.exists(CSV_PATH):
        try:
            df = pd.read_csv(CSV_PATH)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['success'] = df['success'].astype(str).str.lower() == 'true'
            return df
        except Exception as e:
            st.error(f"Error loading recovery data: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

df = load_recovery_data()

if df.empty:
    st.info("No recovery events have occurred yet. Waiting for drift or fault triggers...")
else:
    # Latest is the last row in CSV
    latest = df.iloc[-1] 
    
    total_recoveries = len(df)
    successful = len(df[df['success'] == True])
    success_rate = (successful / total_recoveries) * 100 if total_recoveries > 0 else 100.0
    mttr = df['execution_time'].mean()
    
    st.subheader("Pipeline Recovery KPIs")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Latest Executed Action", latest['recovery_action'])
    with col2:
        st.metric("Mean Time To Recovery (MTTR)", f"{mttr:.2f}s")
    with col3:
        st.metric("Recovery Success Rate", f"{success_rate:.1f}%")
    with col4:
        st.metric("Total Recovery Attempts", total_recoveries)
        
    st.divider()
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Recovery Action Distribution")
        action_counts = df['recovery_action'].value_counts().reset_index()
        action_counts.columns = ['recovery_action', 'count']
        fig_pie = px.pie(action_counts, names='recovery_action', values='count', hole=0.3, 
                         title="Frequency of Orchestrated Actions")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_chart2:
        st.subheader("Recovery Execution Timeline")
        fig_bar = px.bar(df, x='timestamp', y='execution_time', color='success', 
                         title="Execution Time (s) per Recovery Event",
                         color_discrete_map={True: 'green', False: 'red'},
                         labels={'execution_time': 'Execution Time (s)', 'timestamp': 'Time'})
        st.plotly_chart(fig_bar, use_container_width=True)
        
    st.subheader("Comprehensive Recovery Log")
    st.dataframe(df.sort_values(by='timestamp', ascending=False), use_container_width=True, hide_index=True)
