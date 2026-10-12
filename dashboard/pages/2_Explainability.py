import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Explainability", layout="wide")

st.title("🧠 SHAP Explainability Dashboard")
st.markdown("Understand **why** ADRIFT made its decisions for Drift Classification, Severity, and Recovery Actions.")

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
csv_path = os.path.join(base_dir, "data", "explanations.csv")

if not os.path.exists(csv_path):
    st.warning("No explanations found. Run the ADRIFT pipeline to generate drift events.")
    st.stop()

try:
    df = pd.read_csv(csv_path)
except pd.errors.EmptyDataError:
    st.info("No explainability logs generated yet.")
    st.stop()

if df.empty:
    st.info("No explainability logs generated yet.")
    st.stop()
    
st.subheader("📋 Decision Logs")
st.dataframe(df.tail(20), use_container_width=True)

st.divider()

st.subheader("📊 Latest SHAP Visualizations")
st.markdown("These waterfall plots show exactly how the Random Forest Surrogate Model calculated the final Severity Score for recent events based on the underlying heuristic engine's logic.")

fig_dir = os.path.join(base_dir, "evaluation", "results", "figures")
if os.path.exists(fig_dir):
    shap_plots = [f for f in os.listdir(fig_dir) if f.startswith("shap_waterfall_")]
    if shap_plots:
        cols = st.columns(min(len(shap_plots), 3))
        for i, plot in enumerate(shap_plots[:3]):
            with cols[i]:
                st.image(os.path.join(fig_dir, plot), use_container_width=True)
                st.caption(f"Event: {plot.replace('shap_waterfall_', '').replace('.png', '')}")
    else:
        st.info("No SHAP waterfall plots found.")
