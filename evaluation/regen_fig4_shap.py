#!/usr/bin/env python3
"""
Regenerate Figure 4 (global SHAP importance bar chart) from stored CSV.
Title corrected to match manuscript: 'synthetically generated states'.
All values preserved exactly from global_shap_importance.csv.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE, "evaluation", "results", "global_shap_importance.csv")
FIG_PATH = os.path.join(BASE, "evaluation", "results", "ieee_figures", "global_shap.png")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

df = pd.read_csv(CSV_PATH)
print("Loaded SHAP values:")
print(df.to_string(index=False))

plt.figure(figsize=(10, 6))
sns.barplot(
    x="Mean Absolute SHAP Value",
    y="Feature",
    data=df,
    palette="rocket"
)
plt.title("Global mean absolute SHAP attribution over 200 synthetically generated states")
plt.xlabel("Mean |SHAP Value| (Impact on Action Index)")
plt.ylabel("")
plt.tight_layout()
plt.savefig(FIG_PATH, dpi=300)
plt.close()
print(f"Saved updated Figure 4 to: {FIG_PATH}")
