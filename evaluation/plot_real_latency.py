import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_plots():
    base_dir = os.path.dirname(__file__)
    results_dir = os.path.join(base_dir, "results", "phase2b1_real_latency")
    raw_csv = os.path.join(results_dir, "raw_results.csv")
    
    if not os.path.exists(raw_csv):
        print("Raw CSV not found. Run the experiment first.")
        return
        
    df = pd.read_csv(raw_csv)
    
    # Configure Seaborn style
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    
    # 1. Decision Latency vs Engine
    plt.figure(figsize=(10, 6))
    # Decision latency is very small (microseconds), so we convert to milliseconds for readability
    df['decision_latency_ms'] = df['decision_latency_seconds'] * 1000
    sns.boxplot(x="engine", y="decision_latency_ms", data=df, showfliers=True)
    plt.title("Figure A: Decision Latency vs Engine")
    plt.ylabel("Decision Latency (ms)")
    plt.xlabel("Recovery Engine")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_A_decision_latency.png"))
    plt.close()
    
    # 2. Action Execution Latency vs Engine
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="engine", y="action_execution_latency_seconds", hue="action", data=df, showfliers=True)
    plt.title("Figure B: Action Execution Latency vs Engine (by Action)")
    plt.ylabel("Execution Latency (s)")
    plt.xlabel("Recovery Engine")
    plt.legend(title="Recovery Action", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_B_action_latency.png"))
    plt.close()
    
    # 3. End-to-End Recovery Time vs Engine (by Scenario)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="scenario", y="end_to_end_recovery_seconds", hue="engine", data=df, showfliers=True)
    plt.title("Figure C: End-to-End Recovery Time vs Engine (by Scenario)")
    plt.ylabel("End-to-End Latency (s)")
    plt.xlabel("Fault/Drift Scenario")
    plt.xticks(rotation=45)
    plt.legend(title="Recovery Engine")
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "fig_C_end_to_end_latency.png"))
    plt.close()

    print("Figures A, B, and C successfully generated.")

if __name__ == "__main__":
    generate_plots()
