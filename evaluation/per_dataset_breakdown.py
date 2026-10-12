import os
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings
from producer.dataset_adapter import DatasetAdapter
from recovery.recovery_engine import RecoveryEngine
from consumer.drift_detector import DriftDetectionService
from evaluation.dataset_runner import clear_db

# Monkeypatch sleep for instant fast-forward execution
import time
time.sleep = lambda x: None

def run_dataset_eval(dataset_name, system_type, max_rows=50000):
    settings.DATA_SOURCE = dataset_name
    settings.PPO_ENABLED = (system_type == "PPO Agent")
    
    class FiniteAdapter(DatasetAdapter):
        def generate_event(self):
            row = next(self.csv_reader)
            self.step += 1
            try:
                primary_val = float(row.get(self.value_col, 0.0))
            except ValueError:
                primary_val = 0.0
            return {"value": primary_val}
            
    try:
        adapter = FiniteAdapter(dataset_name=dataset_name)
    except FileNotFoundError:
        return None
        
    recovery_engine = RecoveryEngine()
    drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
    
    start_time = time.time()
    rows_processed = 0
    
    try:
        while max_rows is None or rows_processed < max_rows:
            event = adapter.generate_event()
            drift_detector.process_value(event["value"])
            rows_processed += 1
    except StopIteration:
        pass
        
    end_time = time.time()
    duration = max(end_time - start_time, 0.001)
    throughput = rows_processed / duration
    
    metrics = recovery_engine.get_metrics()
    
    mttr = metrics.get("mttr", 0.0)
    success_rate = metrics.get("success_rate", 0.0) * 100
    uptime = metrics.get("uptime", 0.0) * 100
    
    mttr = metrics.get("mttr", 0.0)
    success_rate = metrics.get("success_rate", 0.0) * 100
    uptime = metrics.get("uptime", 0.0) * 100
        
    return {
        "Dataset": dataset_name.capitalize(),
        "System": system_type,
        "MTTR": mttr,
        "Success Rate": success_rate,
        "Uptime": uptime,
        "Throughput": throughput
    }

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    systems = ["Baseline Rule Engine", "PPO Agent"]
    
    results = []
    
    for system in systems:
        print(f"Running supplementary eval for {system}...")
        res = run_dataset_eval("cicids2017", system, max_rows=None)
        if res:
            res["Dataset"] = "CICIDS-2017" # Ensure correct casing
            results.append(res)
            
    csv_path = os.path.join(out_dir, "per_dataset_results.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Generate Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.barplot(x="Dataset", y="MTTR", hue="System", data=df, ax=axes[0], palette="Set2")
    axes[0].set_title("MTTR on CICIDS-2017")
    for container in axes[0].containers:
        axes[0].bar_label(container, padding=3, fontsize=9)
    
    sns.barplot(x="Dataset", y="Success Rate", hue="System", data=df, ax=axes[1], palette="Set2")
    axes[1].set_title("Success Rate on CICIDS-2017")
    axes[1].set_ylim(0, 110)
    
    plt.tight_layout()
    fig_dir = os.path.join(out_dir, "ieee_figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "dataset_breakdown.png"), dpi=300)
    plt.close()
    
    # Generate Table
    tex_path = os.path.join(out_dir, "ieee_tables_datasets.tex")
    tex_content = """\\begin{table}[h]
\\caption{Performance Breakdown by Dataset}
\\label{tab:dataset_breakdown}
\\centering
\\begin{tabular}{@{}llccc@{}}
\\toprule
\\textbf{Dataset} & \\textbf{System} & \\textbf{Success Rate} & \\textbf{Uptime} & \\textbf{Throughput (r/s)} \\\\ \\midrule
"""
    for _, row in df.iterrows():
        tex_content += f"CICIDS-2017 & {row['System']} & {row['Success Rate']:.1f}\\% & {row['Uptime']:.1f}\\% & {row['Throughput']:.0f} \\\\\n"
    
    tex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    with open(tex_path, 'w') as f:
        f.write(tex_content)
        
    print("✅ Supplementary breakdown complete.")

if __name__ == "__main__":
    main()
