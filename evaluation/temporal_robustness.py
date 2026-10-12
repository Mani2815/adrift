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

# Monkeypatch sleep for instant fast-forward execution
time.sleep = lambda x: None

def run_temporal_segment(dataset_name, start_row, end_row, segment_name):
    print(f"\n--- Running Segment: {segment_name} ({start_row:,} to {end_row:,}) ---")
    settings.DATA_SOURCE = dataset_name
    settings.PPO_ENABLED = True # Evaluate PPO Agent
    
    # We need to seek the adapter to the start_row
    class SegmentAdapter(DatasetAdapter):
        def __init__(self, dataset_name, skip_rows):
            super().__init__(dataset_name)
            # Skip rows
            for _ in range(skip_rows):
                try:
                    next(self.csv_reader)
                except StopIteration:
                    break
                    
        def generate_event(self):
            row = next(self.csv_reader)
            self.step += 1
            try:
                primary_val = float(row.get(self.value_col, 0.0))
            except ValueError:
                primary_val = 0.0
            return {"value": primary_val}
            
    try:
        adapter = SegmentAdapter(dataset_name, start_row)
    except FileNotFoundError:
        return None
        
    recovery_engine = RecoveryEngine()
    drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
    
    start_time = time.time()
    rows_processed = 0
    max_rows = end_row - start_row
    
    try:
        while rows_processed < max_rows:
            event = adapter.generate_event()
            drift_detector.process_value(event["value"])
            rows_processed += 1
            if rows_processed % 50000 == 0:
                print(f"  Processed {rows_processed:,}/{max_rows:,} rows...")
    except StopIteration:
        pass
        
    end_time = time.time()
    duration = max(end_time - start_time, 0.001)
    throughput = rows_processed / duration
    
    metrics = recovery_engine.get_metrics()
    mttr = metrics.get("mttr", 0.0)
    success_rate = metrics.get("success_rate", 0.0) * 100
    uptime = metrics.get("uptime", 0.0) * 100
    
    return {
        "Segment": segment_name,
        "MTTR": mttr,
        "Success Rate": success_rate,
        "Uptime": uptime,
        "Throughput": throughput
    }

def main():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    
    total_rows = 2830743
    segments = [
        {"name": "0-10% (Early)", "start": 0, "end": int(total_rows * 0.1)},
        {"name": "40-50% (Mid)", "start": int(total_rows * 0.4), "end": int(total_rows * 0.5)},
        {"name": "90-100% (Late)", "start": int(total_rows * 0.9), "end": total_rows}
    ]
    
    results = []
    for seg in segments:
        res = run_temporal_segment("cicids2017", seg["start"], seg["end"], seg["name"])
        if res:
            results.append(res)
            
    csv_path = os.path.join(out_dir, "temporal_robustness.csv")
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    
    # Generate Chart
    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Segment", y="Success Rate", marker="o", label="Simulated Recovery Success Rate (%)")
    sns.lineplot(data=df, x="Segment", y="Uptime", marker="s", label="Uptime (%)")
    plt.title("Temporal Consistency Across Stream Segments")
    plt.xlabel("Predeclared temporal consistency segments")
    plt.ylim(0, 110)
    plt.grid(True)
    
    fig_dir = os.path.join(out_dir, "ieee_figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "temporal_robustness.png"), dpi=300)
    plt.close()
    
    # Generate Table
    tex_path = os.path.join(out_dir, "ieee_tables_temporal.tex")
    tex_content = """\\begin{table}[h]
\\caption{TEMPORAL CONSISTENCY OF PPO AGENT ACROSS CICIDS-2017 SEGMENTS}
\\label{tab:temporal_robustness}
\\centering
\\resizebox{\\columnwidth}{!}{%
\\begin{tabular}{@{}lccc@{}}
\\toprule
\\textbf{Segment} & \\textbf{Simulated Recovery Success Rate (\\%)} & \\textbf{Uptime (\\%)} & \\textbf{Throughput} \\\\ \\midrule
"""
    for _, row in df.iterrows():
        seg_escaped = row['Segment'].replace('%', '\\%')
        tex_content += f"{seg_escaped} & {row['Success Rate']:.1f}\\% & {row['Uptime']:.1f}\\% & {row['Throughput']:.0f} \\\\\n"
    
    tex_content += """\\bottomrule
\\end{tabular}%
}
\\end{table}
"""
    with open(tex_path, 'w') as f:
        f.write(tex_content)
        
    print(f"\n✅ Temporal robustness evaluation complete. Data saved to {csv_path}")

if __name__ == "__main__":
    main()
