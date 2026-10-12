import pandas as pd
import os

def generate_final_report():
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    df = pd.read_csv(os.path.join(out_dir, "dataset_drifts.csv"))
    throughput_df = pd.read_csv(os.path.join(out_dir, "dataset_throughput.csv"))
    
    report = "# ADRIFT Dataset Evaluation Comparison\n\n"
    
    for dataset in ["Electricity", "CICIDS-2017"]:
        d_df = df[df["dataset"] == dataset]
        t_df = throughput_df[throughput_df["dataset"] == dataset]
        if t_df.empty:
            continue
        t_df = t_df.iloc[0]
        
        rows = t_df["rows_processed"]
        rps = t_df["throughput_rps"]
        total_drifts = len(d_df)
        dpm = (total_drifts / rows) * 1000000 if rows > 0 else 0
        
        sudden = len(d_df[d_df["drift_type"] == "Sudden Drift"])
        gradual = len(d_df[d_df["drift_type"] == "Gradual Drift"])
        incremental = len(d_df[d_df["drift_type"] == "Incremental Drift"])
        recurring = len(d_df[d_df["drift_type"] == "Recurring Drift"])
        
        avg_sev = d_df["severity_score"].mean() if total_drifts > 0 else 0
        max_sev = d_df["severity_score"].max() if total_drifts > 0 else 0
        
        report += f"## {dataset} Dataset\n"
        report += f"### Pipeline Metrics\n"
        report += f"- **Rows Processed**: {rows:,}\n"
        report += f"- **Processing Throughput**: {rps:,.2f} rows/sec\n\n"
        
        report += f"### Drift Detection Statistics\n"
        report += f"- **Total Drift Events**: {total_drifts:,}\n"
        report += f"- **Drifts per Million Records**: {dpm:,.2f}\n\n"
        
        report += f"### Drift Classification\n"
        report += f"- Sudden: {sudden:,}\n"
        report += f"- Gradual: {gradual:,}\n"
        report += f"- Incremental: {incremental:,}\n"
        report += f"- Recurring: {recurring:,}\n\n"
        
        report += f"### Severity Analysis\n"
        report += f"- **Average Severity**: {avg_sev:.2f}\n"
        report += f"- **Maximum Severity**: {max_sev:.2f}\n\n"
        
        report += "---\n\n"
        
    with open(os.path.join(out_dir, "dataset_comparison_report.md"), "w") as f:
        f.write(report)
        
    print(f"Report generated successfully at {os.path.join(out_dir, 'dataset_comparison_report.md')}")
    
if __name__ == "__main__":
    generate_final_report()
