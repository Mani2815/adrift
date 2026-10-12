import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "phase2b2_threshold_sensitivity")
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "raw_results.csv")

def main():
    if not os.path.exists(RAW_CSV_PATH):
        print("Run threshold_sensitivity_experiment.py first.")
        return
        
    df = pd.read_csv(RAW_CSV_PATH)
    
    # Calculate success rate for each config
    # Since success is bool, mean() gives the success rate.
    
    summary = df.groupby(['sweep_parameter', 'threshold_value']).agg(
        n=('success', 'count'),
        success_rate=('success', 'mean'),
        uptime_mean=('uptime', 'mean'),
        throughput_mean=('throughput', 'mean')
    ).reset_index()
    
    summary['success_rate_percent'] = summary['success_rate'] * 100
    
    # Save summary stats
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"), index=False)
    
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    
    # Separate sweeps
    sudden = summary[summary['sweep_parameter'] == 'Sudden Sweep']
    gradual = summary[summary['sweep_parameter'] == 'Gradual Sweep']
    default = summary[summary['sweep_parameter'] == 'Default Sweep']
    
    def plot_sweep(data, title, xlabel, filename, baseline_val):
        plt.figure(figsize=(8, 5))
        sns.lineplot(x="threshold_value", y="success_rate_percent", data=data, marker='o', linewidth=2, markersize=8)
        
        # Mark baseline
        baseline_row = data[data['threshold_value'] == baseline_val]
        if not baseline_row.empty:
            plt.plot(baseline_val, baseline_row['success_rate_percent'].values[0], 'ro', markersize=10, label=f"Baseline ({baseline_val})")
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Recovery Success Rate (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()

    plot_sweep(sudden, "Sudden Threshold Sensitivity", "Sudden Drift Threshold", "fig_sudden_sensitivity.png", 70)
    plot_sweep(gradual, "Gradual Threshold Sensitivity", "Gradual Drift Threshold", "fig_gradual_sensitivity.png", 50)
    plot_sweep(default, "Default Threshold Sensitivity", "Default Fallback Threshold", "fig_default_sensitivity.png", 60)
    
    print("Plots generated successfully.")
    
    # Verify Baseline with the original
    # The actual baseline in Phase 2B-1 / Phase 2A for Rule engine was:
    # If the success rate of the (70,50,60) config matches exactly.
    baseline_rows = df[(df['sweep_parameter'] == 'Sudden Sweep') & (df['threshold_value'] == 70)]
    baseline_success = baseline_rows['success'].mean() * 100
    print(f"Baseline Success Rate (70, 50, 60): {baseline_success:.2f}%")

if __name__ == "__main__":
    main()
