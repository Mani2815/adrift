import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "summary_stats_corrected.csv")

def main():
    if not os.path.exists(SUMMARY_PATH):
        print("Summary stats file not found.")
        return
        
    summary = pd.read_csv(SUMMARY_PATH)
    
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})
    
    # Separate sweeps
    sudden = summary[summary['sweep_parameter'] == 'Sudden Sweep']
    gradual = summary[summary['sweep_parameter'] == 'Gradual Sweep']
    default = summary[summary['sweep_parameter'] == 'Default Sweep']
    
    def plot_sweep(data, title, xlabel, filename, baseline_val):
        plt.figure(figsize=(8, 5))
        sns.lineplot(
            x="threshold_value", 
            y="authoritative_trial_success", 
            data=data, 
            marker='o', 
            linewidth=2, 
            markersize=8
        )
        
        # Mark baseline
        baseline_row = data[data['threshold_value'] == baseline_val]
        if not baseline_row.empty:
            plt.plot(
                baseline_val, 
                baseline_row['authoritative_trial_success'].values[0], 
                'ro', 
                markersize=10, 
                label=f"Baseline ({baseline_val})"
            )
            
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel("Recovery Success Rate (%)")
        plt.ylim(0, 100)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, filename))
        plt.close()

    plot_sweep(sudden, "Sudden Threshold Sensitivity (Corrected)", "Sudden Drift Threshold", "fig_sudden_sensitivity_corrected.png", 70)
    plot_sweep(gradual, "Gradual Threshold Sensitivity (Corrected)", "Gradual Drift Threshold", "fig_gradual_sensitivity_corrected.png", 50)
    plot_sweep(default, "Default Threshold Sensitivity (Corrected)", "Default Fallback Threshold", "fig_default_sensitivity_corrected.png", 60)
    
    print("Corrected plots generated successfully.")
    
    # Compute baseline-relative deltas analytically
    baseline_val_total = 97.17
    
    summary['delta_pp'] = summary['authoritative_trial_success'] - baseline_val_total
    
    print("\n--- Baseline-Relative Analysis ---")
    print(f"Max Absolute Deviation: {summary['delta_pp'].abs().max():.2f} pp")
    print(f"Min Deviation: {summary['delta_pp'].min():.2f} pp")
    print(f"Max Deviation: {summary['delta_pp'].max():.2f} pp")
    
    print("\n--- Perturbation-Only Range ---")
    print(f"Min Success: {summary['authoritative_trial_success'].min():.2f}%")
    print(f"Max Success: {summary['authoritative_trial_success'].max():.2f}%")

if __name__ == "__main__":
    main()
