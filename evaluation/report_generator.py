import pandas as pd
import os

class ReportGenerator:
    """
    Consumes the evaluation results CSV, aggregates the findings, and produces
    both a summary CSV and a formatted textual conclusion report for research publication.
    """
    def __init__(self):
        self.results_dir = os.path.join(os.path.dirname(__file__), "results")
        self.csv_path = os.path.join(self.results_dir, "results.csv")
        self.report_csv = os.path.join(self.results_dir, "comparison_report.csv")
        self.ieee_report = os.path.join(self.results_dir, "ieee_conclusion_report.txt")

    def generate_report(self):
        if not os.path.exists(self.csv_path):
            print("No results.csv found. Run experiment_runner.py first.")
            return

        df = pd.read_csv(self.csv_path)
        if df.empty:
            print("results.csv is empty.")
            return
            
        # Aggregate metrics across all 70 trials for each system
        summary = df.groupby('system_type').agg({
            'mttr': 'mean',
            'success_rate': 'mean',
            'uptime': 'mean'
        }).reset_index()
        
        # Save CSV
        summary.to_csv(self.report_csv, index=False)
        
        # Extract specific values
        try:
            baseline = summary[summary['system_type'] == 'Baseline Rule Engine'].iloc[0]
            ppo = summary[summary['system_type'] == 'PPO Agent'].iloc[0]
        except IndexError:
            print("Incomplete data for both systems. Skipping IEEE text report generation.")
            return
            
        # Success Criteria Evaluation
        mttr_improved = ppo['mttr'] < baseline['mttr']
        success_improved = ppo['success_rate'] > baseline['success_rate']
        uptime_improved = ppo['uptime'] > baseline['uptime']
        
        hypothesis_proven = mttr_improved and success_improved and uptime_improved
        
        report_text = f"""
================================================================================
ADRIFT PROJECT - PHASE 6 EVALUATION REPORT
================================================================================
Hypothesis:
A PPO-based adaptive recovery agent will outperform a static rule-based 
baseline engine in real-time streaming pipeline fault mitigation.

--------------------------------------------------------------------------------
Summary Table:
--------------------------------------------------------------------------------
| Metric                | Baseline Rule Engine | PPO Adaptive Agent | Improvement |
|-----------------------|----------------------|--------------------|-------------|
| MTTR (seconds)        | {baseline['mttr']:<20.2f} | {ppo['mttr']:<18.2f} | {"Yes" if mttr_improved else "No":<11} |
| Recovery Success (%)  | {baseline['success_rate']:<20.2f} | {ppo['success_rate']:<18.2f} | {"Yes" if success_improved else "No":<11} |
| Pipeline Uptime (%)   | {baseline['uptime']:<20.2f} | {ppo['uptime']:<18.2f} | {"Yes" if uptime_improved else "No":<11} |

--------------------------------------------------------------------------------
Conclusion:
--------------------------------------------------------------------------------
"""
        if hypothesis_proven:
            report_text += "The PPO-based recovery agent successfully outperformed the static baseline across all defined metrics.\n"
            report_text += "By dynamically mapping state conditions to recovery actions based on learned rewards,\n"
            report_text += "the RL agent achieved a higher success rate and lower MTTR, validating the core hypothesis of the ADRIFT framework.\n"
        else:
            report_text += "The PPO-based recovery agent did NOT consistently outperform the static baseline across all metrics.\n"
            report_text += "Further reward function tuning, extended offline training duration, or state-space feature refinement is required.\n"

        with open(self.ieee_report, 'w') as f:
            f.write(report_text)
            
        print(f"✅ Comparison Report generated at evaluation/results/comparison_report.csv")
        print(f"✅ IEEE Conclusion generated at evaluation/results/ieee_conclusion_report.txt")

if __name__ == "__main__":
    ReportGenerator().generate_report()
