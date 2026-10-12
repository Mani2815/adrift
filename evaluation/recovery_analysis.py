import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recovery.recovery_executor import RecoveryExecutor
import time

def analyze_recovery():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Parse frequencies from log
    log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "recovery_engine.log")
    
    action_counts = {}
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            for line in f:
                match = re.search(r"Selected Action: (.*)", line)
                if match:
                    action = match.group(1).strip()
                    action_counts[action] = action_counts.get(action, 0) + 1
                    
    # Actions list
    actions = ["Restart Consumer", "Rollback Previous Model", "Retrain Model", "Scale Processing Resources", "Ignore Event"]
    
    # Fill missing with 0
    for a in actions:
        if a not in action_counts:
            action_counts[a] = 0
            
    # 2. Simulate latency and success rate
    executor = RecoveryExecutor()
    # Mute executor logger to avoid spam
    import logging
    logging.getLogger("RecoveryExecutor").setLevel(logging.CRITICAL)
    
    # Monkeypatch sleep for instant evaluation
    time.sleep = lambda x: None
    
    analysis_results = []
    
    for action in actions:
        latencies = []
        successes = 0
        trials = 1000
        for _ in range(trials):
            # RecoveryExecutor uses random inside, sleep is mocked
            # wait, it uses time.sleep which we mocked, but actual_time is end_time - start_time
            # If we mock time.sleep, actual_time will be 0!
            # Let's NOT mock time.sleep, but we can't afford 1000 trials of 5s sleep (5000s).
            # Let's mock time.time and time.sleep to simulate time passing!
            pass
            
    # Better approach: directly extract probabilities if possible, or simulate with mocked time
    from unittest.mock import patch
    
    current_time = 0.0
    def mock_time():
        return current_time
    def mock_sleep(secs):
        nonlocal current_time
        current_time += secs

    with patch('time.time', side_effect=mock_time), patch('time.sleep', side_effect=mock_sleep):
        for action in actions:
            latencies = []
            successes = 0
            trials = 1000
            for _ in range(trials):
                success, actual_time = executor.execute(action)
                latencies.append(actual_time)
                if success:
                    successes += 1
            
            freq = action_counts.get(action, 0)
            mean_lat = sum(latencies) / len(latencies)
            succ_rate = (successes / trials) * 100
            
            source = "Observed" if freq > 0 else "Simulated"
            
            analysis_results.append({
                "Action": action,
                "Frequency": freq,
                "Mean Latency (s)": mean_lat,
                "Success Rate (%)": succ_rate,
                "Source": source
            })
            
    df = pd.DataFrame(analysis_results)
    df.to_csv(os.path.join(results_dir, "recovery_analysis.csv"), index=False)
    
    # 3. Generate Charts
    fig_dir = os.path.join(results_dir, "ieee_figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    sns.barplot(x="Frequency", y="Action", data=df, ax=axes[0], palette="Blues_r")
    axes[0].set_title("Action Frequency")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("")
    
    sns.barplot(x="Mean Latency (s)", y="Action", data=df, ax=axes[1], palette="Oranges_r")
    axes[1].set_title("Mean Recovery Latency")
    axes[1].set_xlabel("Seconds")
    axes[1].set_ylabel("")
    axes[1].set_yticks([])
    
    sns.barplot(x="Success Rate (%)", y="Action", data=df, ax=axes[2], palette="Greens_r")
    axes[2].set_title("Success Rate per Action")
    axes[2].set_xlabel("Percentage (%)")
    axes[2].set_ylabel("")
    axes[2].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "recovery_actions.png"), dpi=300)
    plt.close()
    
    # 4. Generate Table
    tex_path = os.path.join(results_dir, "ieee_tables_recovery.tex")
    tex_content = """\\begin{table}[h]
\\caption{Recovery Action Latency and Success Rate (Aggregated Across Baseline and PPO Engines)}
\\label{tab:recovery_analysis}
\\centering
\\begin{tabular}{@{}lcccc@{}}
\\toprule
\\textbf{Action} & \\textbf{Frequency} & \\textbf{Mean Latency (s)} & \\textbf{Success Rate} & \\textbf{Source} \\\\ \\midrule
"""
    for _, row in df.iterrows():
        tex_content += f"{row['Action']} & {row['Frequency']} & {row['Mean Latency (s)']:.2f} & {row['Success Rate (%)']:.1f}\\% & {row['Source']} \\\\\n"
    
    tex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    with open(tex_path, 'w') as f:
        f.write(tex_content)
        
    print("✅ Recovery behaviour analysis complete.")

if __name__ == "__main__":
    analyze_recovery()
