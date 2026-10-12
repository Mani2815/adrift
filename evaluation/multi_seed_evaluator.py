import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as plt_sns

def run_multi_seed_eval():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, "results.csv")
    
    if not os.path.exists(results_path):
        print("Error: results.csv not found.")
        return

    df = pd.read_csv(results_path)
    
    def ci95(x):
        return stats.sem(x) * stats.t.ppf((1 + 0.95) / 2., len(x)-1) if len(x) > 1 else 0

    agg_df = df.groupby(['scenario', 'system_type']).agg(
        mttr_mean=('mttr', 'mean'),
        mttr_std=('mttr', 'std'),
        mttr_ci=('mttr', ci95),
        success_mean=('success_rate', 'mean'),
        success_std=('success_rate', 'std'),
        success_ci=('success_rate', ci95),
        uptime_mean=('uptime', 'mean'),
        uptime_std=('uptime', 'std'),
        uptime_ci=('uptime', ci95)
    ).reset_index()

    agg_df.to_csv(os.path.join(results_dir, "multi_seed_stats.csv"), index=False)

    df['trial'] = df.groupby(['scenario', 'system_type']).cumcount()
    pivot = df.pivot_table(index=['scenario', 'trial'], columns='system_type', values=['mttr', 'success_rate', 'uptime'])
    
    wilcoxon_results = []
    
    for metric in ['mttr', 'success_rate', 'uptime']:
        baseline = pivot[metric]['Baseline Rule Engine'].dropna()
        ppo = pivot[metric]['PPO Agent'].dropna()
        
        try:
            diff = baseline - ppo
            if np.all(diff == 0):
                stat, p = 0.0, 1.0
            else:
                stat, p = stats.wilcoxon(baseline, ppo)
        except ValueError:
            stat, p = float('nan'), float('nan')
            
        wilcoxon_results.append({
            'metric': metric,
            'statistic': stat,
            'p_value': p,
            'significant_0_05': p < 0.05
        })
        
    wilcox_df = pd.DataFrame(wilcoxon_results)
    wilcox_df.to_csv(os.path.join(results_dir, "wilcoxon_test.csv"), index=False)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = ['mttr_mean', 'success_mean', 'uptime_mean']
    titles = ['MTTR (decision latency, s)', 'Success Rate (%)', 'Uptime (%)']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        baseline_data = agg_df[agg_df['system_type'] == 'Baseline Rule Engine'][metric]
        ppo_data = agg_df[agg_df['system_type'] == 'PPO Agent'][metric]
        
        baseline_err = agg_df[agg_df['system_type'] == 'Baseline Rule Engine'][metric.replace('mean', 'ci')]
        ppo_err = agg_df[agg_df['system_type'] == 'PPO Agent'][metric.replace('mean', 'ci')]
        
        x = np.arange(len(baseline_data))
        width = 0.35
        
        ax.bar(x - width/2, baseline_data, width, yerr=baseline_err, label='Baseline Rule Engine', capsize=5, color='#4c72b0')
        ax.bar(x + width/2, ppo_data, width, yerr=ppo_err, label='PPO Agent', capsize=5, color='#dd8452')
        
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(agg_df['scenario'].unique(), rotation=45, ha='right')
        if idx == 0:
            ax.legend()
            for container in ax.containers:
                if hasattr(container, 'patches') or type(container).__name__ == 'BarContainer':
                    try:
                        ax.bar_label(container, labels=['0.00 s'] * len(container), padding=3, fontsize=9)
                    except AttributeError:
                        pass
            
    plt.tight_layout()
    fig_dir = os.path.join(results_dir, "ieee_figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "multi_seed_comparison.png"), dpi=300)
    plt.close()
    
    print("✅ Multi-seed evaluation complete.")

if __name__ == "__main__":
    run_multi_seed_eval()
