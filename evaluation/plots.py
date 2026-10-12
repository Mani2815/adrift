import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_learning_curves():
    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "ppo_learning_curve.csv")
    fig_dir = os.path.join(os.path.dirname(__file__), "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        return
        
    # Reward Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestep'], df['mean_reward'], label='Mean Reward', color='blue', linewidth=2)
    plt.title('PPO Learning Curve: Reward vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Mean Episode Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'ppo_reward_curve.png'))
    plt.close()
    
    # Success Rate Curve
    plt.figure(figsize=(10, 6))
    plt.plot(df['timestep'], df['success_rate'], label='Success Rate', color='green', linewidth=2)
    plt.title('PPO Learning Curve: Success Rate vs Timesteps')
    plt.xlabel('Timesteps')
    plt.ylabel('Success Rate (%)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'ppo_success_curve.png'))
    plt.close()

def generate_all_plots():
    csv_path = os.path.join(os.path.dirname(__file__), "results", "results.csv")
    fig_dir = os.path.join(os.path.dirname(__file__), "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    if df.empty:
        return
        
    # Group and aggregate data for easy plotting
    agg_df = df.groupby(['scenario', 'system_type']).mean(numeric_only=True).reset_index()
    
    scenarios = agg_df['scenario'].unique()
    bar_width = 0.35
    
    # Helper to plot clustered bars
    def plot_clustered_bar(metric_col, title, ylabel, filename):
        plt.figure(figsize=(10, 6))
        
        baseline_data = agg_df[agg_df['system_type'] == 'Baseline Rule Engine'].set_index('scenario')
        ppo_data = agg_df[agg_df['system_type'] == 'PPO Agent'].set_index('scenario')
        
        # Ensure alignment
        x = range(len(scenarios))
        b_vals = [baseline_data.loc[s, metric_col] if s in baseline_data.index else 0 for s in scenarios]
        p_vals = [ppo_data.loc[s, metric_col] if s in ppo_data.index else 0 for s in scenarios]
        
        plt.bar([pos - bar_width/2 for pos in x], b_vals, width=bar_width, label='Baseline Rule Engine', color='#1f77b4')
        plt.bar([pos + bar_width/2 for pos in x], p_vals, width=bar_width, label='PPO Agent', color='#ff7f0e')
        
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xticks(x, scenarios, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, filename))
        plt.close()

    # 1. MTTR Comparison
    plot_clustered_bar('mttr', 'Mean Time To Recovery (MTTR) by Scenario', 'MTTR (seconds)', 'mttr_comparison.png')
    
    # 2. Recovery Success Rate
    plot_clustered_bar('success_rate', 'Recovery Success Rate by Scenario', 'Success Rate (%)', 'success_rate_comparison.png')
    
    # 3. Uptime Comparison
    plot_clustered_bar('uptime', 'Pipeline Uptime by Scenario', 'Uptime (%)', 'uptime_comparison.png')
    
    # 4. PPO Performance Curve (Success Rate over time/trials)
    ppo_df = df[df['system_type'] == 'PPO Agent'].copy()
    if not ppo_df.empty:
        ppo_df['trial_num'] = range(1, len(ppo_df) + 1)
        # Smoothing with a rolling average
        ppo_df['smoothed_success'] = ppo_df['success_rate'].rolling(window=5, min_periods=1).mean()
        
        plt.figure(figsize=(10, 6))
        plt.plot(ppo_df['trial_num'], ppo_df['success_rate'], alpha=0.3, label='Raw Success Rate')
        plt.plot(ppo_df['trial_num'], ppo_df['smoothed_success'], color='red', linewidth=2, label='Smoothed Trend')
        plt.title('PPO Agent Evaluation Performance Trend')
        plt.xlabel('Evaluation Trial Number')
        plt.ylabel('Success Rate (%)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'ppo_performance_curve.png'))
        plt.close()
        
    print("✅ Visualizations saved to evaluation/results/figures/")
    generate_learning_curves()

if __name__ == "__main__":
    generate_all_plots()
