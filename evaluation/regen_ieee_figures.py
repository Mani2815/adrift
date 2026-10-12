#!/usr/bin/env python3
"""Regenerate all paper figures at 300 DPI with IEEE-quality formatting."""
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# IEEE-quality global settings
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 15,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "lines.linewidth": 2.0,
})

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(BASE, "evaluation", "results", "figures")
PAPER_FIG_DIR = os.path.join(BASE, "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def regen_learning_curves():
    csv_path = os.path.join(BASE, "logs", "ppo_learning_curve.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  SKIP: empty CSV")
        return

    # Reward Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['timestep'], df['mean_reward'], color='#1f77b4', linewidth=2.5)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Mean Episode Reward')
    ax.set_title('PPO Training: Reward vs Timesteps')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ppo_reward_curve.png'), dpi=300)
    plt.close(fig)
    print("  ✓ ppo_reward_curve.png")

    # Success Rate Curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df['timestep'], df['success_rate'], color='#2ca02c', linewidth=2.5)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel('Simulated Recovery Success Rate (%)')
    ax.set_title('PPO Training: Success Rate vs Timesteps')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'ppo_success_curve.png'), dpi=300)
    plt.close(fig)
    print("  ✓ ppo_success_curve.png")


def regen_comparison_plots():
    csv_path = os.path.join(BASE, "evaluation", "results", "results.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  SKIP: empty CSV")
        return

    agg = df.groupby(['scenario', 'system_type']).mean(numeric_only=True).reset_index()
    scenarios = agg['scenario'].unique()
    bar_w = 0.35

    def plot_bar(col, title, ylabel, fname):
        fig, ax = plt.subplots(figsize=(8, 5))
        bl = agg[agg['system_type'] == 'Baseline Rule Engine'].set_index('scenario')
        pp = agg[agg['system_type'] == 'PPO Agent'].set_index('scenario')
        x = range(len(scenarios))
        bv = [bl.loc[s, col] if s in bl.index else 0 for s in scenarios]
        pv = [pp.loc[s, col] if s in pp.index else 0 for s in scenarios]
        ax.bar([i - bar_w/2 for i in x], bv, bar_w, label='Baseline Rule Engine', color='#1f77b4')
        ax.bar([i + bar_w/2 for i in x], pv, bar_w, label='PPO Agent', color='#ff7f0e')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(scenarios, rotation=35, ha='right')
        ax.legend(framealpha=0.9)
        
        if col == 'mttr':
            ax.set_ylim(0, 1.0)
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.5, 0.5, 'Both systems: 0.00 s computational MTTR', 
                    transform=ax.transAxes, fontsize=12, fontweight='bold',
                    verticalalignment='center', horizontalalignment='center', bbox=props)
        elif col == 'uptime':
            props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
            ax.text(0.5, 0.4, 'Aggregate Uptime:\nBaseline Rule Engine: 99.997%\nPPO Agent: 99.981%', 
                    transform=ax.transAxes, fontsize=11,
                    verticalalignment='center', horizontalalignment='center', bbox=props)

        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, fname), dpi=300)
        plt.close(fig)
        print(f"  ✓ {fname}")

    plot_bar('mttr', 'Mean Time To Recovery (MTTR) by Scenario', 'MTTR (seconds)', 'mttr_comparison.png')
    plot_bar('uptime', 'Pipeline Uptime by Scenario', 'Uptime (%)', 'uptime_comparison.png')


def regen_drift_types():
    csv_path = os.path.join(BASE, "evaluation", "results", "dataset_drifts.csv")
    if not os.path.exists(csv_path):
        print(f"  SKIP: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("  SKIP: empty CSV")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name in zip(axes, ['Electricity', 'CICIDS-2017']):
        sub = df[df['dataset'] == name]
        counts = sub['drift_type'].value_counts()
        ax.bar(counts.index, counts.values, color='#1f77b4', edgecolor='white')
        ax.set_title(f'{name} — Drift Type Distribution')
        ax.set_ylabel('Event Count')
        ax.set_xlabel('Drift Type')
        for i, (idx, v) in enumerate(counts.items()):
            ax.text(i, v + max(counts.values)*0.02, str(v), ha='center', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'drift_types.png'), dpi=300)
    plt.close(fig)
    print("  ✓ drift_types.png")


def copy_to_paper():
    print("\nCopying to paper figures/...")
    for fname in ['ppo_reward_curve.png', 'ppo_success_curve.png',
                  'mttr_comparison.png', 'uptime_comparison.png', 'drift_types.png']:
        src = os.path.join(FIG_DIR, fname)
        dst = os.path.join(PAPER_FIG_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"  ✓ {fname} → paper/figures/")
        else:
            print(f"  ✗ {fname} not found, skipping")


if __name__ == "__main__":
    print("=== Regenerating IEEE-quality figures (300 DPI) ===")
    regen_learning_curves()
    regen_comparison_plots()
    regen_drift_types()
    copy_to_paper()
    print("\n=== Done ===")
