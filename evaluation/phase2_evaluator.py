"""
phase2_evaluator.py
===================
Camera-Ready Phase 2: Evaluation Matrix & Robustness Analysis.

Evaluates independently trained PPO models (seeds 42, 43, 44, 45, 46) against
the Baseline Rule Engine across a fixed matrix of 7 scenarios x 10 evaluation seeds
(70 matched conditions, 350 PPO trials + 70 baseline trials = 420 total trials).

Verifies:
- Identical SHA-256 stream fingerprints across all models for each (scenario, eval_seed) condition.
- Clear separation of training_seed (policy generation) from eval_seed (stream condition).
- Per-seed summaries, per-scenario breakdowns, and hierarchical statistical analysis.
"""

import os
import sys
import time
import csv
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from config.settings import settings
from producer.fault_simulator_corrected import CorrectedFaultSimulator
from evaluation.protocol_b_runner import (
    generate_event_stream,
    run_one_trial,
    RAW_COLUMNS,
)

PHASE2_RESULTS_DIR = os.path.join(_BASE_DIR, "evaluation", "results", "camera_ready_phase2")


def evaluate_phase2_matrix(
    training_seeds=None,
    num_eval_seeds: int = 10,
    trial_length: int = 250,
    fast_simulation: bool = True,
    output_dir: str = PHASE2_RESULTS_DIR,
) -> dict:
    if training_seeds is None:
        training_seeds = [42, 43, 44, 45, 46]

    os.makedirs(output_dir, exist_ok=True)
    scenarios = CorrectedFaultSimulator.ALL_SCENARIOS

    # Check that model checkpoints exist
    model_paths = {}
    for ts in training_seeds:
        ckpt = os.path.join(output_dir, f"training_seed_{ts}", "ppo_agent.zip")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Missing checkpoint for training seed {ts} at {ckpt}")
        model_paths[ts] = ckpt

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_csv = os.path.join(output_dir, f"raw_phase2_trials_{run_id}.csv")
    symlink_raw = os.path.join(output_dir, "raw_phase2_trials.csv")

    with open(raw_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()

    all_rows = []
    print("=" * 80)
    print("ADRIFT Phase 2: Independent PPO Training-Seed Evaluation Matrix")
    print(f"  Training Seeds : {training_seeds} ({len(training_seeds)} independent policies)")
    print(f"  Scenarios      : {len(scenarios)}")
    print(f"  Eval Seeds/scen: {num_eval_seeds}")
    print(f"  Total conditions: {len(scenarios) * num_eval_seeds}")
    print(f"  Total trials   : {(len(training_seeds) + 1) * len(scenarios) * num_eval_seeds}")
    print(f"  Raw output     : {raw_csv}")
    print("=" * 80)

    # 1. Loop through conditions (scenario x eval_seed)
    condition_count = 0
    total_conditions = len(scenarios) * num_eval_seeds

    for scenario in scenarios:
        for eval_seed in range(num_eval_seeds):
            condition_count += 1
            # Pre-generate SHARED stream for this condition
            events, fingerprint, total_injections = generate_event_stream(
                scenario, trial_length, eval_seed
            )

            # A. Evaluate Baseline Rule Engine
            b_row = run_one_trial(
                scenario=scenario,
                engine_name="Baseline Rule Engine",
                eval_seed=eval_seed,
                pre_generated_events=events,
                fast_simulation=fast_simulation,
                training_seed="N/A",
                stream_fingerprint=fingerprint,
                model_path=None,
            )
            all_rows.append(b_row)
            with open(raw_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
                writer.writerow(b_row)

            # B. Evaluate each trained PPO policy on identical stream
            for ts in training_seeds:
                p_row = run_one_trial(
                    scenario=scenario,
                    engine_name="PPO Agent",
                    eval_seed=eval_seed,
                    pre_generated_events=events,
                    fast_simulation=fast_simulation,
                    training_seed=ts,
                    stream_fingerprint=fingerprint,
                    model_path=model_paths[ts],
                )
                all_rows.append(p_row)
                with open(raw_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
                    writer.writerow(p_row)

            if condition_count % 10 == 0 or condition_count == total_conditions:
                print(
                    f"  [Progress {condition_count}/{total_conditions} conditions evaluated] "
                    f"Last condition: {scenario} eval_seed={eval_seed}"
                )

    # Save a stable copy of raw_trials.csv
    import shutil
    shutil.copyfile(raw_csv, symlink_raw)

    print("\n--- Verifying Stream Pairing Integrity ---")
    df = pd.DataFrame(all_rows)
    # Check that for each (scenario, eval_seed), all rows have the exact same fingerprint
    grouped_fp = df.groupby(["scenario", "eval_seed"])["stream_fingerprint"].nunique()
    if (grouped_fp == 1).all():
        print(f"✅ All {len(grouped_fp)} evaluation conditions confirmed identical SHA-256 fingerprints across Baseline and all {len(training_seeds)} PPO models.")
    else:
        print("❌ Warning: Stream fingerprint mismatch detected!")

    # 2. Analyze & Generate Summaries
    analysis_results = analyze_phase2_results(df, training_seeds, output_dir, run_id)
    return analysis_results


def analyze_phase2_results(df: pd.DataFrame, training_seeds: list, output_dir: str, run_id: str) -> dict:
    """Computes all summary tables, per-seed metrics, scenario breakdowns, and statistical tests."""
    valid_df = df[df["valid"]].copy()

    # -------------------------------------------------------------------------
    # A. Summary by Training Seed (and Baseline)
    # -------------------------------------------------------------------------
    # For baseline, training_seed is "N/A"
    seed_summaries = []

    # Baseline summary
    base_df = valid_df[valid_df["engine"] == "Baseline Rule Engine"]
    seed_summaries.append({
        "entity": "Baseline Rule Engine",
        "training_seed": "N/A",
        "n_trials": len(base_df),
        "success_rate_mean": round(base_df["success_rate_pct"].mean(), 4),
        "success_rate_std": round(base_df["success_rate_pct"].std(), 4),
        "uptime_mean": round(base_df["simulated_uptime_pct"].mean(), 6),
        "uptime_std": round(base_df["simulated_uptime_pct"].std(), 6),
        "decision_lat_mean_us": round(base_df["mean_decision_latency_s"].mean() * 1e6, 2),
        "decision_lat_std_us": round(base_df["mean_decision_latency_s"].std() * 1e6, 2),
    })

    # PPO summaries per training seed
    for ts in training_seeds:
        ts_df = valid_df[(valid_df["engine"] == "PPO Agent") & (valid_df["training_seed"] == ts)]
        seed_summaries.append({
            "entity": f"PPO Agent (seed {ts})",
            "training_seed": ts,
            "n_trials": len(ts_df),
            "success_rate_mean": round(ts_df["success_rate_pct"].mean(), 4),
            "success_rate_std": round(ts_df["success_rate_pct"].std(), 4),
            "uptime_mean": round(ts_df["simulated_uptime_pct"].mean(), 6),
            "uptime_std": round(ts_df["simulated_uptime_pct"].std(), 6),
            "decision_lat_mean_us": round(ts_df["mean_decision_latency_s"].mean() * 1e6, 2),
            "decision_lat_std_us": round(ts_df["mean_decision_latency_s"].std() * 1e6, 2),
        })

    summary_seed_df = pd.DataFrame(seed_summaries)
    seed_summary_path = os.path.join(output_dir, "summary_by_training_seed.csv")
    summary_seed_df.to_csv(seed_summary_path, index=False)
    print("\n--- Summary by Training Seed ---")
    print(summary_seed_df.to_string(index=False))

    # Inter-policy variation across the PPO seeds:
    ppo_seed_means = [s["success_rate_mean"] for s in seed_summaries if s["training_seed"] != "N/A"]
    ppo_seed_uptimes = [s["uptime_mean"] for s in seed_summaries if s["training_seed"] != "N/A"]
    ppo_seed_lats = [s["decision_lat_mean_us"] for s in seed_summaries if s["training_seed"] != "N/A"]

    inter_seed_stats = {
        "n_policies": len(training_seeds),
        "success_rate_macro_mean": float(np.mean(ppo_seed_means)),
        "success_rate_inter_seed_std": float(np.std(ppo_seed_means, ddof=1)),
        "success_rate_range": [float(np.min(ppo_seed_means)), float(np.max(ppo_seed_means))],
        "uptime_macro_mean": float(np.mean(ppo_seed_uptimes)),
        "uptime_inter_seed_std": float(np.std(ppo_seed_uptimes, ddof=1)),
        "decision_lat_macro_mean_us": float(np.mean(ppo_seed_lats)),
        "decision_lat_inter_seed_std_us": float(np.std(ppo_seed_lats, ddof=1)),
    }

    # -------------------------------------------------------------------------
    # B. Scenario-Level Breakdown
    # -------------------------------------------------------------------------
    scen_records = []
    scenarios = CorrectedFaultSimulator.ALL_SCENARIOS

    for scenario in scenarios:
        b_scen = base_df[base_df["scenario"] == scenario]
        b_mean = b_scen["success_rate_pct"].mean()
        b_std  = b_scen["success_rate_pct"].std()

        rec = {
            "scenario": scenario,
            "baseline_mean": round(b_mean, 2),
            "baseline_std": round(b_std, 2),
        }

        ppo_scen_means = []
        for ts in training_seeds:
            p_scen = valid_df[
                (valid_df["engine"] == "PPO Agent")
                & (valid_df["training_seed"] == ts)
                & (valid_df["scenario"] == scenario)
            ]
            m = p_scen["success_rate_pct"].mean()
            rec[f"ppo_seed_{ts}"] = round(m, 2)
            ppo_scen_means.append(m)

        rec["ppo_macro_mean"] = round(float(np.mean(ppo_scen_means)), 2)
        rec["ppo_inter_seed_std"] = round(float(np.std(ppo_scen_means, ddof=1)), 2)
        rec["mean_diff_vs_baseline"] = round(rec["ppo_macro_mean"] - rec["baseline_mean"], 2)
        scen_records.append(rec)

    scen_df = pd.DataFrame(scen_records)
    scen_csv_path = os.path.join(output_dir, "scenario_level_breakdown.csv")
    scen_df.to_csv(scen_csv_path, index=False)
    print("\n--- Scenario-Level Breakdown ---")
    print(scen_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # C. Per-Seed Wilcoxon Signed-Rank Tests (Paired on 70 conditions)
    # -------------------------------------------------------------------------
    wilcoxon_tests = []
    for ts in training_seeds:
        p_ts = valid_df[(valid_df["engine"] == "PPO Agent") & (valid_df["training_seed"] == ts)].set_index(
            ["scenario", "eval_seed"]
        )
        b_ts = base_df.set_index(["scenario", "eval_seed"])

        common_idx = b_ts.index.intersection(p_ts.index)
        b_vals = b_ts.loc[common_idx, "success_rate_pct"]
        p_vals = p_ts.loc[common_idx, "success_rate_pct"]

        diff = b_vals - p_vals
        n_pairs = len(common_idx)
        n_zeros = int((diff == 0).sum())
        n_nonzeros = int((diff != 0).sum())

        if n_nonzeros == 0:
            stat, pval = 0.0, 1.0
        else:
            try:
                stat, pval = stats.wilcoxon(b_vals, p_vals)
            except Exception as e:
                stat, pval = float("nan"), float("nan")

        wilcoxon_tests.append({
            "training_seed": ts,
            "n_pairs": n_pairs,
            "n_zero_diff": n_zeros,
            "n_nonzero_diff": n_nonzeros,
            "baseline_mean": round(float(b_vals.mean()), 4),
            "ppo_seed_mean": round(float(p_vals.mean()), 4),
            "wilcoxon_stat": float(stat),
            "p_value": float(pval),
            "significant_p005": bool(pval < 0.05),
        })

    wilcoxon_df = pd.DataFrame(wilcoxon_tests)
    wilcoxon_csv_path = os.path.join(output_dir, "per_seed_wilcoxon_results.csv")
    wilcoxon_df.to_csv(wilcoxon_csv_path, index=False)
    print("\n--- Per-Seed Wilcoxon Signed-Rank Tests vs. Baseline ---")
    print(wilcoxon_df.to_string(index=False))

    # -------------------------------------------------------------------------
    # D. Clustered / Hierarchical Variance Decomposition
    # -------------------------------------------------------------------------
    # For PPO trials, decompose total variance of success_rate_pct into
    # between-policy variance (var across seed means) and within-policy variance.
    ppo_all = valid_df[valid_df["engine"] == "PPO Agent"]
    grand_mean = ppo_all["success_rate_pct"].mean()
    total_var = ppo_all["success_rate_pct"].var(ddof=1)

    seed_group_means = ppo_all.groupby("training_seed")["success_rate_pct"].mean()
    between_seed_var = seed_group_means.var(ddof=1)
    within_seed_var = ppo_all.groupby("training_seed")["success_rate_pct"].var().mean()

    # -------------------------------------------------------------------------
    # Recovery Action Distribution Analysis
    # -------------------------------------------------------------------------
    # Inspect the action distributions across seeds
    action_distributions = {}
    for ts in training_seeds:
        # Check action counts recorded during evaluation
        ts_subset = ppo_all[ppo_all["training_seed"] == ts]
        action_distributions[f"seed_{ts}"] = {
            "total_triggers": int(ts_subset["recovery_trigger_count"].sum()),
            "total_successes": int(ts_subset["recovery_success_count"].sum()),
            "total_fails": int(ts_subset["recovery_fail_count"].sum()),
        }

    full_stats = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "training_seeds": training_seeds,
        "inter_seed_stats": inter_seed_stats,
        "per_seed_wilcoxon": wilcoxon_tests,
        "hierarchical_variance": {
            "grand_mean": float(grand_mean),
            "total_variance": float(total_var) if not np.isnan(total_var) else 0.0,
            "between_policy_variance": float(between_seed_var) if not np.isnan(between_seed_var) else 0.0,
            "within_policy_variance": float(within_seed_var) if not np.isnan(within_seed_var) else 0.0,
        },
        "action_summary": action_distributions,
    }

    stats_json_path = os.path.join(output_dir, "phase2_statistical_analysis.json")
    with open(stats_json_path, "w") as f:
        json.dump(full_stats, f, indent=4)

    return full_stats


if __name__ == "__main__":
    evaluate_phase2_matrix()
