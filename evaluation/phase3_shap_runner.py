"""
phase3_shap_runner.py
=====================
Camera-Ready Phase 3: Validated Action-Probability SHAP Attribution Runner.

Executes SHAP KernelExplainer across all five independently trained PPO policies
(seeds 42, 43, 44, 45, 46) using:
1. Semantically valid explanation (N=200) and background (N=25) datasets.
2. True action-probability outputs (5 discrete actions) rather than continuous action IDs.
3. Additivity/reconstruction checks, finiteness, and range validation.
4. Per-seed, per-action, and cross-seed stability analyses with publication-quality plots.
"""

import os
import sys
import time
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from stable_baselines3 import PPO
from explainability.valid_state_generator import (
    generate_explanation_dataset,
    generate_background_dataset,
    validate_dataset,
    FEATURE_NAMES,
)
from ppo.action_mapper import ActionMapper

PHASE3_OUTPUT_DIR = os.path.join(_BASE_DIR, "evaluation", "results", "camera_ready_phase3_shap")
FIG_DIR = os.path.join(PHASE3_OUTPUT_DIR, "figures")

TRAINING_SEEDS = [42, 43, 44, 45, 46]
ACTION_NAMES = [
    "Ignore Event",
    "Retrain Model",
    "Restart Consumer",
    "Rollback Previous Model",
    "Scale Processing Resources",
]


def make_action_prob_predictor(model):
    """
    Creates a callable wrapper returning a 2D numpy array of shape (N, 5)
    containing valid action probability distributions.
    """
    def _predict(states_array):
        # Ensure 2D float32 tensor
        states_tensor = torch.as_tensor(states_array, dtype=torch.float32).to(model.policy.device)
        with torch.no_grad():
            dist = model.policy.get_distribution(states_tensor)
            probs = dist.distribution.probs.cpu().numpy()
        return probs

    return _predict


def run_phase3_shap_analysis(
    training_seeds=None,
    n_explanation: int = 200,
    n_background: int = 25,
    output_dir: str = PHASE3_OUTPUT_DIR,
) -> dict:
    if training_seeds is None:
        training_seeds = TRAINING_SEEDS

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 80)
    print("ADRIFT Phase 3: SHAP Validity & Cross-Seed Policy Explainability")
    print(f"  Target Policies    : {training_seeds} ({len(training_seeds)} independent seeds)")
    print(f"  Explanation states : {n_explanation} instances (stratified)")
    print(f"  Background states  : {n_background} instances (reference population)")
    print(f"  Action space       : {ACTION_NAMES}")
    print(f"  Output directory   : {output_dir}")
    print("=" * 80)

    # 1. Generate & Validate Explanation and Background Datasets
    print("\n--- 1. Generating & Validating Datasets ---")
    explanation_encoded, raw_explanation_df = generate_explanation_dataset(
        n_samples=n_explanation, seed=42
    )
    background_encoded = generate_background_dataset(n_background=n_background, seed=999)

    val_explanation = validate_dataset(explanation_encoded)
    val_background = validate_dataset(background_encoded)

    # Save datasets
    exp_csv_path = os.path.join(output_dir, "explanation_states.csv")
    raw_explanation_df.to_csv(exp_csv_path, index=False)

    exp_encoded_csv = os.path.join(output_dir, "explanation_states_encoded.csv")
    pd.DataFrame(explanation_encoded, columns=FEATURE_NAMES).to_csv(exp_encoded_csv, index=False)

    bg_csv_path = os.path.join(output_dir, "background_states.csv")
    pd.DataFrame(background_encoded, columns=FEATURE_NAMES).to_csv(bg_csv_path, index=False)

    dataset_metadata = {
        "n_explanation": n_explanation,
        "n_background": n_background,
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
        "explanation_validation": val_explanation,
        "background_validation": val_background,
    }
    with open(os.path.join(output_dir, "dataset_metadata.json"), "w") as f:
        json.dump(dataset_metadata, f, indent=4)
    print(f"✅ Datasets generated and validated. Saved to {output_dir}")

    # 2. Iterate through all 5 trained models
    per_seed_importances = []       # Global macro importance across features
    per_action_records = []         # Feature importance broken down by action
    shap_arrays_dict = {}           # Raw arrays
    validation_records = []

    for seed in training_seeds:
        ckpt_path = os.path.join(
            _BASE_DIR, "evaluation", "results", "camera_ready_phase2", f"training_seed_{seed}", "ppo_agent.zip"
        )
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Missing policy checkpoint for seed {seed} at {ckpt_path}")

        print(f"\n--- Explaining Policy Seed {seed} ---")
        t0 = time.time()
        model = PPO.load(ckpt_path)
        predictor = make_action_prob_predictor(model)

        # Sanity check predictor on background
        bg_probs = predictor(background_encoded)
        assert bg_probs.shape == (n_background, 5)
        assert np.allclose(bg_probs.sum(axis=1), 1.0, atol=1e-5)

        # Initialize KernelExplainer
        explainer = shap.KernelExplainer(predictor, background_encoded)

        # Compute SHAP values on explanation states
        # shap_values shape: (N, D, K) where N=200, D=8 features, K=5 actions
        shap_values = explainer.shap_values(explanation_encoded, nsamples=100)

        # Handle list vs array formatting across SHAP library versions
        if isinstance(shap_values, list):
            # List of 5 arrays, each (N, 8) -> stack to (N, 8, 5)
            shap_arr = np.stack(shap_values, axis=-1)
        else:
            shap_arr = np.array(shap_values)

        t1 = time.time()
        run_time_s = t1 - t0
        print(f"  SHAP computation completed in {run_time_s:.2f}s. Array shape: {shap_arr.shape}")

        # Additivity & reconstruction check
        # Expected value shape: (5,)
        exp_val = explainer.expected_value
        if isinstance(exp_val, list):
            exp_val = np.array(exp_val)

        # Predict actual outputs
        actual_preds = predictor(explanation_encoded)  # (N, 5)

        # Reconstructed = exp_val + sum_j(shap_values)
        reconstructed = exp_val + shap_arr.sum(axis=1)  # (N, 5)
        reconstruction_error = float(np.abs(actual_preds - reconstructed).max())
        print(f"  Max additivity reconstruction error: {reconstruction_error:.6f}")

        val_rec = {
            "training_seed": seed,
            "runtime_seconds": round(run_time_s, 2),
            "shap_array_shape": list(shap_arr.shape),
            "all_finite": bool(np.isfinite(shap_arr).all()),
            "max_reconstruction_error": round(reconstruction_error, 8),
            "additivity_pass_tol_005": bool(reconstruction_error < 0.05),
        }
        validation_records.append(val_rec)

        # Compute Mean Absolute SHAP Attribution
        # 1. Per-action mean |SHAP| for each feature: shape (8, 5)
        mean_abs_by_action = np.abs(shap_arr).mean(axis=0)  # (8, 5)

        # 2. Overall global mean |SHAP| across all actions: shape (8,)
        global_mean_abs = mean_abs_by_action.mean(axis=1)   # (8,)

        # Store for seed summary
        seed_rec = {"training_seed": seed}
        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            seed_rec[feat_name] = round(float(global_mean_abs[feat_idx]), 6)
        per_seed_importances.append(seed_rec)

        # Store per-action breakdown
        for act_idx, act_name in enumerate(ACTION_NAMES):
            for feat_idx, feat_name in enumerate(FEATURE_NAMES):
                per_action_records.append({
                    "training_seed": seed,
                    "action_index": act_idx,
                    "action_name": act_name,
                    "feature_name": feat_name,
                    "mean_abs_shap": round(float(mean_abs_by_action[feat_idx, act_idx]), 6),
                })

        shap_arrays_dict[f"seed_{seed}"] = shap_arr

    # 3. Create Summary Tables
    print("\n--- 3. Aggregating Cross-Seed Stability ---")
    seed_imp_df = pd.DataFrame(per_seed_importances)
    seed_imp_csv = os.path.join(output_dir, "per_seed_feature_importance.csv")
    seed_imp_df.to_csv(seed_imp_csv, index=False)

    action_imp_df = pd.DataFrame(per_action_records)
    action_imp_csv = os.path.join(output_dir, "per_action_feature_importance.csv")
    action_imp_df.to_csv(action_imp_csv, index=False)

    # Cross-seed stability table: mean, std, and rank across the 5 policies
    stability_records = []
    for feat_name in FEATURE_NAMES:
        vals = seed_imp_df[feat_name].values
        stability_records.append({
            "feature": feat_name,
            "mean_importance": round(float(np.mean(vals)), 6),
            "std_importance": round(float(np.std(vals, ddof=1)), 6),
            "min_importance": round(float(np.min(vals)), 6),
            "max_importance": round(float(np.max(vals)), 6),
            "cv_pct": round(float(np.std(vals, ddof=1) / max(1e-6, np.mean(vals)) * 100.0), 2),
        })

    stability_df = pd.DataFrame(stability_records).sort_values(by="mean_importance", ascending=False).reset_index(drop=True)
    stability_df["overall_rank"] = range(1, len(stability_df) + 1)
    stability_csv = os.path.join(output_dir, "cross_seed_stability.csv")
    stability_df.to_csv(stability_csv, index=False)

    print("\nCross-Seed Feature Importance Ranking:")
    print(stability_df[["overall_rank", "feature", "mean_importance", "std_importance", "cv_pct"]].to_string(index=False))

    # Save Validation Results
    val_path = os.path.join(output_dir, "validation_results.json")
    with open(val_path, "w") as f:
        json.dump(validation_records, f, indent=4)

    # Save SHAP Config
    config_dict = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "shap_version": shap.__version__,
        "n_explanation": n_explanation,
        "n_background": n_background,
        "kernel_nsamples": 100,
        "training_seeds": training_seeds,
        "feature_names": FEATURE_NAMES,
        "action_names": ACTION_NAMES,
    }
    with open(os.path.join(output_dir, "shap_configuration.json"), "w") as f:
        json.dump(config_dict, f, indent=4)

    # 4. Generate Visualizations
    print("\n--- 4. Generating Publication Figures ---")
    _generate_figures(seed_imp_df, stability_df, action_imp_df, FIG_DIR)

    return {
        "stability_df": stability_df,
        "seed_imp_df": seed_imp_df,
        "action_imp_df": action_imp_df,
        "validation_records": validation_records,
    }


def _generate_figures(seed_imp_df, stability_df, action_imp_df, fig_dir):
    """Generates publication-quality figures for the camera-ready report."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Figure 1: Cross-Seed Global Feature Importance
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
    ordered_feats = stability_df["feature"].tolist()
    mean_vals = stability_df["mean_importance"].tolist()
    std_vals = stability_df["std_importance"].tolist()

    y_pos = np.arange(len(ordered_feats))
    ax.barh(y_pos, mean_vals, xerr=std_vals, align="center", color="#2b5c8f", ecolor="#d95f02", capsize=4, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ordered_feats, fontsize=10)
    ax.invert_yaxis()  # top-down
    ax.set_xlabel("Mean Absolute SHAP Value (Impact on Action Probabilities)", fontsize=11)
    ax.set_title("ADRIFT: Cross-Seed PPO Feature Attribution (N=5 Independent Policies)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "global_feature_importance_cross_seed.png"), dpi=300)
    fig.savefig(os.path.join(fig_dir, "global_feature_importance_cross_seed.pdf"))
    plt.close(fig)

    # Figure 2: Action-Specific Feature Importance Heatmap
    # Pivot action_imp_df: rows = features, cols = actions (average across seeds)
    pivot_action = action_imp_df.groupby(["feature_name", "action_name"])["mean_abs_shap"].mean().unstack()
    # Reindex rows to match overall rank
    pivot_action = pivot_action.reindex(ordered_feats)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=300)
    sns.heatmap(pivot_action, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Mean |SHAP Value|"}, ax=ax)
    ax.set_title("Action-Specific Feature Attributions (Averaged Across 5 Policies)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recovery Action", fontsize=11)
    ax.set_ylabel("State Feature", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "action_specific_feature_importance.png"), dpi=300)
    fig.savefig(os.path.join(fig_dir, "action_specific_feature_importance.pdf"))
    plt.close(fig)

    # Figure 3: Seed-by-Seed Policy Feature Importance Comparison
    melted = seed_imp_df.melt(id_vars=["training_seed"], var_name="feature", value_name="importance")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    sns.barplot(
        data=melted,
        x="feature",
        y="importance",
        hue="training_seed",
        palette="viridis",
        order=ordered_feats,
        ax=ax,
    )
    ax.set_title("Feature Importance by Independent Training Seed (Seeds 42–46)", fontsize=12, fontweight="bold")
    ax.set_xlabel("State Feature", fontsize=11)
    ax.set_ylabel("Mean |SHAP Value|", fontsize=11)
    ax.legend(title="Training Seed", frameon=True)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "policy_rank_comparison.png"), dpi=300)
    fig.savefig(os.path.join(fig_dir, "policy_rank_comparison.pdf"))
    plt.close(fig)

    print(f"✅ Generated 3 publication figures in {fig_dir}")


if __name__ == "__main__":
    run_phase3_shap_analysis()
