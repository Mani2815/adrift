"""
valid_state_generator.py
========================
Camera-Ready Phase 3: Valid State Generator for Explainable RL.

Generates a representative, semantically valid explanation dataset (200 instances)
and background reference dataset (25 instances) for PPO SHAP KernelExplainer analysis.

Key validity rules:
1. Feature 0 (Drift Type): Categorical mapped to {0: None, 1: Sudden, 2: Gradual, 3: Incremental, 4: Recurring}.
2. Feature 1 (Severity Score): Normalized strictly in [0.0, 1.0] (raw score / 100.0).
3. Feature 2 (Latency): Scaled to [0.0, 1.0] (min(raw_latency / 5.0, 1.0)).
4. Feature 3 (Throughput): Scaled to [0.0, 1.0] (min(raw_tput / 500.0, 1.0)).
5. Feature 4 (Consumer Health): Binary {0: Healthy, 1: Failed/Crash}.
6. Feature 5 (Success Rate): In [0.0, 1.0].
7. Feature 6 (Drift Frequency): In [0.0, 1.0].
8. Feature 7 (Error Rate): In [0.0, 1.0].
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from ppo.state_encoder import StateEncoder

FEATURE_NAMES = [
    "Drift Type",
    "Severity Score",
    "Latency",
    "Throughput",
    "Consumer Health",
    "Success Rate",
    "Drift Freq",
    "Error Rate",
]


def generate_single_state_dict(scenario_mode: str, rng: random.Random) -> dict:
    """
    Generates a semantically consistent raw pipeline state dictionary
    for a given scenario mode.
    """
    if scenario_mode == "Healthy":
        return {
            "scenario": "Healthy",
            "drift_type": "None",
            "severity_score": rng.uniform(0.0, 15.0),
            "latency": rng.uniform(0.1, 0.4),
            "throughput": rng.uniform(420.0, 490.0),
            "consumer_health": 0,
            "success_rate": 1.0,
            "drift_frequency": 0.0,
            "error_rate": rng.uniform(0.0, 0.01),
        }
    elif scenario_mode == "Sudden Drift":
        return {
            "scenario": "Sudden Drift",
            "drift_type": "Sudden Drift",
            "severity_score": rng.uniform(70.0, 95.0),
            "latency": rng.uniform(0.2, 0.6),
            "throughput": rng.uniform(300.0, 450.0),
            "consumer_health": 0,
            "success_rate": rng.uniform(0.85, 1.0),
            "drift_frequency": rng.uniform(0.05, 0.2),
            "error_rate": rng.uniform(0.05, 0.25),
        }
    elif scenario_mode == "Gradual Drift":
        return {
            "scenario": "Gradual Drift",
            "drift_type": "Gradual Drift",
            "severity_score": rng.uniform(40.0, 70.0),
            "latency": rng.uniform(0.2, 0.5),
            "throughput": rng.uniform(350.0, 470.0),
            "consumer_health": 0,
            "success_rate": rng.uniform(0.9, 1.0),
            "drift_frequency": rng.uniform(0.02, 0.15),
            "error_rate": rng.uniform(0.02, 0.15),
        }
    elif scenario_mode == "Incremental Drift":
        return {
            "scenario": "Incremental Drift",
            "drift_type": "Incremental Drift",
            "severity_score": rng.uniform(20.0, 50.0),
            "latency": rng.uniform(0.15, 0.45),
            "throughput": rng.uniform(380.0, 480.0),
            "consumer_health": 0,
            "success_rate": rng.uniform(0.95, 1.0),
            "drift_frequency": rng.uniform(0.01, 0.1),
            "error_rate": rng.uniform(0.01, 0.05),
        }
    elif scenario_mode == "Recurring Drift":
        return {
            "scenario": "Recurring Drift",
            "drift_type": "Recurring Drift",
            "severity_score": rng.uniform(65.0, 90.0),
            "latency": rng.uniform(0.2, 0.6),
            "throughput": rng.uniform(320.0, 440.0),
            "consumer_health": 0,
            "success_rate": rng.uniform(0.85, 1.0),
            "drift_frequency": rng.uniform(0.15, 0.4),
            "error_rate": rng.uniform(0.05, 0.2),
        }
    elif scenario_mode == "Consumer Crash":
        return {
            "scenario": "Consumer Crash",
            "drift_type": "None",
            "severity_score": 100.0,
            "latency": 10.0,
            "throughput": 0.0,
            "consumer_health": 1,
            "success_rate": rng.uniform(0.5, 0.9),
            "drift_frequency": rng.uniform(0.0, 0.1),
            "error_rate": rng.uniform(0.2, 0.6),
        }
    elif scenario_mode == "High Latency":
        return {
            "scenario": "High Latency",
            "drift_type": "None",
            "severity_score": rng.uniform(50.0, 85.0),
            "latency": rng.uniform(2.5, 5.0),
            "throughput": rng.uniform(60.0, 180.0),
            "consumer_health": 0,
            "success_rate": rng.uniform(0.7, 0.95),
            "drift_frequency": rng.uniform(0.0, 0.1),
            "error_rate": rng.uniform(0.1, 0.35),
        }
    elif scenario_mode == "Data Processing Failure":
        return {
            "scenario": "Data Processing Failure",
            "drift_type": "None",
            "severity_score": rng.uniform(60.0, 100.0),
            "latency": rng.uniform(0.4, 1.2),
            "throughput": rng.uniform(200.0, 350.0),
            "consumer_health": 0,
            "success_rate": rng.uniform(0.6, 0.9),
            "drift_frequency": rng.uniform(0.0, 0.15),
            "error_rate": rng.uniform(0.4, 0.8),
        }
    else:
        raise ValueError(f"Unknown scenario_mode: {scenario_mode}")


def generate_explanation_dataset(
    n_samples: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Generates n_samples explanation states balanced across all 8 operating regimes
    (7 fault/drift scenarios + Healthy baseline), returning:
    (encoded_features_array, raw_states_df).
    """
    rng = random.Random(seed)
    encoder = StateEncoder()

    modes = [
        "Healthy",
        "Sudden Drift",
        "Gradual Drift",
        "Incremental Drift",
        "Recurring Drift",
        "Consumer Crash",
        "High Latency",
        "Data Processing Failure",
    ]

    samples_per_mode = n_samples // len(modes)  # 200 // 8 = 25 each
    remainder = n_samples % len(modes)

    raw_dicts = []
    encoded_rows = []

    for idx, mode in enumerate(modes):
        count = samples_per_mode + (1 if idx < remainder else 0)
        for _ in range(count):
            s_dict = generate_single_state_dict(mode, rng)
            vec = encoder.encode(s_dict)
            raw_dicts.append(s_dict)
            encoded_rows.append(vec)

    encoded_arr = np.array(encoded_rows, dtype=np.float32)
    raw_df = pd.DataFrame(raw_dicts)
    return encoded_arr, raw_df


def generate_background_dataset(
    n_background: int = 25,
    seed: int = 999,
) -> np.ndarray:
    """
    Generates a stratified background reference population (default: 25 samples).
    """
    bg_encoded, _ = generate_explanation_dataset(n_samples=n_background, seed=seed)
    return bg_encoded


def validate_dataset(encoded_arr: np.ndarray, feature_names=FEATURE_NAMES) -> dict:
    """
    Performs exhaustive programmatic validation on an encoded state array.
    """
    n_rows, n_cols = encoded_arr.shape
    assert n_cols == 8, f"Expected 8 features, got {n_cols}"

    checks = {
        "n_samples": n_rows,
        "n_features": n_cols,
        "all_finite": bool(np.isfinite(encoded_arr).all()),
        "drift_type_values": sorted(list(set(encoded_arr[:, 0].tolist()))),
        "severity_range": [float(encoded_arr[:, 1].min()), float(encoded_arr[:, 1].max())],
        "latency_range": [float(encoded_arr[:, 2].min()), float(encoded_arr[:, 2].max())],
        "throughput_range": [float(encoded_arr[:, 3].min()), float(encoded_arr[:, 3].max())],
        "consumer_health_values": sorted(list(set(encoded_arr[:, 4].tolist()))),
        "success_rate_range": [float(encoded_arr[:, 5].min()), float(encoded_arr[:, 5].max())],
        "drift_freq_range": [float(encoded_arr[:, 6].min()), float(encoded_arr[:, 6].max())],
        "error_rate_range": [float(encoded_arr[:, 7].min()), float(encoded_arr[:, 7].max())],
    }

    # Range assertions
    assert checks["all_finite"], "Found non-finite values"
    assert set(checks["drift_type_values"]).issubset({0.0, 1.0, 2.0, 3.0, 4.0}), "Invalid drift mapping"
    assert 0.0 <= checks["severity_range"][0] and checks["severity_range"][1] <= 1.0, "Severity out of [0, 1]"
    assert set(checks["consumer_health_values"]).issubset({0.0, 1.0}), "Consumer health not binary {0, 1}"
    assert 0.0 <= checks["success_rate_range"][0] and checks["success_rate_range"][1] <= 1.0
    return checks
