"""
test_phase3_regression.py
=========================
Regression tests for Phase 3: SHAP Validity & Cross-Seed Policy Explainability.

Verifies:
1. Feature order and dimensionality (8 features).
2. Severity normalization in [0, 1].
3. Drift category mapping to {0, 1, 2, 3, 4}.
4. Consumer health semantics (0=Healthy, 1=Failed).
5. Synthetic sample validity.
6. Cross-policy checkpoint loading across all 5 seeds.
7. Action-probability wrapper shape (N, 5) and sum to 1.0.
8. SHAP output dimensions and finiteness.
9. Additivity reconstruction check (error < 0.01).
10. Preservation of previous phase artifacts.
"""

import os
import sys
import json
import unittest
import numpy as np
import torch
import shap

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from stable_baselines3 import PPO
from ppo.state_encoder import StateEncoder
from ppo.action_mapper import ActionMapper
from explainability.valid_state_generator import (
    generate_explanation_dataset,
    generate_background_dataset,
    validate_dataset,
    FEATURE_NAMES,
)
from evaluation.phase3_shap_runner import (
    make_action_prob_predictor,
    TRAINING_SEEDS,
    PHASE3_OUTPUT_DIR,
)


class TestPhase3StateSchemaAndDatasetValidity(unittest.TestCase):
    """Verifies that the state schema is mathematically and semantically correct."""

    def setUp(self):
        self.encoder = StateEncoder()

    def test_feature_order_and_dimensionality(self):
        self.assertEqual(len(FEATURE_NAMES), 8)
        self.assertEqual(self.encoder.state_dim, 8)
        self.assertEqual(
            FEATURE_NAMES,
            [
                "Drift Type", "Severity Score", "Latency", "Throughput",
                "Consumer Health", "Success Rate", "Drift Freq", "Error Rate"
            ],
        )

    def test_severity_normalization(self):
        raw_state = {
            "drift_type": "None",
            "severity_score": 85.0,
            "latency": 0.2,
            "throughput": 450.0,
            "consumer_health": 0,
            "success_rate": 1.0,
            "drift_frequency": 0.0,
            "error_rate": 0.0,
        }
        vec = self.encoder.encode(raw_state)
        # Feature 1 must be 0.85, NOT 85.0
        self.assertAlmostEqual(vec[1], 0.85, places=5)
        self.assertLessEqual(vec[1], 1.0)
        self.assertGreaterEqual(vec[1], 0.0)

    def test_drift_category_mapping(self):
        expected_mappings = {
            "None": 0.0,
            "Sudden Drift": 1.0,
            "Gradual Drift": 2.0,
            "Incremental Drift": 3.0,
            "Recurring Drift": 4.0,
        }
        for name, val in expected_mappings.items():
            vec = self.encoder.encode({"drift_type": name})
            self.assertEqual(vec[0], val, f"Drift category {name} mapped incorrectly")

    def test_consumer_health_semantics(self):
        healthy_vec = self.encoder.encode({"consumer_health": 0})
        failed_vec = self.encoder.encode({"consumer_health": 1})
        self.assertEqual(healthy_vec[4], 0.0, "Consumer health 0 must encode as 0.0")
        self.assertEqual(failed_vec[4], 1.0, "Consumer health 1 must encode as 1.0")

    def test_explanation_dataset_semantic_validity(self):
        encoded, raw_df = generate_explanation_dataset(n_samples=200, seed=42)
        self.assertEqual(encoded.shape, (200, 8))
        self.assertEqual(len(raw_df), 200)

        checks = validate_dataset(encoded)
        self.assertTrue(checks["all_finite"])
        self.assertEqual(checks["drift_type_values"], [0.0, 1.0, 2.0, 3.0, 4.0])
        self.assertGreaterEqual(checks["severity_range"][0], 0.0)
        self.assertLessEqual(checks["severity_range"][1], 1.0)


class TestPhase3ActionProbabilityWrapperAndSHAP(unittest.TestCase):
    """Verifies that PPO probability extraction and SHAP KernelExplainer are valid."""

    @classmethod
    def setUpClass(cls):
        ckpt = os.path.join(
            _BASE_DIR, "evaluation", "results", "camera_ready_phase2", "training_seed_42", "ppo_agent.zip"
        )
        cls.model = PPO.load(ckpt)

    def test_all_five_policies_load_successfully(self):
        for seed in TRAINING_SEEDS:
            ckpt = os.path.join(
                _BASE_DIR, "evaluation", "results", "camera_ready_phase2", f"training_seed_{seed}", "ppo_agent.zip"
            )
            self.assertTrue(os.path.exists(ckpt), f"Missing checkpoint for seed {seed}")
            m = PPO.load(ckpt)
            self.assertEqual(m.observation_space.shape, (8,))
            self.assertEqual(m.action_space.n, 5)

    def test_action_probability_wrapper_shape_and_sum(self):
        predictor = make_action_prob_predictor(self.model)
        test_inputs = np.random.uniform(0, 1, size=(20, 8)).astype(np.float32)
        probs = predictor(test_inputs)

        self.assertEqual(probs.shape, (20, 5))
        self.assertTrue(np.all(probs >= 0.0))
        self.assertTrue(np.all(probs <= 1.0))
        row_sums = probs.sum(axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))

    def test_shap_additivity_and_reconstruction(self):
        """Verifies local additivity on a small test batch: pred == expected_value + sum(shap_values)."""
        predictor = make_action_prob_predictor(self.model)
        bg = generate_background_dataset(n_background=10, seed=1)
        exp = generate_explanation_dataset(n_samples=5, seed=2)[0]

        explainer = shap.KernelExplainer(predictor, bg)
        sh_vals = explainer.shap_values(exp, nsamples=50)

        if isinstance(sh_vals, list):
            sh_arr = np.stack(sh_vals, axis=-1)
        else:
            sh_arr = np.array(sh_vals)

        self.assertEqual(sh_arr.shape, (5, 8, 5))
        self.assertTrue(np.isfinite(sh_arr).all())

        # Check additivity
        exp_val = np.array(explainer.expected_value)
        preds = predictor(exp)
        recon = exp_val + sh_arr.sum(axis=1)

        err = np.abs(preds - recon).max()
        self.assertLess(err, 0.01, f"Additivity error {err} exceeds tolerance 0.01")


class TestPhase3ArtifactPreservation(unittest.TestCase):
    """Verifies that all Phase 3 deliverables and prior outputs are present."""

    def test_phase3_deliverables_exist(self):
        files_to_check = [
            "explanation_states.csv",
            "background_states.csv",
            "per_seed_feature_importance.csv",
            "per_action_feature_importance.csv",
            "cross_seed_stability.csv",
            "validation_results.json",
            "shap_configuration.json",
            "figures/global_feature_importance_cross_seed.png",
            "figures/global_feature_importance_cross_seed.pdf",
            "figures/action_specific_feature_importance.png",
            "figures/policy_rank_comparison.png",
        ]
        for f in files_to_check:
            p = os.path.join(PHASE3_OUTPUT_DIR, f)
            self.assertTrue(os.path.exists(p), f"Missing expected deliverable: {p}")


if __name__ == "__main__":
    unittest.main()
