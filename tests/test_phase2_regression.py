"""
test_phase2_regression.py
=========================
Regression tests for Phase 2: Independent PPO Training-Seed Robustness.

Verifies:
1. Training seed configurability.
2. Checkpoint and log isolation by seed directory (no cross-overwriting).
3. Checkpoint SHA-256 integrity hashing.
4. Model loading from seed-specific checkpoints via custom model_path.
5. Stream fingerprint identity across evaluated models.
6. Separation of training_seed from eval_seed in recorded data.
7. Preservation of Phase 1 and legacy results.
"""

import os
import sys
import json
import unittest
import tempfile
import shutil

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from ppo.multi_seed_trainer import (
    train_single_seed,
    compute_file_sha256,
    PPO_HYPERPARAMETERS,
    DEFAULT_TRAINING_SEEDS,
)
from ppo.ppo_inference import PPOInferenceEngine
from evaluation.protocol_b_runner import generate_event_stream, run_one_trial
from config.settings import settings


class TestPhase2TrainingConfigAndIsolation(unittest.TestCase):
    """Verifies configurable training seeds and isolated artifact directories."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="adrift_phase2_test_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_training_seed_isolation_and_no_overwriting(self):
        """Trains 2 small pilot runs (seeds 901 and 902) to verify isolation."""
        # 2048 steps minimum for 1 PPO rollout
        m1 = train_single_seed(seed=901, total_timesteps=2048, base_output_dir=self.test_dir)
        m2 = train_single_seed(seed=902, total_timesteps=2048, base_output_dir=self.test_dir)

        dir1 = os.path.join(self.test_dir, "training_seed_901")
        dir2 = os.path.join(self.test_dir, "training_seed_902")

        self.assertTrue(os.path.isdir(dir1), "Directory for seed 901 must exist")
        self.assertTrue(os.path.isdir(dir2), "Directory for seed 902 must exist")

        ckpt1 = os.path.join(dir1, "ppo_agent.zip")
        ckpt2 = os.path.join(dir2, "ppo_agent.zip")

        self.assertTrue(os.path.isfile(ckpt1))
        self.assertTrue(os.path.isfile(ckpt2))

        # Checkpoints should be distinct files with distinct SHA-256 hashes
        hash1 = compute_file_sha256(ckpt1)
        hash2 = compute_file_sha256(ckpt2)

        self.assertEqual(m1["checkpoint_sha256"], hash1)
        self.assertEqual(m2["checkpoint_sha256"], hash2)
        self.assertNotEqual(hash1, hash2, "Independent seeds should yield distinct weights/hashes")

    def test_seed_specific_model_loading(self):
        """Verifies that PPOInferenceEngine loads models from custom paths correctly."""
        train_single_seed(seed=903, total_timesteps=2048, base_output_dir=self.test_dir)
        ckpt = os.path.join(self.test_dir, "training_seed_903", "ppo_agent.zip")

        engine = PPOInferenceEngine(model_path=ckpt)
        self.assertIsNotNone(engine.model, "Model should load successfully from custom path")

        # Test prediction works
        dummy_state = {
            "drift_type": "Sudden Drift",
            "severity_score": 75.0,
            "latency": 0.3,
            "throughput": 450.0,
            "consumer_health": 0,
            "success_rate": 1.0,
            "drift_frequency": 0.1,
            "error_rate": 0.05,
        }
        action = engine.predict_action(dummy_state)
        self.assertIsInstance(action, str)
        self.assertIn(action, [
            "Retrain Model", "Restart Consumer", "Rollback Previous Model",
            "Scale Processing Resources", "Ignore Event"
        ])


class TestPhase2EvaluationMatrixProtocol(unittest.TestCase):
    """Verifies evaluation stream identity and seed separation."""

    def test_training_seed_and_eval_seed_separation(self):
        """Verifies that training_seed is recorded separately from eval_seed."""
        events, fp, _ = generate_event_stream("Sudden Drift", 250, eval_seed=5)

        row = run_one_trial(
            scenario="Sudden Drift",
            engine_name="PPO Agent",
            eval_seed=5,
            pre_generated_events=events,
            fast_simulation=True,
            training_seed=43,
            stream_fingerprint=fp,
            model_path=None,
        )

        self.assertEqual(row["training_seed"], 43)
        self.assertEqual(row["eval_seed"], 5)
        self.assertEqual(row["stream_fingerprint"], fp)

    def test_stream_fingerprint_match_across_engines(self):
        """Verifies stream fingerprint is identical for baseline and PPO."""
        scenario = "High Latency"
        eval_seed = 3
        events, fp, _ = generate_event_stream(scenario, 250, eval_seed)

        r_base = run_one_trial(
            scenario=scenario,
            engine_name="Baseline Rule Engine",
            eval_seed=eval_seed,
            pre_generated_events=events,
            fast_simulation=True,
            training_seed="N/A",
            stream_fingerprint=fp,
        )

        r_ppo = run_one_trial(
            scenario=scenario,
            engine_name="PPO Agent",
            eval_seed=eval_seed,
            pre_generated_events=events,
            fast_simulation=True,
            training_seed=42,
            stream_fingerprint=fp,
        )

        self.assertEqual(r_base["stream_fingerprint"], r_ppo["stream_fingerprint"])
        self.assertEqual(r_base["stream_fingerprint"], fp)


class TestPreservationOfPriorOutputs(unittest.TestCase):
    """Verifies original submitted results and Phase 1 outputs remain untouched."""

    def test_original_submitted_and_phase1_preserved(self):
        orig_csv = os.path.join(_BASE_DIR, "evaluation", "results", "results.csv")
        self.assertTrue(os.path.exists(orig_csv))

        p1_dir = os.path.join(_BASE_DIR, "evaluation", "results", "camera_ready_protocol_b")
        self.assertTrue(os.path.exists(p1_dir))

        p1_raw = os.path.join(p1_dir, "raw_trials_20261012T163133Z.csv")
        self.assertTrue(os.path.exists(p1_raw))


if __name__ == "__main__":
    unittest.main()
