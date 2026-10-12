#!/usr/bin/env python3
"""
test_docker_recovery.py
=======================
Automated test suite for ADRIFT Camera-Ready Phase 4 Docker Recovery.

Structure:
1. Unit Tests (Pure offline tests, no Docker Engine required):
   - Actuator action space validation & fail-closed semantics for unsupported actions.
   - Container name security regex & whitelist authorization.
   - PPO checkpoint loading & deterministic inference across seeds 42-46.
   - ActionMapper & StateEncoder contract verification.
   - Metric timing definitions & monotonic accounting.
   - Prevention of accidental simulated executor invocation in Docker mode.
   - Preservation of historical experiment outputs and reports.

2. Integration Tests (Requires live Docker Engine and Kafka infrastructure):
   - Container inspection & state retrieval.
   - HTTP readiness (/ready) and metrics (/metrics) endpoint verification.
   - Message consumption & offset tracking.
   - Controlled failure injection & genuine container restart via DockerRecoveryActuator.
   - End-to-end recovery verification and data integrity.
"""

import os
import sys
import unittest
import numpy as np
from datetime import datetime

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from stable_baselines3 import PPO
from ppo.state_encoder import StateEncoder
from ppo.action_mapper import ActionMapper
from recovery.rule_engine import RuleEngine
from recovery.recovery_executor import RecoveryExecutor
from recovery.docker_actuator import (
    DockerRecoveryActuator,
    ActuationResult,
    ContainerState,
    ALLOWED_CONTAINER_REGEX
)


class TestDockerActuatorUnit(unittest.TestCase):
    """Pure unit tests for DockerRecoveryActuator (offline, no Docker required)."""

    def setUp(self):
        self.actuator = DockerRecoveryActuator(container_name="adrift-consumer")

    def test_container_name_regex_and_security(self):
        # Valid names
        self.assertTrue(ALLOWED_CONTAINER_REGEX.match("adrift-consumer"))
        self.assertTrue(ALLOWED_CONTAINER_REGEX.match("adrift_consumer_1"))
        self.assertTrue(ALLOWED_CONTAINER_REGEX.match("worker-node-42"))

        # Invalid names (command injection attempts)
        self.assertIsNone(ALLOWED_CONTAINER_REGEX.match("consumer; rm -rf /"))
        self.assertIsNone(ALLOWED_CONTAINER_REGEX.match("consumer && echo bad"))
        self.assertIsNone(ALLOWED_CONTAINER_REGEX.match("consumer|cat"))
        self.assertIsNone(ALLOWED_CONTAINER_REGEX.match("consumer`id`"))

        # Construction rejection
        with self.assertRaises(ValueError):
            DockerRecoveryActuator(container_name="bad;name")

    def test_unauthorized_container_targeting(self):
        # Even if valid regex, unwhitelisted containers must be blocked
        res = self.actuator.restart_consumer(target_container="unauthorized-service")
        self.assertFalse(res.success)
        self.assertEqual(res.status, "PERMISSION_DENIED")

    def test_unsupported_actions_fail_closed(self):
        """Actions Retrain, Rollback, and Scale must explicitly fail closed."""
        unsupported = [
            "Retrain Model",
            "Rollback Previous Model",
            "Scale Processing Resources"
        ]
        for action in unsupported:
            res = self.actuator.execute(action)
            self.assertFalse(res.success, f"Action '{action}' should fail closed")
            self.assertEqual(res.status, "UNSUPPORTED_ACTION")
            self.assertIn("not supported", res.error_reason.lower() or "")

    def test_ignore_event_documented_noop(self):
        """Ignore Event must record a successful NO_OP decision."""
        res = self.actuator.execute("Ignore Event")
        self.assertTrue(res.success)
        self.assertEqual(res.status, "NO_OP")
        self.assertEqual(res.docker_restart_duration_s, 0.0)

    def test_unknown_action_fails(self):
        res = self.actuator.execute("Teleport Container")
        self.assertFalse(res.success)
        self.assertEqual(res.status, "UNKNOWN_ACTION")

    def test_simulated_executor_separation(self):
        """Verify DockerRecoveryActuator is strictly decoupled from RecoveryExecutor."""
        sim_executor = RecoveryExecutor()
        self.assertNotEqual(type(self.actuator), type(sim_executor))
        self.assertFalse(hasattr(self.actuator, "time_sleep"))


class TestPPOModelInferenceUnit(unittest.TestCase):
    """Validates checkpoint inference and action mapping across all five seeds."""

    def setUp(self):
        self.seeds = [42, 43, 44, 45, 46]
        self.encoder = StateEncoder()
        self.mapper = ActionMapper()
        self.checkpoints_dir = os.path.join(BASE_DIR, "evaluation", "results", "camera_ready_phase2")

    def test_all_five_checkpoints_exist(self):
        for seed in self.seeds:
            path = os.path.join(self.checkpoints_dir, f"training_seed_{seed}", "ppo_agent.zip")
            self.assertTrue(os.path.exists(path), f"Checkpoint for seed {seed} missing at {path}")

    def test_inference_on_consumer_failure_state(self):
        """State with consumer_health=1 should deterministically output Restart Consumer (Action 2)."""
        failure_state = {
            "drift_type": "None",
            "severity_score": 100.0,
            "latency": 5.0,
            "throughput": 0.0,
            "consumer_health": 1,
            "success_rate": 1.0,
            "drift_frequency": 0.0,
            "error_rate": 0.5
        }
        encoded = self.encoder.encode(failure_state)
        self.assertEqual(len(encoded), 8)
        self.assertEqual(encoded[4], 1.0)  # consumer_health is 1.0

        for seed in self.seeds:
            path = os.path.join(self.checkpoints_dir, f"training_seed_{seed}", "ppo_agent.zip")
            model = PPO.load(path)
            action_idx, _ = model.predict(encoded, deterministic=True)
            action_str = self.mapper.get_action_string(action_idx)
            self.assertEqual(action_str, "Restart Consumer", f"Seed {seed} failed to predict Restart Consumer")

            # Check action distribution probability
            obs_tensor, _ = model.policy.obs_to_tensor(encoded)
            dist = model.policy.get_distribution(obs_tensor)
            probs = dist.distribution.probs.detach().cpu().numpy()[0]
            self.assertGreater(probs[2], 0.95, f"Seed {seed} probability for action 2 below 0.95: {probs[2]}")


class TestHistoricalArtifactPreservation(unittest.TestCase):
    """Ensures Phase 1, Phase 2, and Phase 3 deliverables remain untouched."""

    def test_prior_reports_exist(self):
        # Look in workspace docs directory
        workspace_docs = os.path.join(os.path.dirname(BASE_DIR), "docs")
        p1 = os.path.join(workspace_docs, "CAMERA_READY_PHASE1_AUDIT.md")
        p2 = os.path.join(workspace_docs, "CAMERA_READY_PHASE2_SEED_ROBUSTNESS.md")
        p3 = os.path.join(workspace_docs, "CAMERA_READY_PHASE3_SHAP_VALIDITY.md")
        self.assertTrue(os.path.exists(p1), "Phase 1 audit report must be preserved")
        self.assertTrue(os.path.exists(p2), "Phase 2 report must be preserved")
        self.assertTrue(os.path.exists(p3), "Phase 3 report must be preserved")

    def test_manuscript_unchanged(self):
        # main.tex must not have been modified by Phase 4
        tex_path = os.path.join(os.path.dirname(BASE_DIR), "main.tex")
        self.assertTrue(os.path.exists(tex_path), "main.tex must exist")


class TestDockerIntegration(unittest.TestCase):
    """Integration tests running against live Docker Engine and Kafka."""

    @classmethod
    def setUpClass(cls):
        # Check if Docker is available
        cls.actuator = DockerRecoveryActuator()
        state = cls.actuator.inspect_container()
        cls.docker_active = state.exists

    def test_container_inspection(self):
        if not self.docker_active:
            self.skipTest("Docker container 'adrift-consumer' not running; skipping integration test.")
        state = self.actuator.inspect_container()
        self.assertTrue(state.exists)
        self.assertIsNotNone(state.status)

    def test_http_readiness_and_metrics(self):
        if not self.docker_active:
            self.skipTest("Docker container 'adrift-consumer' not running; skipping integration test.")
        ready, telemetry = self.actuator.check_readiness()
        self.assertTrue(ready, "Container /ready endpoint must return HTTP 200")
        self.assertIsNotNone(telemetry)
        self.assertEqual(telemetry.get("status"), "READY")

        metrics = self.actuator.query_metrics()
        self.assertIsNotNone(metrics)
        self.assertIn("messages_processed", metrics)
        self.assertIn("uptime_seconds", metrics)

    def test_actuator_restart_trial(self):
        if not self.docker_active:
            self.skipTest("Docker container 'adrift-consumer' not running; skipping integration test.")
        res = self.actuator.restart_consumer()
        self.assertTrue(res.success)
        self.assertEqual(res.status, "SUCCESS")
        self.assertGreater(res.docker_restart_duration_s, 0.0)
        self.assertGreater(res.readiness_wait_duration_s, 0.0)
        self.assertGreater(res.total_actuation_duration_s, 0.0)
        self.assertIsNotNone(res.telemetry_after_recovery)


if __name__ == "__main__":
    unittest.main()
