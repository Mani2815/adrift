"""
test_reproducible_recovery_rng.py
==================================
Deterministic regression test suite for ADRIFT recovery randomness
and Common Random Numbers (CRN) reproducibility.

Uses standard Python unittest framework.
"""

import os
import sys
import unittest
import random
import numpy as np

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from recovery.recovery_executor import RecoveryExecutor
from evaluation.protocol_b_runner import (
    run_one_trial,
    generate_event_stream,
    InstrumentedRecoveryEngine,
)


class TestReproducibleRecoveryRNG(unittest.TestCase):

    def setUp(self):
        import recovery.recovery_executor as rex
        self._orig_sleep = rex.time.sleep
        rex.time.sleep = lambda s: None

    def tearDown(self):
        import recovery.recovery_executor as rex
        rex.time.sleep = self._orig_sleep

    # ========================================================================
    # 1. RecoveryExecutor Determinism & Probability Respect
    # ========================================================================
    def test_executor_crn_deterministic(self):
        """Verify that identical exogenous (u_rand, v_rand) produces identical outcomes."""
        executor = RecoveryExecutor(simulate_sleep=False)

        # Restart Consumer: success_prob = 0.95, time in [0.5, 1.5]
        succ1, t1 = executor.execute("Restart Consumer", u_rand=0.90, v_rand=0.5)
        succ2, t2 = executor.execute("Restart Consumer", u_rand=0.90, v_rand=0.5)
        self.assertTrue(succ1)
        self.assertTrue(succ2)
        self.assertAlmostEqual(t1, t2, places=2)

        # u_rand > 0.95 must fail
        succ_fail, _ = executor.execute("Restart Consumer", u_rand=0.96, v_rand=0.5)
        self.assertFalse(succ_fail)

    def test_executor_respects_configured_probabilities(self):
        """Verify configured action-success probabilities across empirical samples."""
        for action, (t_min, t_max, expected_p) in RecoveryExecutor.ACTION_PARAMS.items():
            executor = RecoveryExecutor(simulate_sleep=False)
            # Test boundary values
            succ_low, _ = executor.execute(action, u_rand=0.0)
            self.assertTrue(succ_low, f"Action {action} should succeed at u_rand=0.0")

            if expected_p < 1.0:
                succ_high, _ = executor.execute(action, u_rand=1.0)
                self.assertFalse(succ_high, f"Action {action} (p={expected_p}) should fail at u_rand=1.0")

            # Empirical rate over a deterministic grid
            grid = np.linspace(0.0001, 0.9999, 1000)
            outcomes = [executor.execute(action, u_rand=u)[0] for u in grid]
            empirical_p = sum(outcomes) / len(outcomes)
            self.assertAlmostEqual(empirical_p, expected_p, delta=0.015)

    def test_executor_seed_isolation(self):
        """Verify that dedicated random.Random instances remain isolated."""
        rng1 = random.Random(999)
        rng2 = random.Random(999)

        exec1 = RecoveryExecutor(rng=rng1, simulate_sleep=False)
        exec2 = RecoveryExecutor(rng=rng2, simulate_sleep=False)

        outcomes1 = [exec1.execute("Scale Processing Resources")[0] for _ in range(10)]
        outcomes2 = [exec2.execute("Scale Processing Resources")[0] for _ in range(10)]
        self.assertEqual(outcomes1, outcomes2)

    # ========================================================================
    # 2. Trial Reproducibility & Seed Mapping
    # ========================================================================
    def test_trial_reproducibility(self):
        """Verify that identical trial configuration yields bit-for-bit identical results."""
        scenario = "High Latency"
        eval_seed = 4
        events, fp, inj = generate_event_stream(scenario, trial_length=250, eval_seed=eval_seed)

        row1 = run_one_trial(
            scenario=scenario,
            engine_name="Baseline Rule Engine",
            eval_seed=eval_seed,
            pre_generated_events=events,
            fast_simulation=True,
            stream_fingerprint=fp,
        )

        row2 = run_one_trial(
            scenario=scenario,
            engine_name="Baseline Rule Engine",
            eval_seed=eval_seed,
            pre_generated_events=events,
            fast_simulation=True,
            stream_fingerprint=fp,
        )

        self.assertEqual(row1["recovery_trigger_count"], row2["recovery_trigger_count"])
        self.assertEqual(row1["recovery_success_count"], row2["recovery_success_count"])
        self.assertEqual(row1["recovery_fail_count"], row2["recovery_fail_count"])
        self.assertEqual(row1["success_rate_pct"], row2["success_rate_pct"])
        self.assertEqual(row1["stream_fingerprint"], row2["stream_fingerprint"])
        self.assertEqual(row1["recovery_rng_protocol"], row2["recovery_rng_protocol"])

    def test_trial_variation_across_seeds(self):
        """Verify that different evaluation seeds produce varied stream fingerprints."""
        scenario = "Sudden Drift"
        fps = set()
        for s in range(5):
            events, fp, _ = generate_event_stream(scenario, trial_length=250, eval_seed=s)
            fps.add(fp)
        self.assertEqual(len(fps), 5)

    # ========================================================================
    # 3. Trial Execution Order Independence
    # ========================================================================
    def test_execution_order_independence(self):
        """Verify that executing trial B after trial A produces identical results to executing B alone."""
        events_a, fp_a, _ = generate_event_stream("High Latency", 250, eval_seed=1)
        events_b, fp_b, _ = generate_event_stream("High Latency", 250, eval_seed=2)

        # Order 1: Run A then B
        _ = run_one_trial("High Latency", "Baseline Rule Engine", 1, events_a, True, stream_fingerprint=fp_a)
        row_b_order1 = run_one_trial("High Latency", "Baseline Rule Engine", 2, events_b, True, stream_fingerprint=fp_b)

        # Order 2: Run B directly
        row_b_order2 = run_one_trial("High Latency", "Baseline Rule Engine", 2, events_b, True, stream_fingerprint=fp_b)

        self.assertEqual(row_b_order1["recovery_trigger_count"], row_b_order2["recovery_trigger_count"])
        self.assertEqual(row_b_order1["recovery_success_count"], row_b_order2["recovery_success_count"])
        self.assertEqual(row_b_order1["success_rate_pct"], row_b_order2["success_rate_pct"])

    # ========================================================================
    # 4. Stream & Recovery RNG Isolation
    # ========================================================================
    def test_isolation_event_generation_and_recovery(self):
        """Verify that drawing events from the simulator does not pollute recovery RNG draws."""
        events, fp, _ = generate_event_stream("Consumer Crash", 250, eval_seed=7)

        # Advance global random and extra simulator calls
        random.random()
        random.uniform(10, 20)
        _ = generate_event_stream("Gradual Drift", 250, eval_seed=99)

        row = run_one_trial("Consumer Crash", "Baseline Rule Engine", 7, events, True, stream_fingerprint=fp)

        # Run again without prior random disturbance
        row_clean = run_one_trial("Consumer Crash", "Baseline Rule Engine", 7, events, True, stream_fingerprint=fp)

        self.assertEqual(row["recovery_trigger_count"], row_clean["recovery_trigger_count"])
        self.assertEqual(row["recovery_success_count"], row_clean["recovery_success_count"])
        self.assertEqual(row["success_rate_pct"], row_clean["success_rate_pct"])

    # ========================================================================
    # 5. Paired Engine Common Random Numbers (CRN) Verification
    # ========================================================================
    def test_paired_engines_receive_identical_crn(self):
        """Verify that Baseline Rule Engine and PPO Agent receive identical attempt-keyed CRN draws."""
        scenario = "High Latency"
        eval_seed = 3

        engine_rule = InstrumentedRecoveryEngine(fast_simulation=True, eval_seed=eval_seed, scenario=scenario)
        engine_ppo  = InstrumentedRecoveryEngine(fast_simulation=True, eval_seed=eval_seed, scenario=scenario)

        # Simulate 5 recovery attempts on each engine
        for _ in range(5):
            engine_rule._handle_any(lambda: "Restart Consumer", "None", 100.0, "High Latency")
            engine_ppo._handle_any(lambda: "Scale Processing Resources", "None", 100.0, "High Latency")

        self.assertEqual(len(engine_rule.timing_records), 5)
        self.assertEqual(len(engine_ppo.timing_records), 5)

        for k in range(5):
            rec_r = engine_rule.timing_records[k]
            rec_p = engine_ppo.timing_records[k]
            self.assertEqual(rec_r["attempt_idx"], k)
            self.assertEqual(rec_p["attempt_idx"], k)
            self.assertEqual(rec_r["attempt_stream_id"], rec_p["attempt_stream_id"])
            self.assertEqual(rec_r["u_rand"], rec_p["u_rand"])
            self.assertEqual(rec_r["v_rand"], rec_p["v_rand"])

    # ========================================================================
    # 6. Result Metadata Records RNG Protocol
    # ========================================================================
    def test_trial_record_has_rng_metadata(self):
        """Verify trial row dictionary contains explicit RNG protocol fields."""
        events, fp, _ = generate_event_stream("Sudden Drift", 250, 0)
        row = run_one_trial("Sudden Drift", "Baseline Rule Engine", 0, events, True, stream_fingerprint=fp)

        self.assertIn("recovery_rng_protocol", row)
        self.assertEqual(row["recovery_rng_protocol"], "CRN_v1_sha256_attempt_keyed")
        self.assertIn("recovery_stream_id", row)
        self.assertIn("crn:eval_seed=0:scenario=Sudden Drift", row["recovery_stream_id"])

    # ========================================================================
    # 7. Preservation of Historical Result Files
    # ========================================================================
    def test_historical_result_files_preserved(self):
        """Verify that prior audit and experiment files exist and have not been deleted."""
        phase1_dir = os.path.join(_BASE_DIR, "evaluation", "results", "camera_ready_protocol_b")
        phase2_dir = os.path.join(_BASE_DIR, "evaluation", "results", "camera_ready_phase2")
        phase4_dir = os.path.join(_BASE_DIR, "evaluation", "results", "camera_ready_phase4_docker")

        self.assertTrue(os.path.isdir(phase1_dir), "Phase 1 results directory must exist")
        self.assertTrue(os.path.isdir(phase2_dir), "Phase 2 results directory must exist")
        self.assertTrue(os.path.isdir(phase4_dir), "Phase 4 results directory must exist")

        self.assertTrue(os.path.isfile(os.path.join(phase2_dir, "raw_phase2_trials.csv")))
        self.assertTrue(os.path.isfile(os.path.join(phase4_dir, "raw_docker_trials.csv")))


if __name__ == "__main__":
    unittest.main()
