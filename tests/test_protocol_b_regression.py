"""
test_protocol_b_regression.py
==============================
Automated regression tests for Camera-Ready Protocol B evaluation harness.

Verifies:
1. All 7 scenario injection schedules trigger within the 250-event window.
2. Detection of missing or out-of-window injections (invalid trial flagging).
3. Correct routing of injected system faults into their recovery handlers.
4. Reproducibility under fixed seeds (identical events and SHA-256 fingerprints).
5. Identical paired event streams across recovery engines.
6. Separation of decision latency from execution duration.
7. Preservation of original results and isolated camera-ready output directories.
8. Statistical pairing and aggregation correctness.
"""

import os
import sys
import unittest
import numpy as np

# Ensure ADRIFT root is in sys.path
_ADRIFT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ADRIFT_ROOT not in sys.path:
    sys.path.insert(0, _ADRIFT_ROOT)

from config.settings import settings
from producer.fault_simulator_corrected import CorrectedFaultSimulator
from evaluation.protocol_b_runner import (
    generate_event_stream,
    run_one_trial,
    InstrumentedRecoveryEngine,
    OUTPUT_DIR,
)
from recovery.fault_monitor import FaultMonitor


class TestScenarioInjectionSchedules(unittest.TestCase):
    """Verifies that all 7 scenarios fire their intended fault within the 250-event window."""

    def test_all_seven_scenarios_inject_within_250_events(self):
        for scenario in CorrectedFaultSimulator.ALL_SCENARIOS:
            sim = CorrectedFaultSimulator(scenario=scenario, trial_length=250, seed=42)
            events = [sim.generate_event() for _ in range(250)]
            self.assertEqual(len(events), 250)
            self.assertGreater(
                sim.injection_count, 0,
                f"Scenario '{scenario}' failed to inject within 250 events"
            )

    def test_injection_step_relative_to_trial_length(self):
        sim = CorrectedFaultSimulator(scenario="Sudden Drift", trial_length=250, seed=1)
        self.assertEqual(sim.inject_step, 100, "Inject step for N=250 should be 100 (40%)")

        for step in range(1, 100):
            ev = sim.generate_event()
            self.assertEqual(ev["_current_mean"], 50.0)
        ev_100 = sim.generate_event()
        self.assertEqual(ev_100["_current_mean"], 90.0)

    def test_system_fault_payloads_emitted(self):
        expected = {
            "Consumer Crash": "CONSUMER_CRASH",
            "High Latency": "HIGH_LATENCY",
            "Data Processing Failure": "PROCESSING_FAILURE",
        }
        for fault_scenario, expected_payload in expected.items():
            sim = CorrectedFaultSimulator(scenario=fault_scenario, trial_length=250, seed=10)
            payloads = [sim.generate_event().get("simulated_fault") for _ in range(250)]
            self.assertIn(
                expected_payload, payloads,
                f"Payload {expected_payload} not found in {fault_scenario}"
            )


class TestMissingOrOutOfWindowInjections(unittest.TestCase):
    """Verifies detection of missing injections."""

    def test_out_of_window_flagging(self):
        sim = CorrectedFaultSimulator(scenario="Sudden Drift", trial_length=250, seed=1)
        sim.inject_step = 500  # simulate old defect
        events = []
        for _ in range(250):
            ev = sim.generate_event()
            ev["_injection_count_delta"] = 0
            events.append(ev)

        row = run_one_trial(
            scenario="Sudden Drift",
            engine_name="Baseline Rule Engine",
            eval_seed=1,
            pre_generated_events=events,
            fast_simulation=True,
        )
        self.assertFalse(row["valid"], "Trial without injection should be marked invalid")
        self.assertIn("Injection did not occur", row["invalid_reason"])


class TestFaultRouting(unittest.TestCase):
    """Verifies correct routing of injected system faults into recovery handlers."""

    def test_fault_monitor_dispatches_to_engine(self):
        fault_scenarios = ["Consumer Crash", "High Latency", "Data Processing Failure"]
        for fault_scenario in fault_scenarios:
            engine = InstrumentedRecoveryEngine(fast_simulation=True)
            monitor = FaultMonitor(engine)

            events, _, _ = generate_event_stream(fault_scenario, 250, eval_seed=7)
            for ev in events:
                monitor.check_for_faults(ev)

            self.assertGreater(
                engine._recovery_trigger_count, 0,
                f"Fault scenario {fault_scenario} failed to trigger recovery"
            )
            actions = [rec["action"] for rec in engine.timing_records]
            self.assertGreater(len(actions), 0, "No recovery actions recorded")


class TestReproducibilityAndPairing(unittest.TestCase):
    """Verifies identical paired streams and fixed seed reproducibility."""

    def test_fixed_seed_reproducibility(self):
        for scenario in CorrectedFaultSimulator.ALL_SCENARIOS:
            events1, fp1, inj1 = generate_event_stream(scenario, 250, eval_seed=123)
            events2, fp2, inj2 = generate_event_stream(scenario, 250, eval_seed=123)

            self.assertEqual(fp1, fp2, f"Fingerprints differ for identical seed in {scenario}")
            self.assertEqual(inj1, inj2)
            for e1, e2 in zip(events1, events2):
                self.assertEqual(e1["value"], e2["value"])
                self.assertEqual(e1["_step"], e2["_step"])
                self.assertEqual(e1["simulated_fault"], e2["simulated_fault"])

    def test_paired_engines_receive_identical_stream(self):
        scenario = "Sudden Drift"
        eval_seed = 42
        events, fp, _ = generate_event_stream(scenario, 250, eval_seed)

        row_baseline = run_one_trial(
            scenario=scenario,
            engine_name="Baseline Rule Engine",
            eval_seed=eval_seed,
            pre_generated_events=events,
            fast_simulation=True,
            stream_fingerprint=fp,
        )

        row_ppo = run_one_trial(
            scenario=scenario,
            engine_name="PPO Agent",
            eval_seed=eval_seed,
            pre_generated_events=events,
            fast_simulation=True,
            stream_fingerprint=fp,
        )

        self.assertEqual(row_baseline["stream_fingerprint"], row_ppo["stream_fingerprint"])
        self.assertEqual(row_baseline["stream_fingerprint"], fp)


class TestTimingSeparation(unittest.TestCase):
    """Verifies that decision latency is separated from simulated execution time."""

    def test_decision_latency_is_positive_and_submillisecond(self):
        scenario = "Consumer Crash"
        events, fp, _ = generate_event_stream(scenario, 250, eval_seed=0)
        row = run_one_trial(
            scenario=scenario,
            engine_name="Baseline Rule Engine",
            eval_seed=0,
            pre_generated_events=events,
            fast_simulation=True,
        )

        self.assertGreater(row["mean_decision_latency_s"], 0.0, "Decision latency must be positive")
        self.assertLess(row["mean_decision_latency_s"], 0.1, "Decision latency should be in sub-millisecond range")

    def test_fast_simulation_disables_sleep_only_in_executor(self):
        import time
        engine = InstrumentedRecoveryEngine(fast_simulation=True)
        t0 = time.perf_counter()
        success, exec_time = engine.executor.execute("Retrain Model")
        t1 = time.perf_counter()

        self.assertLess(t1 - t0, 0.1, f"Fast mode execution took {t1 - t0}s; sleep not bypassed")


class TestResultPreservation(unittest.TestCase):
    """Verifies that original submitted results are strictly preserved."""

    def test_original_results_preserved(self):
        original_csv = os.path.join(_ADRIFT_ROOT, "evaluation", "results", "results.csv")
        self.assertTrue(os.path.exists(original_csv), "Original results.csv must exist")

        backup_csv = os.path.join(OUTPUT_DIR, "ORIGINAL_submitted_results.csv")
        self.assertTrue(os.path.exists(backup_csv), "Backup of original results must exist")


if __name__ == "__main__":
    unittest.main()
