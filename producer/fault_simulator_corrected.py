"""
fault_simulator_corrected.py
============================
Camera-Ready Protocol B -- corrected FaultSimulator.

Key fixes vs. the original fault_simulator.py
----------------------------------------------
1. SCENARIO SCHEDULING BUG (Critical)
   Original: All step-based injections (Sudden, Gradual, Recurring, Consumer Crash,
   High Latency, Data Processing Failure) triggered at step >= 500 or step == 500.
   With 250 events per trial, NO step-based injection was ever reached.
   Incremental Drift was the only scenario that produced a real signal.

   Fix: Injection windows are now scaled to the actual trial window.
   Defaults: TRIAL_LENGTH = 250 events; injections at or after step 100
   (40 % of the trial), giving detectors ≥ 150 events of post-injection stream.

2. INJECTION COUNTER
   Each generator exposes `injection_count` (how many fault events were emitted)
   so the harness can verify that the intended injection occurred.

3. STREAM FINGERPRINT
   The simulator records a SHA-256 digest of the (step, value) sequence so
   paired trials can be verified to have received identical input events.

4. SEPARATION OF RANDOM SOURCES
   Event generation uses a *dedicated* `random.Random` instance seeded from
   the trial seed.  This isolates generator randomness from recovery-engine
   randomness (which may consume from the module-level `random` state).

Usage (by corrected experiment runner):
    sim = CorrectedFaultSimulator(scenario="Consumer Crash",
                                  trial_length=250, seed=42)
    for _ in range(trial_length):
        event = sim.generate_event()
        ...
    assert sim.injection_count > 0, "Scenario not exercised"
    fingerprint = sim.get_fingerprint()
"""

import hashlib
import json
import random as _std_random
import uuid
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Injection schedule constants (all relative to a trial_length=250 window)
# ---------------------------------------------------------------------------
_DEFAULT_TRIAL_LENGTH = 250
_INJECTION_START_STEP = 100   # 40 % into the trial


class CorrectedFaultSimulator:
    """
    Drop-in replacement for the original FaultSimulator with corrected
    injection schedules and reproducibility bookkeeping.
    """

    # Maps scenario name -> human description of when/what is injected
    SCENARIO_DESCRIPTIONS = {
        "Sudden Drift":            "Mean shifts abruptly at step INJECT_STEP.",
        "Gradual Drift":           "Mean ramps linearly from INJECT_STEP to INJECT_STEP+100.",
        "Incremental Drift":       "Mean increases by 0.05 on every step (from step 1).",
        "Recurring Drift":         "Mean alternates every RECURRING_INTERVAL steps (starting at 1).",
        "Consumer Crash":          "CONSUMER_CRASH fault payload emitted at step INJECT_STEP.",
        "High Latency":            "HIGH_LATENCY fault payload emitted for 10 steps after INJECT_STEP.",
        "Data Processing Failure": "PROCESSING_FAILURE fault payload emitted at step INJECT_STEP.",
    }

    ALL_SCENARIOS = list(SCENARIO_DESCRIPTIONS.keys())

    def __init__(
        self,
        scenario: str = "Sudden Drift",
        trial_length: int = _DEFAULT_TRIAL_LENGTH,
        seed: Optional[int] = None,
    ):
        if scenario not in self.SCENARIO_DESCRIPTIONS:
            raise ValueError(
                f"Unknown scenario '{scenario}'. "
                f"Valid options: {self.ALL_SCENARIOS}"
            )
        self.scenario = scenario
        self.trial_length = trial_length

        # Injection step: 40 % of trial length, minimum 10
        self.inject_step = max(10, int(trial_length * 0.40))

        # Recurring drift alternates every 20 % of trial length
        self.recurring_interval = max(5, int(trial_length * 0.20))

        # Gradual ramp length: 40 % of trial length
        self.gradual_ramp_length = max(10, int(trial_length * 0.40))

        # Dedicated RNG -- isolated from module-level random state
        self._rng = _std_random.Random(seed)

        # State
        self.step = 0
        self.base_mean = 50.0
        self.current_mean = self.base_mean

        # Bookkeeping
        self.injection_count = 0          # faults/drifts emitted
        self._stream_hash = hashlib.sha256()  # rolling fingerprint

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_event(self) -> dict:
        self.step += 1
        fault_payload = self._compute_fault_payload()
        value = round(self._rng.gauss(self.current_mean, 10.0), 4)

        sources = ["sensor_A", "sensor_B", "network_node_1"]
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_id": str(uuid.uuid4()),
            "value": value,
            "source": self._rng.choice(sources),
            "status": "OK",
            "simulated_fault": fault_payload,
            # Extra bookkeeping fields (stripped downstream if needed)
            "_step": self.step,
            "_current_mean": round(self.current_mean, 4),
        }

        # Update stream fingerprint (step + value only, for determinism)
        self._stream_hash.update(f"{self.step}:{value}".encode())

        return event

    def get_fingerprint(self) -> str:
        """Return SHA-256 hex digest of the (step, value) sequence so far."""
        return self._stream_hash.hexdigest()

    def injection_schedule_summary(self) -> dict:
        """Return a human-readable summary of when injections fire."""
        return {
            "scenario": self.scenario,
            "trial_length": self.trial_length,
            "inject_step": self.inject_step,
            "recurring_interval": self.recurring_interval if self.scenario == "Recurring Drift" else None,
            "gradual_ramp_end": (
                self.inject_step + self.gradual_ramp_length
                if self.scenario == "Gradual Drift" else None
            ),
            "high_latency_end": (
                self.inject_step + 10 if self.scenario == "High Latency" else None
            ),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_fault_payload(self):
        """
        Apply drift/fault logic and return fault_payload string or None.
        This also updates self.current_mean and self.injection_count.
        """
        fault_payload = None

        s = self.step
        inject = self.inject_step

        if self.scenario == "Sudden Drift":
            if s == inject:
                self.current_mean = 90.0
                self.injection_count += 1

        elif self.scenario == "Gradual Drift":
            ramp_end = inject + self.gradual_ramp_length
            if inject <= s <= ramp_end:
                self.current_mean += (90.0 - 50.0) / self.gradual_ramp_length
                if s == inject:
                    self.injection_count += 1  # Count once at ramp start

        elif self.scenario == "Incremental Drift":
            # Starts immediately from step 1 — same semantics as original
            self.current_mean += 0.05
            if s == 1:
                self.injection_count += 1  # Continuous; count once as "started"

        elif self.scenario == "Recurring Drift":
            if s > 0 and s % self.recurring_interval == 0:
                self.current_mean = 90.0 if self.current_mean == self.base_mean else self.base_mean
                self.injection_count += 1

        elif self.scenario == "Consumer Crash":
            if s == inject:
                fault_payload = "CONSUMER_CRASH"
                self.injection_count += 1

        elif self.scenario == "High Latency":
            if inject <= s <= inject + 10:
                fault_payload = "HIGH_LATENCY"
                if s == inject:
                    self.injection_count += 1  # Count once at window start

        elif self.scenario == "Data Processing Failure":
            if s == inject:
                fault_payload = "PROCESSING_FAILURE"
                self.injection_count += 1

        return fault_payload
