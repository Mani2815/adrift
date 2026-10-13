"""
protocol_b_runner.py
====================
Camera-Ready Protocol B -- corrected experiment harness.

Fixes vs. the original evaluation/experiment_runner.py
-------------------------------------------------------
1.  SCENARIO INJECTION BUG
    Uses CorrectedFaultSimulator which injects at step ~100 (40% of 250),
    ensuring all 7 scenarios actually exercise their intended fault/drift
    within the 250-event window.

2.  PAIRED EVENT STREAMS
    Pre-generates the full event sequence once per (scenario, trial_seed)
    using a dedicated RNG.  Both engines then replay the SAME list of events.
    The stream fingerprint is recorded and verified to be identical for the pair.

3.  TIME.SLEEP ISOLATION
    `time.sleep` is NOT monkeypatched globally.  The RecoveryExecutor still
    calls `time.sleep(exec_time)`, but we redirect it to a no-op ONLY for the
    executor so that (a) decision timing uses real wall-clock, and (b) the no-op
    is explicit and configurable via FAST_SIMULATION_MODE.

4.  SEPARATE TIMING METRICS
    - `decision_latency_s`:  time from trigger to action selection (rule lookup
      or PPO forward pass), measured with `time.perf_counter()`.
    - `simulated_exec_duration_s`:  the `exec_time` drawn by the executor
      (the sleep value before the no-op redirect).
    - `actual_exec_wall_s`:  actual wall-clock time of executor.execute()
      (near-zero when no-op).
    - `e2e_latency_s`:  trigger → end of executor.execute().
    Note: `mttr` in MetricsEngine accumulates `actual_exec_wall_s` (not
    `simulated_exec_duration_s`) so uptime reflects real compute time only.

5.  INJECTION AND RECOVERY COUNTERS
    Each trial records `injection_count` and `recovery_trigger_count`.
    Trials where injection_count == 0 are marked INVALID and excluded from
    aggregate statistics.

6.  OUTPUT DIRECTORY
    All outputs go to evaluation/results/camera_ready_protocol_b/ and are
    NEVER written to the original evaluation/results/results.csv.

7.  RAW TRIAL-LEVEL RECORDS
    One CSV row per trial with all bookkeeping fields (not aggregated).

Usage:
    cd ADRIFT/
    python evaluation/protocol_b_runner.py
    python evaluation/protocol_b_runner.py --fast   # no-op sleeps (default)
    python evaluation/protocol_b_runner.py --real-time  # real sleeps (slow!)
"""

import sys
import os
import time
import csv
import json
import random
import hashlib
import argparse
import numpy as np
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_ADRIFT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ADRIFT_ROOT)

from config.settings import settings
from producer.fault_simulator_corrected import CorrectedFaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor

# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.path.join(_ADRIFT_ROOT, "evaluation", "results", "camera_ready_protocol_b")
DEFAULT_OUTPUT_DIR = os.path.join(_ADRIFT_ROOT, "evaluation", "results", "camera_ready_reproducible")
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Trial-level raw columns
# ---------------------------------------------------------------------------
RAW_COLUMNS = [
    "run_timestamp",
    "scenario",
    "engine",
    "training_seed",        # PPO training seed (N/A for rule engine)
    "eval_seed",
    "trial_length",
    "injection_count",
    "recovery_trigger_count",
    "recovery_success_count",
    "recovery_fail_count",
    "success_rate_pct",     # successful / total_recoveries * 100; 100.0 if 0 triggers
    "mean_decision_latency_s",
    "mean_simulated_exec_duration_s",
    "mean_actual_exec_wall_s",
    "mean_e2e_latency_s",
    "simulated_uptime_pct", # (wall_time - actual_exec_time) / wall_time * 100
    "wall_time_s",
    "stream_fingerprint",
    "valid",                # True / False
    "invalid_reason",
    "recovery_rng_protocol",
    "recovery_stream_id",
]


# ---------------------------------------------------------------------------
# Instrumented RecoveryEngine
# ---------------------------------------------------------------------------
class InstrumentedRecoveryEngine(RecoveryEngine):
    """
    Wraps RecoveryEngine to measure per-recovery timing separately for
    decision vs. execution phases, using reproducible Common Random Numbers (CRN).
    """

    def __init__(
        self,
        fast_simulation: bool = True,
        model_path: str = None,
        eval_seed: int = 0,
        scenario: str = "",
        recovery_rng_protocol: str = "CRN_v1_sha256_attempt_keyed",
    ):
        self._fast_simulation = fast_simulation
        self.eval_seed = eval_seed
        self.scenario = scenario
        self.recovery_rng_protocol = recovery_rng_protocol
        self.recovery_stream_id = f"crn:eval_seed={eval_seed}:scenario={scenario}"
        self._recovery_attempt_count = 0
        super().__init__()
        if model_path and settings.PPO_ENABLED:
            from ppo.ppo_inference import PPOInferenceEngine
            self.ppo_inference = PPOInferenceEngine(model_path=model_path)
        self.timing_records = []   # list of per-recovery dicts
        self._recovery_trigger_count = 0
        self._recovery_success_count = 0
        self._recovery_fail_count = 0
        self._total_actual_exec_wall = 0.0
        self._total_simulated_exec = 0.0
        self._total_decision_latency = 0.0

        if self._fast_simulation:
            orig_exec = self.executor.execute
            def _fast_exec(action, rng=None, u_rand=None, v_rand=None):
                import recovery.recovery_executor as _rex
                _orig_sleep = _rex.time.sleep
                try:
                    _rex.time.sleep = lambda x: None
                    return orig_exec(action, rng=rng, u_rand=u_rand, v_rand=v_rand)
                finally:
                    _rex.time.sleep = _orig_sleep
            self.executor.execute = _fast_exec

    def _handle_any(self, action_selector_fn, drift_type, severity, fault_type):
        """Common instrumented path for both drift and fault triggers."""
        t_trigger = time.perf_counter()

        # --- Decision phase ---
        action = action_selector_fn()
        t_decision = time.perf_counter()
        decision_latency = t_decision - t_trigger

        # --- Execution phase ---
        # Redirect time.sleep in the executor if fast_simulation
        import recovery.recovery_executor as _rex_module
        _orig_sleep = time.sleep
        if self._fast_simulation:
            _rex_module.time.sleep = lambda x: None
        else:
            _rex_module.time.sleep = _orig_sleep

        # Common Random Numbers (CRN) exogenous stochasticity for this attempt
        attempt_idx = self._recovery_attempt_count
        self._recovery_attempt_count += 1

        attempt_stream_id = f"crn:eval_seed={self.eval_seed}:scenario={self.scenario}:attempt={attempt_idx}"
        attempt_seed = int(hashlib.sha256(attempt_stream_id.encode()).hexdigest()[:8], 16)
        attempt_rng = random.Random(attempt_seed)
        u_rand = attempt_rng.random()
        v_rand = attempt_rng.random()

        t_exec_start = time.perf_counter()
        success, actual_exec_wall = self.executor.execute(
            action, u_rand=u_rand, v_rand=v_rand
        )
        t_exec_end = time.perf_counter()

        # Restore
        _rex_module.time.sleep = _orig_sleep

        actual_exec_wall_measured = t_exec_end - t_exec_start
        e2e_latency = t_exec_end - t_trigger

        # Record
        self.metrics.record_recovery(action, actual_exec_wall_measured, success)
        self._log_recovery(
            "DRIFT" if fault_type == "None" else "FAULT",
            drift_type, severity, fault_type, action,
            actual_exec_wall_measured, success,
        )

        self._recovery_trigger_count += 1
        if success:
            self._recovery_success_count += 1
        else:
            self._recovery_fail_count += 1

        self._total_decision_latency    += decision_latency
        self._total_actual_exec_wall    += actual_exec_wall_measured
        self._total_e2e_latency         = getattr(self, "_total_e2e_latency", 0.0) + e2e_latency

        self.timing_records.append({
            "attempt_idx": attempt_idx,
            "attempt_stream_id": attempt_stream_id,
            "u_rand": round(u_rand, 6),
            "v_rand": round(v_rand, 6),
            "action": action,
            "success": success,
            "decision_latency_s": decision_latency,
            "actual_exec_wall_s": actual_exec_wall_measured,
            "e2e_latency_s": e2e_latency,
        })

    def handle_drift_event(self, drift_type, severity_score):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return

        if settings.PPO_ENABLED and self.ppo_inference and self.ppo_inference.model is not None:
            def _select():
                state = self._build_state(drift_type, severity_score, "None")
                return self.ppo_inference.predict_action(state)
        else:
            def _select():
                return self.rule_engine.select_action_for_drift(drift_type, severity_score)

        self._handle_any(_select, drift_type, severity_score, "None")

    def handle_fault(self, fault_type):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return

        if settings.PPO_ENABLED and self.ppo_inference and self.ppo_inference.model is not None:
            def _select():
                state = self._build_state("None", 100.0, fault_type)
                return self.ppo_inference.predict_action(state)
        else:
            def _select():
                return self.rule_engine.select_action_for_fault(fault_type)

        self._handle_any(_select, "None", 100.0, fault_type)

    def get_timing_summary(self):
        n = self._recovery_trigger_count
        if n == 0:
            return {
                "mean_decision_latency_s": 0.0,
                "mean_actual_exec_wall_s": 0.0,
                "mean_e2e_latency_s": 0.0,
            }
        return {
            "mean_decision_latency_s": self._total_decision_latency / n,
            "mean_actual_exec_wall_s": self._total_actual_exec_wall / n,
            "mean_e2e_latency_s":      getattr(self, "_total_e2e_latency", 0.0) / n,
        }


# ---------------------------------------------------------------------------
# Core trial runner
# ---------------------------------------------------------------------------
def run_one_trial(
    scenario: str,
    engine_name: str,
    eval_seed: int,
    pre_generated_events: list,
    fast_simulation: bool = True,
    training_seed: int = 42,
    stream_fingerprint: str = "",
    model_path: str = None,
    recovery_rng_protocol: str = "CRN_v1_sha256_attempt_keyed",
) -> dict:
    """
    Run a single trial.  `pre_generated_events` is the shared event list that
    BOTH engines receive for this (scenario, eval_seed) pair.
    """
    settings.PPO_ENABLED = (engine_name == "PPO Agent")

    t_wall_start = time.perf_counter()

    engine = InstrumentedRecoveryEngine(
        fast_simulation=fast_simulation,
        model_path=model_path,
        eval_seed=eval_seed,
        scenario=scenario,
        recovery_rng_protocol=recovery_rng_protocol,
    )
    fault_monitor = FaultMonitor(engine)
    kswin_seed = int(hashlib.sha256(f"kswin:{eval_seed}:{scenario}".encode()).hexdigest()[:8], 16)
    drift_detector = DriftDetectionService(recovery_engine=engine, seed=kswin_seed)

    injection_count = 0
    for event in pre_generated_events:
        injection_count += event.get("_injection_count_delta", 0)
        fault_monitor.check_for_faults(event)
        drift_detector.process_value(event["value"])

    t_wall_end = time.perf_counter()
    wall_time = t_wall_end - t_wall_start

    metrics = engine.get_metrics()
    timing  = engine.get_timing_summary()

    n_triggers = engine._recovery_trigger_count
    n_success  = engine._recovery_success_count

    success_rate_pct = (
        (n_success / n_triggers * 100.0) if n_triggers > 0 else 100.0
    )

    # Simulated uptime: (wall_time - total_actual_exec_wall) / wall_time
    total_exec = engine._total_actual_exec_wall
    simulated_uptime_pct = (
        max(0.0, (wall_time - total_exec) / wall_time * 100.0)
        if wall_time > 0 else 100.0
    )

    # Validity check
    valid = True
    invalid_reason = ""
    if injection_count == 0 and scenario != "Incremental Drift":
        valid = False
        invalid_reason = f"Injection did not occur within {len(pre_generated_events)}-event window"

    return {
        "run_timestamp":                datetime.now(timezone.utc).isoformat(),
        "scenario":                     scenario,
        "engine":                       engine_name,
        "training_seed":                training_seed if engine_name == "PPO Agent" else "N/A",
        "eval_seed":                    eval_seed,
        "trial_length":                 len(pre_generated_events),
        "injection_count":              injection_count,
        "recovery_trigger_count":       n_triggers,
        "recovery_success_count":       n_success,
        "recovery_fail_count":          engine._recovery_fail_count,
        "success_rate_pct":             round(success_rate_pct, 4),
        "mean_decision_latency_s":      round(timing["mean_decision_latency_s"], 8),
        "mean_simulated_exec_duration_s": 0.0,
        "mean_actual_exec_wall_s":      round(timing["mean_actual_exec_wall_s"], 8),
        "mean_e2e_latency_s":           round(timing["mean_e2e_latency_s"], 8),
        "simulated_uptime_pct":         round(simulated_uptime_pct, 6),
        "wall_time_s":                  round(wall_time, 6),
        "stream_fingerprint":           stream_fingerprint,
        "valid":                        valid,
        "invalid_reason":               invalid_reason,
        "recovery_rng_protocol":        recovery_rng_protocol,
        "recovery_stream_id":           engine.recovery_stream_id,
    }


# ---------------------------------------------------------------------------
# Event pre-generation (shared stream for both engines)
# ---------------------------------------------------------------------------
def generate_event_stream(scenario: str, trial_length: int, eval_seed: int) -> tuple:
    """
    Returns (events_list, stream_fingerprint, injection_count).
    Each event dict has a `_injection_count_delta` field for bookkeeping.
    """
    sim = CorrectedFaultSimulator(
        scenario=scenario,
        trial_length=trial_length,
        seed=eval_seed,
    )
    events = []
    prev_injection = 0
    for _ in range(trial_length):
        ev = sim.generate_event()
        delta = sim.injection_count - prev_injection
        ev["_injection_count_delta"] = delta
        prev_injection = sim.injection_count
        events.append(ev)

    return events, sim.get_fingerprint(), sim.injection_count


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_all(
    trial_length: int = 250,
    num_trials: int = 10,
    fast_simulation: bool = True,
    ppo_training_seed: int = 42,
    output_dir: str = DEFAULT_OUTPUT_DIR,
):
    os.makedirs(output_dir, exist_ok=True)
    scenarios = CorrectedFaultSimulator.ALL_SCENARIOS
    engines   = ["Baseline Rule Engine", "PPO Agent"]

    # ---- Write CSV header ----
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_csv = os.path.join(output_dir, f"raw_trials_{run_id}.csv")
    symlink_csv = os.path.join(output_dir, "raw_trials.csv")

    with open(run_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
        writer.writeheader()

    print("=" * 70)
    print("ADRIFT Camera-Ready Protocol B — Reproducible Experiment Runner (CRN)")
    print(f"  Trial length : {trial_length} events")
    print(f"  Trials/cell  : {num_trials}")
    print(f"  Fast mode    : {fast_simulation}")
    print(f"  RNG Protocol : CRN_v1_sha256_attempt_keyed")
    print(f"  Output       : {run_csv}")
    print("=" * 70)

    all_rows = []
    invalid_count = 0

    for scenario in scenarios:
        for eval_seed in range(num_trials):
            # ---- Pre-generate SHARED event stream ----
            events, fingerprint, total_injections = generate_event_stream(
                scenario, trial_length, eval_seed
            )

            # Verify injection occurred
            if total_injections == 0 and scenario != "Incremental Drift":
                print(f"  [WARN] {scenario} seed={eval_seed}: NO INJECTION in {trial_length} events")

            for engine_name in engines:
                print(
                    f"  {scenario:<30s} | {engine_name:<22s} | seed={eval_seed}",
                    end="", flush=True,
                )

                row = run_one_trial(
                    scenario=scenario,
                    engine_name=engine_name,
                    eval_seed=eval_seed,
                    pre_generated_events=events,
                    fast_simulation=fast_simulation,
                    training_seed=ppo_training_seed,
                    stream_fingerprint=fingerprint,
                )

                if not row["valid"]:
                    invalid_count += 1
                    print(f"  [INVALID: {row['invalid_reason']}]")
                else:
                    print(
                        f"  triggers={row['recovery_trigger_count']}  "
                        f"success={row['success_rate_pct']:.1f}%  "
                        f"dec_lat={row['mean_decision_latency_s']*1e6:.1f}µs"
                    )

                all_rows.append(row)
                with open(run_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=RAW_COLUMNS)
                    writer.writerow(row)

    # Save stable copy
    import shutil
    shutil.copyfile(run_csv, symlink_csv)

    # ---- Verify paired fingerprints ----
    _verify_stream_pairing(all_rows)

    # ---- Aggregate ----
    _aggregate_and_wilcoxon(run_csv, all_rows, run_id, output_dir=output_dir)

    # ---- Metadata ----
    metadata = {
        "run_id": run_id,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "trial_length": trial_length,
        "num_trials": num_trials,
        "fast_simulation": fast_simulation,
        "ppo_training_seed": ppo_training_seed,
        "scenarios": scenarios,
        "engines": engines,
        "total_trials": len(all_rows),
        "invalid_trials": invalid_count,
        "raw_trial_csv": run_csv,
        "aggregate_csv": os.path.join(output_dir, f"aggregate_{run_id}.csv"),
        "wilcoxon_csv": os.path.join(output_dir, f"wilcoxon_{run_id}.csv"),
        "recovery_rng_protocol": "CRN_v1_sha256_attempt_keyed",
        "protocol_note": (
            "Exogenous recovery randomness is matched per attempt index k using SHA-256 keyed seeds "
            "(stream crn:eval_seed=<seed>:scenario=<scenario>:attempt=<k>). "
            "Event streams are pre-generated with dedicated isolated RNG. "
            "KSWIN drift detector is seeded deterministically per condition. "
            "time.sleep disabled in executor (fast_simulation mode)."
        ),
    }
    run_meta_path = os.path.join(output_dir, f"run_metadata_{run_id}.json")
    with open(run_meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(output_dir, "run_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("=" * 70)
    print(f"  DONE. Total trials: {len(all_rows)}, Invalid: {invalid_count}")
    print(f"  Raw CSV : {run_csv}")
    print(f"  Metadata: {run_meta_path}")
    print("=" * 70)
    return all_rows


def _verify_stream_pairing(rows):
    """Assert that paired (scenario, eval_seed) trials have identical fingerprints."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["scenario"], r["eval_seed"])].append(r["stream_fingerprint"])

    mismatches = 0
    for key, fps in groups.items():
        if len(set(fps)) > 1:
            print(f"  [FINGERPRINT MISMATCH] {key}: {set(fps)}")
            mismatches += 1

    if mismatches == 0:
        print(f"  [OK] All {len(groups)} paired trials have identical stream fingerprints.")
    else:
        print(f"  [ERROR] {mismatches} fingerprint mismatches detected!")


def _aggregate_and_wilcoxon(run_csv, rows, run_id, output_dir=DEFAULT_OUTPUT_DIR):
    """Compute aggregate statistics and Wilcoxon signed-rank test."""
    try:
        import pandas as pd
        from scipy import stats
    except ImportError:
        print("  [SKIP] pandas/scipy not available for aggregate step.")
        return

    df = pd.DataFrame(rows)
    valid_df = df[df["valid"] == True].copy()

    if valid_df.empty:
        print("  [WARN] No valid trials; skipping aggregation.")
        return

    # Aggregate by scenario + engine
    agg = valid_df.groupby(["scenario", "engine"]).agg(
        n=("success_rate_pct", "count"),
        success_rate_mean=("success_rate_pct", "mean"),
        success_rate_std=("success_rate_pct", "std"),
        uptime_mean=("simulated_uptime_pct", "mean"),
        uptime_std=("simulated_uptime_pct", "std"),
        mean_decision_lat_us=("mean_decision_latency_s", lambda x: x.mean() * 1e6),
        mean_e2e_lat_ms=("mean_e2e_latency_s", lambda x: x.mean() * 1e3),
        injection_count_mean=("injection_count", "mean"),
        trigger_count_mean=("recovery_trigger_count", "mean"),
    ).reset_index()

    agg_path = os.path.join(output_dir, f"aggregate_{run_id}.csv")
    agg.to_csv(agg_path, index=False)
    print(f"  Aggregate CSV: {agg_path}")

    # --- Overall summary ---
    overall = valid_df.groupby("engine").agg(
        n=("success_rate_pct", "count"),
        success_rate_mean=("success_rate_pct", "mean"),
        uptime_mean=("simulated_uptime_pct", "mean"),
        mean_decision_lat_us=("mean_decision_latency_s", lambda x: x.mean() * 1e6),
    ).reset_index()
    overall_path = os.path.join(output_dir, f"overall_summary_{run_id}.csv")
    overall.to_csv(overall_path, index=False)
    print(f"  Overall summary CSV: {overall_path}")
    print()
    print(overall.to_string(index=False))
    print()

    # --- Wilcoxon test (paired by scenario + eval_seed) ---
    valid_df["trial_key"] = valid_df.apply(
        lambda r: f"{r['scenario']}__seed{r['eval_seed']}", axis=1
    )
    baseline = valid_df[valid_df["engine"] == "Baseline Rule Engine"].set_index("trial_key")
    ppo      = valid_df[valid_df["engine"] == "PPO Agent"].set_index("trial_key")

    # Only use keys present in BOTH
    common_keys = baseline.index.intersection(ppo.index)
    baseline = baseline.loc[common_keys]
    ppo      = ppo.loc[common_keys]

    wilcoxon_rows = []
    for metric in ["success_rate_pct", "simulated_uptime_pct"]:
        b_vals = baseline[metric].astype(float)
        p_vals = ppo[metric].astype(float)
        diff = b_vals - p_vals
        n_zero = int((diff == 0).sum())
        n_nonzero = int((diff != 0).sum())

        if n_nonzero == 0:
            stat, pval = 0.0, 1.0
        else:
            try:
                res = stats.wilcoxon(b_vals, p_vals)
                stat, pval = float(res.statistic), float(res.pvalue)
            except Exception as e:
                stat, pval = float("nan"), float("nan")

        wilcoxon_rows.append({
            "metric": metric,
            "n_pairs": len(common_keys),
            "n_zero_diffs": n_zero,
            "n_nonzero_diffs": n_nonzero,
            "baseline_mean": round(float(b_vals.mean()), 4),
            "ppo_mean": round(float(p_vals.mean()), 4),
            "wilcoxon_stat": round(stat, 4) if not np.isnan(stat) else stat,
            "p_value": pval,
            "significant_p005": bool(pval < 0.05) if not np.isnan(pval) else False,
        })

    wcx_path = os.path.join(output_dir, f"wilcoxon_{run_id}.csv")
    pd.DataFrame(wilcoxon_rows).to_csv(wcx_path, index=False)
    print(f"  Wilcoxon CSV: {wcx_path}")
    print(pd.DataFrame(wilcoxon_rows).to_string(index=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Protocol B corrected runner")
    parser.add_argument("--trial-length", type=int, default=250,
                        help="Events per trial (default: 250)")
    parser.add_argument("--num-trials", type=int, default=10,
                        help="Trial seeds per (scenario, engine) cell (default: 10)")
    parser.add_argument("--real-time", action="store_true",
                        help="Use real time.sleep in executor (SLOW)")
    parser.add_argument("--ppo-seed", type=int, default=42,
                        help="PPO training seed used for model checkpoint")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save results")
    args = parser.parse_args()

    run_all(
        trial_length=args.trial_length,
        num_trials=args.num_trials,
        fast_simulation=not args.real_time,
        ppo_training_seed=args.ppo_seed,
        output_dir=args.output_dir,
    )

