import sys
import os
import random
import numpy as np
import time
import csv
import argparse
from datetime import datetime, timezone
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor

# DO NOT monkeypatch time.sleep here!

# Create output dir
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "phase2b1_real_latency")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "raw_results.csv")

class RealTimingRecoveryEngine(RecoveryEngine):
    def __init__(self, system, scenario, seed):
        super().__init__()
        self.system = system
        self.scenario = scenario
        self.seed = seed
        self.timing_records = []
        
    def handle_drift_event(self, drift_type, severity_score):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return
            
        t_trigger = time.perf_counter()
        dt_trigger = datetime.now(timezone.utc).isoformat()
        
        # 1. Select Action (Decision)
        if settings.PPO_ENABLED and self.ppo_inference and self.ppo_inference.model is not None:
            state = self._build_state(drift_type, severity_score, "None")
            action = self.ppo_inference.predict_action(state)
        else:
            action = self.rule_engine.select_action_for_drift(drift_type, severity_score)
            
        t_decision = time.perf_counter()
        dt_decision = datetime.now(timezone.utc).isoformat()
        
        # 2. Execute Action
        t_action_start = time.perf_counter()
        dt_action_start = datetime.now(timezone.utc).isoformat()
        
        success, exec_time = self.executor.execute(action)
        
        t_action_end = time.perf_counter()
        dt_action_end = datetime.now(timezone.utc).isoformat()
        
        # 3. Record Metrics and Log
        self.metrics.record_recovery(action, exec_time, success)
        
        decision_latency = t_decision - t_trigger
        action_latency = t_action_end - t_action_start
        e2e_latency = t_action_end - t_trigger
        
        self.timing_records.append({
            "timestamp_trigger": dt_trigger,
            "timestamp_decision": dt_decision,
            "timestamp_action_start": dt_action_start,
            "timestamp_action_end": dt_action_end,
            "decision_latency_seconds": decision_latency,
            "action_execution_latency_seconds": action_latency,
            "end_to_end_recovery_seconds": e2e_latency,
            "engine": self.system,
            "scenario": self.scenario,
            "seed": self.seed,
            "action": action,
            "success": success
        })

    def handle_fault(self, fault_type):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return
            
        t_trigger = time.perf_counter()
        dt_trigger = datetime.now(timezone.utc).isoformat()
        
        # 1. Select Action (Decision)
        if settings.PPO_ENABLED and self.ppo_inference and self.ppo_inference.model is not None:
            state = self._build_state("None", 100.0, fault_type) # faults are max severity
            action = self.ppo_inference.predict_action(state)
        else:
            action = self.rule_engine.select_action_for_fault(fault_type)
            
        t_decision = time.perf_counter()
        dt_decision = datetime.now(timezone.utc).isoformat()
        
        # 2. Execute Action
        t_action_start = time.perf_counter()
        dt_action_start = datetime.now(timezone.utc).isoformat()
        
        success, exec_time = self.executor.execute(action)
        
        t_action_end = time.perf_counter()
        dt_action_end = datetime.now(timezone.utc).isoformat()
        
        # 3. Record Metrics and Log
        self.metrics.record_recovery(action, exec_time, success)
        
        decision_latency = t_decision - t_trigger
        action_latency = t_action_end - t_action_start
        e2e_latency = t_action_end - t_trigger
        
        self.timing_records.append({
            "timestamp_trigger": dt_trigger,
            "timestamp_decision": dt_decision,
            "timestamp_action_start": dt_action_start,
            "timestamp_action_end": dt_action_end,
            "decision_latency_seconds": decision_latency,
            "action_execution_latency_seconds": action_latency,
            "end_to_end_recovery_seconds": e2e_latency,
            "engine": self.system,
            "scenario": self.scenario,
            "seed": self.seed,
            "action": action,
            "success": success
        })


def run_experiments(sanity_check=False):
    scenarios = [
        "Sudden Drift", "Gradual Drift", "Incremental Drift", "Recurring Drift", 
        "Consumer Crash", "High Latency", "Data Processing Failure"
    ]
    systems = ["Baseline Rule Engine", "PPO Agent"]
    
    if sanity_check:
        scenarios = ["Sudden Drift", "Consumer Crash"]
        num_trials = 1
        events_per_trial = 250
        print("Running SANITY CHECK mode...")
        # Clear existing sanity records if any
        if os.path.exists(RAW_CSV_PATH):
            os.remove(RAW_CSV_PATH)
    else:
        num_trials = 10
        events_per_trial = 250
        print("Running FULL EXPERIMENT mode...")
    
    # Init CSV
    if not os.path.exists(RAW_CSV_PATH):
        pd.DataFrame(columns=[
            "timestamp_trigger", "timestamp_decision", "timestamp_action_start", 
            "timestamp_action_end", "decision_latency_seconds", "action_execution_latency_seconds", 
            "end_to_end_recovery_seconds", "engine", "scenario", "seed", "action", "success"
        ]).to_csv(RAW_CSV_PATH, index=False)
        
    for system in systems:
        settings.PPO_ENABLED = (system == "PPO Agent")
        
        for scenario in scenarios:
            for trial in range(num_trials):
                print(f"Executing: {system} | {scenario} | Trial {trial}")
                
                random.seed(trial)
                np.random.seed(trial)
                
                recovery_engine = RealTimingRecoveryEngine(system, scenario, trial)
                fault_monitor = FaultMonitor(recovery_engine)
                drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
                simulator = FaultSimulator(scenario=scenario)
                
                for _ in range(events_per_trial):
                    event = simulator.generate_event()
                    fault_monitor.check_for_faults(event)
                    drift_detector.process_value(event["value"])
                
                # Append records for this trial
                if recovery_engine.timing_records:
                    df = pd.DataFrame(recovery_engine.timing_records)
                    df.to_csv(RAW_CSV_PATH, mode='a', header=False, index=False)
                    
    print("\n✅ Run completed.")

def analyze_results():
    if not os.path.exists(RAW_CSV_PATH):
        return
        
    df = pd.read_csv(RAW_CSV_PATH)
    if df.empty:
        return
        
    summary = df.groupby(['engine', 'scenario', 'action']).agg(
        n=('end_to_end_recovery_seconds', 'count'),
        decision_mean=('decision_latency_seconds', 'mean'),
        decision_std=('decision_latency_seconds', 'std'),
        action_mean=('action_execution_latency_seconds', 'mean'),
        action_std=('action_execution_latency_seconds', 'std'),
        e2e_mean=('end_to_end_recovery_seconds', 'mean'),
        e2e_median=('end_to_end_recovery_seconds', 'median'),
        e2e_std=('end_to_end_recovery_seconds', 'std'),
        e2e_min=('end_to_end_recovery_seconds', 'min'),
        e2e_max=('end_to_end_recovery_seconds', 'max')
    ).reset_index()
    
    # Calculate 95% CI for E2E
    # CI = 1.96 * std / sqrt(n)
    summary['e2e_95_ci'] = 1.96 * summary['e2e_std'] / np.sqrt(summary['n'])
    
    summary_path = os.path.join(OUTPUT_DIR, "summary_stats.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary stats to {summary_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sanity-check', action='store_true')
    args = parser.parse_args()
    
    run_experiments(sanity_check=args.sanity_check)
    analyze_results()
