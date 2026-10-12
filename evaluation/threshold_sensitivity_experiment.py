import sys
import os
import random
import numpy as np
import time
import pandas as pd

# Monkeypatch sleep for instant fast-forward execution (same as Phase 1 / 2A)
time.sleep = lambda x: None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor
from recovery.rule_engine import RuleEngine

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results", "phase2b2_threshold_sensitivity")
os.makedirs(OUTPUT_DIR, exist_ok=True)
RAW_CSV_PATH = os.path.join(OUTPUT_DIR, "raw_results.csv")

class TunableRuleEngine(RuleEngine):
    def __init__(self, sudden_threshold, gradual_threshold, default_threshold):
        self.sudden_threshold = sudden_threshold
        self.gradual_threshold = gradual_threshold
        self.default_threshold = default_threshold
        
    def select_action_for_drift(self, drift_type, severity_score):
        if drift_type == "Sudden Drift" and severity_score > self.sudden_threshold:
            return "Retrain Model"
        if drift_type == "Gradual Drift" and severity_score < self.gradual_threshold:
            return "Ignore Event"
        if drift_type == "Recurring Drift":
            return "Rollback Previous Model"
        return "Retrain Model" if severity_score > self.default_threshold else "Ignore Event"

class TunableRecoveryEngine(RecoveryEngine):
    def __init__(self, sudden, gradual, default):
        super().__init__()
        self.rule_engine = TunableRuleEngine(sudden, gradual, default)
        self.recorded_actions = []
        self.csv_path = "/dev/null" # Prevent writing to the baseline csv
        settings.PPO_ENABLED = False # strictly rule engine

    def handle_drift_event(self, drift_type, severity_score):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return
            
        t_trigger = time.perf_counter()
        
        action = self.rule_engine.select_action_for_drift(drift_type, severity_score)
        t_decision = time.perf_counter()
        
        success, exec_time = self.executor.execute(action)
        t_action = time.perf_counter()
        
        self.metrics.record_recovery(action, exec_time, success)
        
        self.recorded_actions.append({
            "drift_type": drift_type,
            "severity": severity_score,
            "action": action,
            "success": success,
            "decision_latency": t_decision - t_trigger,
            "end_to_end_latency": t_action - t_trigger
        })

    def handle_fault(self, fault_type):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return
            
        t_trigger = time.perf_counter()
        
        action = self.rule_engine.select_action_for_fault(fault_type)
        t_decision = time.perf_counter()
        
        success, exec_time = self.executor.execute(action)
        t_action = time.perf_counter()
        
        self.metrics.record_recovery(action, exec_time, success)
        
        self.recorded_actions.append({
            "drift_type": fault_type,
            "severity": 100.0,
            "action": action,
            "success": success,
            "decision_latency": t_decision - t_trigger,
            "end_to_end_latency": t_action - t_trigger
        })

def run_experiment():
    scenarios = [
        "Sudden Drift", "Gradual Drift", "Incremental Drift", "Recurring Drift", 
        "Consumer Crash", "High Latency", "Data Processing Failure"
    ]
    
    # (sweep_parameter, threshold_value, sudden, gradual, default)
    configs = [
        ("Sudden Sweep", 60, 60, 50, 60),
        ("Sudden Sweep", 65, 65, 50, 60),
        ("Sudden Sweep", 70, 70, 50, 60), # baseline
        ("Sudden Sweep", 75, 75, 50, 60),
        ("Sudden Sweep", 80, 80, 50, 60),
        
        ("Gradual Sweep", 40, 70, 40, 60),
        ("Gradual Sweep", 45, 70, 45, 60),
        ("Gradual Sweep", 50, 70, 50, 60), # baseline
        ("Gradual Sweep", 55, 70, 55, 60),
        ("Gradual Sweep", 60, 70, 60, 60),
        
        ("Default Sweep", 50, 70, 50, 50),
        ("Default Sweep", 55, 70, 50, 55),
        ("Default Sweep", 60, 70, 50, 60), # baseline
        ("Default Sweep", 65, 70, 50, 65),
        ("Default Sweep", 70, 70, 50, 70),
    ]

    all_records = []

    for sweep_parameter, threshold_value, sudden, gradual, default in configs:
        print(f"Running config: {sweep_parameter} = {threshold_value}")
        
        for scenario in scenarios:
            for trial in range(10):
                random.seed(trial)
                np.random.seed(trial)
                
                recovery_engine = TunableRecoveryEngine(sudden, gradual, default)
                fault_monitor = FaultMonitor(recovery_engine)
                drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
                simulator = FaultSimulator(scenario=scenario)
                
                start_time = time.time()
                events_processed = 0
                
                for _ in range(250):
                    event = simulator.generate_event()
                    fault_monitor.check_for_faults(event)
                    drift_detector.process_value(event["value"])
                    events_processed += 1
                    
                end_time = time.time()
                duration = max(end_time - start_time, 0.001)
                
                metrics = recovery_engine.get_metrics()
                uptime = metrics.get("uptime", 0) * 100
                throughput = events_processed / duration
                
                # Expand actions
                for action_data in recovery_engine.recorded_actions:
                    all_records.append({
                        "sweep_parameter": sweep_parameter,
                        "threshold_value": threshold_value,
                        "scenario": scenario,
                        "seed": trial,
                        "trial_id": f"{scenario}_{trial}",
                        "engine": "Rule Engine",
                        "action": action_data["action"],
                        "success": action_data["success"],
                        "decision_latency": action_data["decision_latency"],
                        "computational_decision_latency": action_data["decision_latency"],
                        "end_to_end_recovery_time": action_data["end_to_end_latency"],
                        "throughput": throughput,
                        "uptime": uptime,
                        "severity": action_data["severity"],
                        "drift_type": action_data["drift_type"]
                    })
                    
    df = pd.DataFrame(all_records)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"✅ Generated {len(df)} records at {RAW_CSV_PATH}")

if __name__ == "__main__":
    run_experiment()
