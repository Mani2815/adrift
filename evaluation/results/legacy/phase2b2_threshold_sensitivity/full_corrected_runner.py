import sys
import os
import random
import numpy as np
import time
import pandas as pd
import shutil
import sqlite3

time.sleep = lambda x: None

# Root paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)
SENSITIVITY_DIR = os.path.dirname(os.path.abspath(__file__))

# Data dependencies
import data.history_db as history_db
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from config.settings import settings
from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor
from recovery.rule_engine import RuleEngine
from authoritative_success_aggregation import calculate_authoritative_success

def setup_isolated_state(config_id):
    """Creates a completely isolated state directory and redirects DB for the configuration."""
    state_dir = os.path.join(SENSITIVITY_DIR, "state", config_id)
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    os.makedirs(state_dir, exist_ok=True)
    
    db_path = os.path.join(state_dir, 'drift_history.db')
    history_db.DB_PATH = db_path
    history_db.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
    history_db.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=history_db.engine)
    history_db.init_history_db()
    
    return state_dir

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
    def __init__(self, sudden, gradual, default, state_dir):
        super().__init__()
        self.rule_engine = TunableRuleEngine(sudden, gradual, default)
        self.recorded_actions = []
        self.csv_path = os.path.join(state_dir, "recovery_history.csv")
        self._init_csv()
        settings.PPO_ENABLED = False

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

def run_configuration(sweep_parameter, threshold_value, sudden, gradual, default, config_id):
    state_dir = setup_isolated_state(config_id)
    scenarios = [
        "Sudden Drift", "Gradual Drift", "Incremental Drift", "Recurring Drift", 
        "Consumer Crash", "High Latency", "Data Processing Failure"
    ]
    
    records = []
    
    for scenario in scenarios:
        for trial in range(10):
            random.seed(trial)
            np.random.seed(trial)
            
            recovery_engine = TunableRecoveryEngine(sudden, gradual, default, state_dir)
            fault_monitor = FaultMonitor(recovery_engine)
            drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
            
            drift_detector.csv_path = os.path.join(state_dir, "drift_events.csv")
            drift_detector._init_csv()
            
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
            
            for action_data in recovery_engine.recorded_actions:
                records.append({
                    "configuration_id": config_id,
                    "sweep_parameter": sweep_parameter,
                    "threshold_value": threshold_value,
                    "sudden_threshold": sudden,
                    "gradual_threshold": gradual,
                    "default_threshold": default,
                    "scenario": scenario,
                    "seed": trial,
                    "trial_id": f"{scenario}_{trial}",
                    "action": action_data["action"],
                    "success": action_data["success"],
                    "decision_latency": action_data["decision_latency"],
                    "computational_decision_latency": action_data["decision_latency"],
                    "end_to_end_latency": action_data["end_to_end_latency"],
                    "throughput": throughput,
                    "uptime": uptime,
                    "severity": action_data["severity"],
                    "drift_type": action_data["drift_type"]
                })
    return pd.DataFrame(records)

def main():
    configs = [
        # Sweep A - Sudden
        ("Sudden Sweep", 60, 60, 50, 60),
        ("Sudden Sweep", 65, 65, 50, 60),
        ("Sudden Sweep", 70, 70, 50, 60),
        ("Sudden Sweep", 75, 75, 50, 60),
        ("Sudden Sweep", 80, 80, 50, 60),
        
        # Sweep B - Gradual
        ("Gradual Sweep", 40, 70, 40, 60),
        ("Gradual Sweep", 45, 70, 45, 60),
        ("Gradual Sweep", 50, 70, 50, 60),
        ("Gradual Sweep", 55, 70, 55, 60),
        ("Gradual Sweep", 60, 70, 60, 60),
        
        # Sweep C - Default
        ("Default Sweep", 50, 70, 50, 50),
        ("Default Sweep", 55, 70, 50, 55),
        ("Default Sweep", 60, 70, 50, 60),
        ("Default Sweep", 65, 70, 50, 65),
        ("Default Sweep", 70, 70, 50, 70),
    ]

    # Hard Baseline Gate
    print("Running Baseline Gate (70, 50, 60)...")
    baseline_df = run_configuration("Baseline Gate", 0, 70, 50, 60, "baseline_gate")
    auth_success = calculate_authoritative_success(baseline_df)
    print(f"Baseline Gate Success: {auth_success:.2f}%")
    
    if abs(auth_success - 97.17) > 0.01:
        print("BASELINE GATE FAILED. STOPPING.")
        sys.exit(1)
        
    print("Baseline Gate PASSED. Proceeding with full sweep...")

    all_raw_dfs = []
    summary_records = []
    
    for idx, (sweep_param, t_val, sudden, gradual, default) in enumerate(configs):
        config_id = f"config_{idx+1}_{sweep_param.replace(' ', '')}_{t_val}"
        print(f"Executing: {config_id}")
        
        df = run_configuration(sweep_param, t_val, sudden, gradual, default, config_id)
        all_raw_dfs.append(df)
        
        auth_succ = calculate_authoritative_success(df)
        event_succ = (df['success'].sum() / len(df) * 100) if not df.empty else 100.0
        
        # Action policy stats
        actions = df['action'].value_counts().to_dict()
        
        summary_records.append({
            "configuration_id": config_id,
            "sweep_parameter": sweep_param,
            "threshold_value": t_val,
            "n_trials": 70,
            "n_actionable_events": len(df),
            "successful_events": int(df['success'].sum()),
            "authoritative_trial_success": auth_succ,
            "event_weighted_success_diagnostic": event_succ,
            "throughput": df['throughput'].mean(),
            "uptime": df['uptime'].mean(),
            "retrain_count": actions.get("Retrain Model", 0),
            "ignore_count": actions.get("Ignore Event", 0),
            "rollback_count": actions.get("Rollback Previous Model", 0),
        })

    final_raw_df = pd.concat(all_raw_dfs, ignore_index=True)
    final_raw_df.to_csv(os.path.join(SENSITIVITY_DIR, "raw_results_corrected.csv"), index=False)
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(os.path.join(SENSITIVITY_DIR, "summary_stats_corrected.csv"), index=False)
    
    print("Full sweep completed. Output saved.")

if __name__ == "__main__":
    main()
