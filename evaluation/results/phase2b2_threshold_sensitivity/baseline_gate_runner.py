import sys
import os
import random
import numpy as np
import time
import pandas as pd
import shutil
import sqlite3

# Monkeypatch sleep for instant fast-forward execution
time.sleep = lambda x: None

# Root paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(PROJECT_ROOT)
SENSITIVITY_DIR = os.path.dirname(os.path.abspath(__file__))
STATE_DIR = os.path.join(SENSITIVITY_DIR, "state", "baseline")

# --- PERFECT STATE ISOLATION ---
# Clean the state directory completely before running
if os.path.exists(STATE_DIR):
    shutil.rmtree(STATE_DIR)
os.makedirs(STATE_DIR, exist_ok=True)

db_path = os.path.join(STATE_DIR, 'drift_history.db')

# Override module globals before importing anything that uses them
import data.history_db as history_db
history_db.DB_PATH = db_path

# Force SQLAlchemy to re-create the engine and session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
history_db.engine = create_engine(f"sqlite:///{db_path}", connect_args={"check_same_thread": False})
history_db.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=history_db.engine)

from config.settings import settings
from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor
from recovery.rule_engine import RuleEngine
from authoritative_success_aggregation import calculate_authoritative_success

# Ensure database tables exist
history_db.init_history_db()

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
        self.csv_path = os.path.join(STATE_DIR, "recovery_history.csv")
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

def run_baseline_gate():
    scenarios = [
        "Sudden Drift", "Gradual Drift", "Incremental Drift", "Recurring Drift", 
        "Consumer Crash", "High Latency", "Data Processing Failure"
    ]
    
    sudden, gradual, default = 70, 50, 60
    
    print("Running baseline config with PERFECT STATE ISOLATION...")
    
    all_records = []
    
    for scenario in scenarios:
        for trial in range(10):
            # Seed exactly like the authoritative benchmark
            random.seed(trial)
            np.random.seed(trial)
            
            recovery_engine = TunableRecoveryEngine(sudden, gradual, default)
            fault_monitor = FaultMonitor(recovery_engine)
            drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
            
            # Point drift detector CSV to state dir to avoid global leakage
            drift_detector.csv_path = os.path.join(STATE_DIR, "drift_events.csv")
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
                all_records.append({
                    "scenario": scenario,
                    "seed": trial,
                    "trial_id": f"{scenario}_{trial}",
                    "action": action_data["action"],
                    "success": action_data["success"]
                })

    df = pd.DataFrame(all_records)
    
    # Calculate success
    auth_success = calculate_authoritative_success(df)
    print(f"Authoritative Aggregation Success: {auth_success:.2f}%")
    
    if abs(auth_success - 97.17) < 0.01:
        print("BASELINE GATE: PASS")
    else:
        print("BASELINE GATE: FAIL")

if __name__ == "__main__":
    run_baseline_gate()
