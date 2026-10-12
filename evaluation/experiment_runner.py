import sys
import os
import random
import numpy as np
import time

# Monkeypatch sleep for instant fast-forward execution
time.sleep = lambda x: None

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor
from evaluation.result_logger import ResultLogger

def run_experiments():
    scenarios = [
        "Sudden Drift", "Gradual Drift", "Incremental Drift", "Recurring Drift", 
        "Consumer Crash", "High Latency", "Data Processing Failure"
    ]
    systems = ["Baseline Rule Engine", "PPO Agent"]
    
    logger = ResultLogger()
    
    print("🚀 Starting ADRIFT Research Evaluation Framework (Phase 6)")
    print("=" * 60)
    
    exp_id = 1
    
    try:
        from evaluation.report_generator import ReportGenerator
        from evaluation.plots import generate_all_plots
    except ImportError:
        pass
    
    for system in systems:
        settings.PPO_ENABLED = (system == "PPO Agent")
        print(f"\n--- Initializing System: {system} ---")
        
        for scenario in scenarios:
            for trial in range(10):
                print(f"Executing: {system} | {scenario} | Trial {trial+1}/10")
                
                random.seed(trial)
                np.random.seed(trial)
                
                recovery_engine = RecoveryEngine()
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
                
                mttr = metrics.get("mttr", 0)
                success_rate = metrics.get("success_rate", 0) * 100
                uptime = metrics.get("uptime", 0) * 100
                    
                throughput = events_processed / duration
                latency = duration / events_processed * 1000 # in ms
                
                logger.log_result(exp_id, scenario, system, mttr, success_rate, uptime, latency, throughput)
                exp_id += 1
                
    print("\n✅ All 140 trials completed.")
    print("Generating Reports and Visualizations...")
    
    try:
        ReportGenerator().generate_report()
        generate_all_plots()
        print("\n🎉 Evaluation complete! Check the evaluation/results/ directory.")
    except Exception as e:
        print(f"\n⚠️ Data logged, but report generation pending. ({e})")

if __name__ == "__main__":
    run_experiments()
