import sys
import os
import csv
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor

def evaluate_engine(name, use_ppo=False):
    settings.PPO_ENABLED = use_ppo
    
    scenarios = [
        "Sudden Drift", 
        "Gradual Drift", 
        "Recurring Drift", 
        "Consumer Crash", 
        "High Latency", 
        "Data Processing Failure"
    ]
    
    recovery_engine = RecoveryEngine()
    fault_monitor = FaultMonitor(recovery_engine)
    drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
    
    print(f"\nEvaluating: {name} (PPO_ENABLED={use_ppo})")
    
    for scenario in scenarios:
        simulator = FaultSimulator(scenario=scenario)
        for _ in range(500): # 500 steps per scenario
            event = simulator.generate_event()
            fault_monitor.check_for_faults(event)
            drift_detector.process_value(event["value"])
            
    metrics = recovery_engine.get_metrics()
    return metrics

def run_comparison():
    print("\n" + "="*50)
    print("🚀 Starting Phase 5 Comparative Evaluation")
    print("="*50)
    
    # 1. Evaluate Baseline
    print("\n[Phase 1/2] Testing Baseline Static Rule Engine...")
    baseline_metrics = evaluate_engine("Baseline Rule Engine", use_ppo=False)
    
    # 2. Evaluate PPO
    print("\n[Phase 2/2] Testing PPO Adaptive Agent...")
    ppo_metrics = evaluate_engine("PPO Agent", use_ppo=True)
    
    # 3. Compare and Output
    print("\n--- Final Comparative Results ---")
    print(f"{'Metric':<25} | {'Baseline':<15} | {'PPO Agent':<15}")
    print("-" * 60)
    print(f"{'MTTR (seconds)':<25} | {baseline_metrics['mttr']:<15.2f} | {ppo_metrics['mttr']:<15.2f}")
    print(f"{'Success Rate (%)':<25} | {baseline_metrics['success_rate']*100:<15.1f} | {ppo_metrics['success_rate']*100:<15.1f}")
    print(f"{'Simulated Uptime (%)':<25} | {baseline_metrics['uptime']*100:<15.1f} | {ppo_metrics['uptime']*100:<15.1f}")
    print(f"{'Total Recoveries':<25} | {baseline_metrics['total_recoveries']:<15} | {ppo_metrics['total_recoveries']:<15}")
    
    # Export to CSV
    export_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "ppo_vs_baseline_results.csv")
    with open(export_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Baseline Rule Engine', 'PPO Adaptive Agent'])
        writer.writerow(['MTTR_seconds', round(baseline_metrics['mttr'], 2), round(ppo_metrics['mttr'], 2)])
        writer.writerow(['Success_Rate', round(baseline_metrics['success_rate'], 4), round(ppo_metrics['success_rate'], 4)])
        writer.writerow(['Uptime', round(baseline_metrics['uptime'], 4), round(ppo_metrics['uptime'], 4)])
        writer.writerow(['Total_Recoveries', baseline_metrics['total_recoveries'], ppo_metrics['total_recoveries']])
        
    print(f"\n✅ Results exported to {export_path}")

if __name__ == "__main__":
    run_comparison()
