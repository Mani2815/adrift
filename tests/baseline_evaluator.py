import sys
import os
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService
from recovery.recovery_engine import RecoveryEngine
from recovery.fault_monitor import FaultMonitor

def evaluate_baseline():
    scenarios = [
        "Sudden Drift", 
        "Gradual Drift", 
        "Incremental Drift", 
        "Recurring Drift", 
        "Consumer Crash", 
        "High Latency", 
        "Data Processing Failure"
    ]
    
    # Initialize intelligence and recovery engines
    recovery_engine = RecoveryEngine()
    fault_monitor = FaultMonitor(recovery_engine)
    drift_detector = DriftDetectionService(recovery_engine=recovery_engine)
    
    print("\n" + "="*50)
    print("🚀 Starting Baseline Auto-Recovery Evaluation (Phase 4)")
    print("="*50)
    
    for scenario in scenarios:
        print(f"Injecting Scenario: {scenario}...")
        simulator = FaultSimulator(scenario=scenario)
        
        # Simulate 1000 steps per scenario to give detectors time to catch and recover
        for _ in range(1000):
            event = simulator.generate_event()
            
            # Phase 4 System Fault Intercept
            fault_monitor.check_for_faults(event)
            
            # Phase 2 & 3 Concept Drift Detection
            drift_detector.process_value(event["value"])
            
    # Extract Final Operational Metrics
    metrics = recovery_engine.get_metrics()
    
    print("\n--- Baseline Evaluation Final Results ---")
    print(f"Total Recovery Attempts: {metrics['total_recoveries']}")
    print(f"Mean Time To Recovery (MTTR): {metrics['mttr']:.2f} seconds")
    print(f"Recovery Success Rate: {metrics['success_rate']*100:.1f}%")
    print(f"Simulated Pipeline Uptime: {metrics['uptime']*100:.2f}%")
    
    print("\nRecovery Actions Invoked:")
    for action, count in metrics['action_distribution'].items():
        if count > 0:
            print(f"  - {action}: {count}")
        
    # Export to CSV for Phase 5 comparison
    export_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "baseline_evaluation_results.csv")
    with open(export_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Total Recoveries', metrics['total_recoveries']])
        writer.writerow(['MTTR_seconds', round(metrics['mttr'], 2)])
        writer.writerow(['Success_Rate', round(metrics['success_rate'], 4)])
        writer.writerow(['Uptime', round(metrics['uptime'], 4)])
        for action, count in metrics['action_distribution'].items():
            writer.writerow([f'Action_{action.replace(" ", "_")}', count])
            
    print(f"\n✅ Results exported to {export_path}")
    print("These metrics represent the baseline. In Phase 5, the PPO Agent must outperform this MTTR and Success Rate.")

if __name__ == "__main__":
    evaluate_baseline()
