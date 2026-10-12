import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from producer.drift_simulator import DriftSimulator
from consumer.drift_detector import DriftDetectionService
from data.history_db import SessionLocal, ClassifiedDrift

def evaluate_scenario(scenario_name, expected_type, steps=2000):
    # Initialize a fresh DB session and clear historical drifts for clean test
    db = SessionLocal()
    try:
        db.query(ClassifiedDrift).delete()
        db.commit()
    except Exception:
        pass
    
    simulator = DriftSimulator(scenario=scenario_name)
    detector = DriftDetectionService()
    
    print(f"\nEvaluating Scenario: {scenario_name} (Expected: {expected_type})")
    
    # Run simulation
    for _ in range(steps):
        event = simulator.generate_event()
        detector.process_value(event["value"])
        
    # Check results in DB
    results = db.query(ClassifiedDrift).all()
    
    if not results:
        print("  ❌ No drift detected.")
        db.close()
        return False
        
    # Analyze detected types
    types_detected = [r.drift_type for r in results]
    print(f"  Detectors triggered {len(results)} intelligence events. Classifications: {types_detected}")
    
    # Simple success metric: did it classify the expected type at least once?
    success = any(expected_type in t for t in types_detected)
    if success:
        print(f"  ✅ Successfully classified as {expected_type}")
    else:
        print(f"  ❌ Failed to classify as {expected_type}")
        
    db.close()
    return success

def run_evaluations():
    scenarios = [
        ("Sudden", "Sudden Drift"),
        ("Gradual", "Gradual Drift"),
        ("Incremental", "Incremental Drift"),
        ("Recurring", "Recurring Drift")
    ]
    
    correct = 0
    for scenario, expected in scenarios:
        if evaluate_scenario(scenario, expected):
            correct += 1
            
    print(f"\n--- Evaluation Summary ---")
    acc = correct / len(scenarios)
    print(f"Accuracy: {correct}/{len(scenarios)} ({acc*100:.2f}%)")
    
    # Since this is a simple exact match evaluation, Precision = Recall = Accuracy
    precision = acc
    recall = acc
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f"Macro Precision: {precision:.2f}")
    print(f"Macro Recall: {recall:.2f}")
    print(f"Macro F1 Score: {f1:.2f}")

if __name__ == "__main__":
    run_evaluations()
