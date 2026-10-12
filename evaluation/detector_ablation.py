import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import settings
from producer.fault_simulator import FaultSimulator
from consumer.drift_detector import DriftDetectionService

def run_ablation():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    modes = [
        {"name": "ADWIN Only", "adwin": True, "ddm": False, "kswin": False},
        {"name": "DDM Only", "adwin": False, "ddm": True, "kswin": False},
        {"name": "KSWIN Only", "adwin": False, "ddm": False, "kswin": True},
        {"name": "Fusion", "adwin": True, "ddm": True, "kswin": True},
    ]
    
    ablation_results = []
    
    # Disable downstream DB logging for speed
    settings.DRIFT_CLASSIFICATION_ENABLED = False
    
    for mode in modes:
        settings.ADWIN_ENABLED = mode["adwin"]
        settings.DDM_ENABLED = mode["ddm"]
        settings.KSWIN_ENABLED = mode["kswin"]
        
        sim = FaultSimulator(scenario="Sudden Drift")
        detector = DriftDetectionService()
        
        detected_steps = []
        
        # Inject drift at step 500. Run 1000 steps.
        for _ in range(1000):
            event = sim.generate_event()
            detector.process_value(event["value"])
            
            # Since _trigger_intelligence happens inside, we just check if last_drift_step updated
            if detector.last_drift_step == detector.count:
                detected_steps.append(detector.count)
                
        # Metrics Calculation
        # True drift occurs at step 500.
        # Detection delay: first detection after 500
        valid_detections = [s for s in detected_steps if s >= 500]
        false_alarms = len([s for s in detected_steps if s < 500])
        
        if valid_detections:
            delay = valid_detections[0] - 500
            detection_rate = 1.0 # 100% since it was detected
        else:
            delay = 0
            detection_rate = 0.0
            
        far = false_alarms / 500.0 # False Alarm Rate per pre-drift step
        
        ablation_results.append({
            "Mode": mode["name"],
            "Detection Rate": detection_rate,
            "False Alarm Rate": far,
            "Detection Delay": delay
        })
        
    df = pd.DataFrame(ablation_results)
    csv_path = os.path.join(results_dir, "detector_ablation.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.barplot(x="Mode", y="Detection Delay", data=df, ax=axes[0], palette="viridis")
    axes[0].set_title("Detection Delay by Detector")
    axes[0].set_ylabel("Events Elapsed")
    
    sns.barplot(x="Mode", y="False Alarm Rate", data=df, ax=axes[1], palette="magma")
    axes[1].set_title("False Alarm Rate by Detector")
    axes[1].set_ylabel("Rate")
    
    plt.tight_layout()
    fig_dir = os.path.join(results_dir, "ieee_figures")
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(os.path.join(fig_dir, "ablation_comparison.png"), dpi=300)
    plt.close()
    
    # Generate Table
    tex_path = os.path.join(results_dir, "ieee_tables_ablation.tex")
    tex_content = """\\begin{table}[h]
\\caption{Detector Ablation Study}
\\label{tab:detector_ablation}
\\centering
\\begin{tabular}{@{}lccc@{}}
\\toprule
\\textbf{Mode} & \\textbf{Detection Rate} & \\textbf{False Alarm Rate} & \\textbf{Detection Delay} \\\\ \\midrule
"""
    for _, row in df.iterrows():
        tex_content += f"{row['Mode']} & {row['Detection Rate']*100:.1f}\\% & {row['False Alarm Rate']:.4f} & {row['Detection Delay']} \\\\\n"
    
    tex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    with open(tex_path, 'w') as f:
        f.write(tex_content)
        
    print("✅ Detector ablation complete.")

if __name__ == "__main__":
    run_ablation()
