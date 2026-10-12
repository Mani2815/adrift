import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from explainability.shap_engine import ShapEngine, SHAP_AVAILABLE
from recovery.recovery_engine import RecoveryEngine

def run_global_shap():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    if not SHAP_AVAILABLE:
        print("SHAP is not available. Skipping Global SHAP Analysis.")
        return
        
    engine = RecoveryEngine()
    if not engine.ppo_inference or not engine.ppo_inference.model:
        print("PPO Model not found.")
        return
        
    shap_engine = engine.shap_engine
    if not shap_engine or not shap_engine.explainer:
        print("SHAP explainer not initialized.")
        return

    # Generate a batch of representative states
    # State features: Drift Type, Severity Score, Latency, Throughput, Consumer Health, Success Rate, Drift Freq, Error Rate
    np.random.seed(42)
    num_samples = 200
    
    # Feature 0: Drift Type (0-3)
    f0 = np.random.randint(0, 4, size=num_samples)
    # Feature 1: Severity (0.0 - 100.0)
    f1 = np.random.uniform(0, 100, size=num_samples)
    # Feature 2: Latency (0.0 - 1.0 normalized)
    f2 = np.random.uniform(0, 1, size=num_samples)
    # Feature 3: Throughput (0.0 - 1.0 normalized)
    f3 = np.random.uniform(0, 1, size=num_samples)
    # Feature 4: Consumer Health (0=crash, 1=healthy)
    f4 = np.random.choice([0, 1], size=num_samples, p=[0.1, 0.9])
    # Feature 5: Success Rate (0.0 - 1.0)
    f5 = np.random.uniform(0.5, 1.0, size=num_samples)
    # Feature 6: Drift Freq (normalized)
    f6 = np.random.uniform(0, 1, size=num_samples)
    # Feature 7: Error Rate
    f7 = np.random.uniform(0, 0.2, size=num_samples)
    
    background_data = np.column_stack([f0, f1, f2, f3, f4, f5, f6, f7])
    
    print(f"Computing SHAP values for {num_samples} background states...")
    shap_values = shap_engine.explainer.shap_values(background_data)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
        
    # Aggregate absolute SHAP values for global feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    importance_df = pd.DataFrame({
        "Feature": shap_engine.feature_names,
        "Mean Absolute SHAP Value": mean_abs_shap
    }).sort_values(by="Mean Absolute SHAP Value", ascending=False)
    
    csv_path = os.path.join(results_dir, "global_shap_importance.csv")
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved global SHAP values to {csv_path}")
    
    # Generate Plot
    fig_dir = os.path.join(results_dir, "ieee_figures")
    os.makedirs(fig_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Mean Absolute SHAP Value", 
        y="Feature", 
        data=importance_df, 
        palette="rocket"
    )
    plt.title("Global Feature Importance (SHAP)")
    plt.xlabel("Mean |SHAP Value| (Impact on Model Output)")
    plt.ylabel("")
    plt.tight_layout()
    
    plt.savefig(os.path.join(fig_dir, "global_shap.png"), dpi=300)
    plt.close()
    
    print("✅ Global SHAP analysis complete.")

if __name__ == "__main__":
    run_global_shap()
