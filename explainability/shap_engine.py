import os
import csv
import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timezone

from config.logging_config import setup_logger
logger = setup_logger("ShapEngine", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "explainability.log"))

try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

class ShapEngine:
    def __init__(self, ppo_inference=None, output_dir=None):
        self.ppo_inference = ppo_inference
        
        base_dir = os.path.dirname(os.path.dirname(__file__))
        self.csv_path = os.path.join(base_dir, "data", "explanations.csv")
        self.fig_dir = output_dir or os.path.join(base_dir, "evaluation", "results", "figures")
        os.makedirs(self.fig_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        self._init_csv()
        
        self.explainer = None
        self.feature_names = [
            "Drift Type", "Severity Score", "Latency", "Throughput",
            "Consumer Health", "Success Rate", "Drift Freq", "Error Rate"
        ]
        
        if SHAP_AVAILABLE and self.ppo_inference and self.ppo_inference.model is not None:
            self._init_explainer()

    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'event_id', 'drift_type', 'severity', 'recovery_action', 'top_feature', 'explanation_text'])

    def _ppo_predict_wrapper(self, states_array):
        # SHAP passes a 2D numpy array of shape (N, 8)
        actions = []
        for s in states_array:
            a, _ = self.ppo_inference.model.predict(s, deterministic=True)
            actions.append(a)
        return np.array(actions)

    def _init_explainer(self):
        logger.info("Initializing SHAP KernelExplainer for PPO Agent...")
        # Create a synthetic background dataset (representing average/neutral states)
        # Assuming 8 features based on state_encoder
        background = np.zeros((10, 8), dtype=np.float32)
        # e.g., median throughput, 1.0 success rate, healthy consumer
        background[:, 3] = 0.5  # throughput
        background[:, 5] = 1.0  # success rate
        
        self.explainer = shap.KernelExplainer(self._ppo_predict_wrapper, background)
        logger.info("KernelExplainer ready.")

    def explain_action(self, pipeline_state, encoded_state, action_str):
        if not SHAP_AVAILABLE or not self.explainer:
            return
            
        event_id = str(uuid.uuid4())[:8]
        
        # Calculate SHAP values for the single encoded state (shape: 1x8)
        shap_values = self.explainer.shap_values(np.array([encoded_state]))
        
        # KernelExplainer can return a list or array depending on output shape
        if isinstance(shap_values, list):
            sv = shap_values[0][0]
        else:
            sv = shap_values[0]
            
        base_value = self.explainer.expected_value
        if isinstance(base_value, np.ndarray) or isinstance(base_value, list):
            base_value = base_value[0]
            
        # Extract top feature
        contributions = {feat: val for feat, val in zip(self.feature_names, sv)}
        sorted_feats = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_feat, top_val = sorted_feats[0]
        
        # Build human-readable text
        explanation = f"Action '{action_str}' was strongly influenced by '{top_feat}' (SHAP value: {top_val:.2f})."
        
        # Log to CSV
        timestamp = datetime.now(timezone.utc).isoformat()
        drift_type = pipeline_state.get("drift_type", "None")
        severity = pipeline_state.get("severity_score", 0.0)
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_id, drift_type, severity, action_str, top_feat, explanation])
            
        # Generate plot for the FIRST 3 events only
        plot_count = len([f for f in os.listdir(self.fig_dir) if f.startswith('shap_waterfall_')])
        if plot_count < 3:
            self._generate_waterfall_plot(base_value, sv, encoded_state, event_id)
            
        return explanation

    def _generate_waterfall_plot(self, base_value, shap_vals, features, event_id):
        plt.figure(figsize=(10, 6))
        
        explanation = shap.Explanation(values=shap_vals, 
                                       base_values=base_value, 
                                       data=features, 
                                       feature_names=self.feature_names)
                                       
        shap.plots.waterfall(explanation, show=False)
        plt.title(f"SHAP Explainer Waterfall - Event {event_id}")
        plt.tight_layout()
        
        save_path = os.path.join(self.fig_dir, f"shap_waterfall_{event_id}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved SHAP waterfall plot to {save_path}")
