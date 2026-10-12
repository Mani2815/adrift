import numpy as np

class StateEncoder:
    """
    Transforms raw pipeline metrics into a normalized continuous state vector 
    for the PPO agent to observe.
    """
    def __init__(self):
        self.state_dim = 8
        self.drift_mapping = {
            "None": 0.0,
            "Sudden Drift": 1.0,
            "Gradual Drift": 2.0,
            "Incremental Drift": 3.0,
            "Recurring Drift": 4.0
        }

    def encode(self, pipeline_state: dict) -> np.ndarray:
        """
        pipeline_state example:
        {
            "drift_type": "Sudden Drift",
            "severity_score": 85.0,
            "latency": 0.45,
            "throughput": 150.0,
            "consumer_health": 0, # 0=Healthy, 1=Failed
            "success_rate": 0.95,
            "drift_frequency": 0.05,
            "error_rate": 0.01
        }
        """
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Feature 0: Drift Type
        drift = pipeline_state.get("drift_type", "None")
        state[0] = self.drift_mapping.get(drift, 0.0)
        
        # Feature 1: Severity Score (normalized 0-1)
        state[1] = pipeline_state.get("severity_score", 0.0) / 100.0
        
        # Feature 2: Latency (capped and scaled roughly)
        state[2] = min(pipeline_state.get("latency", 0.0) / 5.0, 1.0)
        
        # Feature 3: Throughput (normalized against theoretical max e.g. 500)
        state[3] = min(pipeline_state.get("throughput", 0.0) / 500.0, 1.0)
        
        # Feature 4: Consumer Health (0=Healthy, 1=Failed)
        state[4] = float(pipeline_state.get("consumer_health", 0))
        
        # Feature 5: Recent Recovery Success Rate (0-1)
        state[5] = pipeline_state.get("success_rate", 1.0)
        
        # Feature 6: Drift Frequency
        state[6] = min(pipeline_state.get("drift_frequency", 0.0), 1.0)
        
        # Feature 7: Error Rate
        state[7] = min(pipeline_state.get("error_rate", 0.0), 1.0)
        
        return state
