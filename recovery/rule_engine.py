class RuleEngine:
    """
    Static Rule-Based logic for selecting a recovery action.
    This module acts as the baseline that a future PPO agent will replace.
    """
    def select_action_for_drift(self, drift_type, severity_score):
        # Rule 1
        if drift_type == "Sudden Drift" and severity_score > 70:
            return "Retrain Model"
            
        # Rule 2
        if drift_type == "Gradual Drift" and severity_score < 50:
            return "Ignore Event"
            
        # Rule 3
        if drift_type == "Recurring Drift":
            return "Rollback Previous Model"
            
        # Default fallback
        return "Retrain Model" if severity_score > 60 else "Ignore Event"
        
    def select_action_for_fault(self, fault_type):
        # Rule 4
        if fault_type == "Consumer Failure":
            return "Restart Consumer"
            
        # Rule 5
        if fault_type == "High Latency":
            return "Scale Processing Resources"
            
        if fault_type == "Data Processing Failure":
            return "Restart Consumer"
            
        return "Ignore Event"
