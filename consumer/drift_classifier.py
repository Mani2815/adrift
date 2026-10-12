from data.history_db import SessionLocal, ClassifiedDrift

class DriftClassifier:
    """
    Analyzes temporal and statistical patterns to classify the drift 
    into Sudden, Gradual, Incremental, or Recurring.
    """
    
    def __init__(self, memory_enabled=True):
        self.memory_enabled = memory_enabled
        self.historical_means = []
        self._load_memory()
        
    def _load_memory(self):
        if not self.memory_enabled:
            return
            
        db = SessionLocal()
        try:
            records = db.query(ClassifiedDrift).order_by(ClassifiedDrift.id.asc()).all()
            for r in records:
                # Approximate historical means by tracking the delta shifts
                # For a full implementation, we'd store exact pre/post means in DB.
                # Here we simulate historical knowledge.
                pass
        finally:
            db.close()

    def classify_drift(self, pre_drift_mean, post_drift_mean, steps_since_last, magnitude):
        is_recurring = False
        
        # Recurring Check: Does the new mean look like a previously known mean?
        if self.memory_enabled and len(self.historical_means) > 0:
            # Compare against all but the immediate previous state
            for hist_mean in self.historical_means[:-1]:
                # If within 5% of historical state
                if abs(post_drift_mean - hist_mean) < (0.05 * abs(hist_mean) + 1e-5):
                    is_recurring = True
                    break
        
        drift_type = "Sudden Drift"
        confidence = 0.85
        
        if is_recurring:
            drift_type = "Recurring Drift"
            confidence = 0.92
        else:
            # Classification based on rate of change
            rate_of_change = abs(magnitude) / max(steps_since_last, 1)
            
            if rate_of_change > 0.5 or (abs(magnitude) > 15 and steps_since_last < 200):
                drift_type = "Sudden Drift"
                confidence = 0.95
            elif rate_of_change > 0.05 or (abs(magnitude) > 5 and steps_since_last < 1000):
                drift_type = "Gradual Drift"
                confidence = 0.82
            else:
                drift_type = "Incremental Drift"
                confidence = 0.78
        
        # Update Memory
        if self.memory_enabled:
            self.historical_means.append(post_drift_mean)
            if len(self.historical_means) > 50:
                self.historical_means.pop(0)
                
        return drift_type, confidence
