class SeverityEngine:
    """
    Calculates the severity of a concept drift based on magnitude, 
    detector agreement, and frequency.
    """
    
    def calculate_severity(self, magnitude_shift, std_dev, detectors_count, steps_since_last_drift):
        """
        Returns a severity score (0-100) and severity level string.
        """
        score = 0.0
        
        # 1. Magnitude Factor (Max 50 points)
        # Z-score represents how many standard deviations the mean shifted.
        if std_dev > 0:
            z_score = abs(magnitude_shift) / std_dev
            mag_score = min(z_score * 10, 50)
        else:
            mag_score = 10 if abs(magnitude_shift) > 0 else 0
            
        score += mag_score
        
        # 2. Detector Consensus (Max 30 points)
        # If multiple algorithms detect it, it's highly likely to be a real, severe drift.
        consensus_score = min(detectors_count * 10, 30)
        score += consensus_score
        
        # 3. Frequency / Rapidity (Max 20 points)
        # Drifts happening too close to each other indicate high instability.
        if steps_since_last_drift is None or steps_since_last_drift > 1000:
            freq_score = 0
        else:
            freq_score = max(0, 20 - (steps_since_last_drift * 20 / 1000))
            
        score += freq_score
        
        # Bound score between 0 and 100
        score = min(max(score, 0), 100)
        
        # Determine Level
        if score <= 25:
            level = "Low"
        elif score <= 50:
            level = "Moderate"
        elif score <= 75:
            level = "High"
        else:
            level = "Critical"
            
        return round(score, 2), level
