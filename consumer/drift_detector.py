import os
import csv
import math
from datetime import datetime, timezone

try:
    from river import drift
except ImportError:
    pass # Will be handled by the runner

from config.settings import settings
from config.logging_config import setup_logger

# Phase 3 imports
from data.history_db import init_history_db, SessionLocal, ClassifiedDrift
from consumer.drift_classifier import DriftClassifier
from consumer.severity_engine import SeverityEngine
from consumer.severity_engine import SeverityEngine

logger = setup_logger("DriftDetector", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "drift_classification.log"))

class DriftDetectionService:
    def __init__(self, recovery_engine=None):
        self.recovery_engine = recovery_engine
        self.csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "drift_events.csv")
        self._init_csv()
        
        # Initialize Phase 2 detectors
        self.adwin = drift.ADWIN() if settings.ADWIN_ENABLED else None
        self.ddm = drift.binary.DDM() if settings.DDM_ENABLED and hasattr(drift, 'binary') else None
        self.kswin = drift.KSWIN() if settings.KSWIN_ENABLED else None
        
        # State for DDM binarization and Phase 3 Heuristics
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
        self.last_drift_step = 0
        self.pre_drift_mean = 50.0 # Baseline
        
        # Phase 3 Intelligence Engines
        init_history_db()
        self.classifier = DriftClassifier(memory_enabled=settings.RECURRING_DRIFT_MEMORY_ENABLED) if settings.DRIFT_CLASSIFICATION_ENABLED else None
        self.severity_engine = SeverityEngine() if settings.SEVERITY_SCORING_ENABLED else None
        
        # Phase 6 Explainability Engine moved to RecoveryEngine
        
        logger.info("Drift Detection & Intelligence Service Initialized.")
        
    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'detector', 'drift_status', 'warning_status', 'drift_score'])

    def _log_event(self, detector_name, drift_status, warning_status, score):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, detector_name, drift_status, warning_status, score])

    def _trigger_intelligence(self, triggered_detectors):
        """Phase 3: Classify and score the drift, then persist it."""
        # Avoid spamming classification for the same drift event (cooldown of 50 steps)
        steps_since_last = self.count - self.last_drift_step
        if steps_since_last < 50 and self.last_drift_step != 0:
            return
            
        post_drift_mean = self.mean
        magnitude = post_drift_mean - self.pre_drift_mean
        std_dev = self._get_std()
        
        self.last_drift_step = self.count
        self.pre_drift_mean = post_drift_mean

        if not self.classifier or not self.severity_engine:
            return
        
        # 1. Classify Drift Type
        drift_type, confidence = self.classifier.classify_drift(
            self.pre_drift_mean, post_drift_mean, steps_since_last, magnitude
        )
        
        # 2. Calculate Severity
        score, level = self.severity_engine.calculate_severity(
            magnitude_shift=magnitude,
            std_dev=std_dev,
            detectors_count=len(triggered_detectors),
            steps_since_last_drift=steps_since_last if self.last_drift_step != 0 else 9999
        )
        
        # 3. Log to Intelligence Logger
        logger.warning(f"--- INTELLIGENCE TRIGGERED ---")
        logger.warning(f"Drift Type: {drift_type} (Confidence: {confidence:.2f})")
        logger.warning(f"Severity: {level} (Score: {score})")
        logger.warning(f"Shift: {self.pre_drift_mean:.2f} -> {post_drift_mean:.2f} (Detectors: {','.join(triggered_detectors)})")
        
        # 4. Save to DB
        db = SessionLocal()
        try:
            intel_record = ClassifiedDrift(
                drift_type=drift_type,
                confidence=confidence,
                severity_score=score,
                severity_level=level,
                detectors_triggered=",".join(triggered_detectors),
                magnitude_shift=magnitude
            )
            db.add(intel_record)
            db.commit()
        except Exception as e:
            logger.error(f"Failed to save drift intelligence: {e}")
        finally:
            db.close()
            
        # 5. Phase 4 Auto Recovery
        recovery_action = "None"
        if self.recovery_engine:
            self.recovery_engine.handle_drift_event(drift_type, score)
            # In a real impl, handle_drift_event should return the action, but we'll mock or read it later.
            # For explainability, we assume it happened.
            recovery_action = "Action Taken"
            
        # 6. Phase 6 SHAP Explainability handled by RecoveryEngine

    def _update_stats(self, value):
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2
        
    def _get_std(self):
        if self.count < 2:
            return 0.0
        return math.sqrt(self.m2 / (self.count - 1))

    def process_value(self, value):
        if not settings.DRIFT_DETECTOR_ENABLED:
            return
            
        self._update_stats(value)
        std_dev = self._get_std()
        
        triggered_detectors = []
        
        # 1. ADWIN
        if self.adwin:
            self.adwin.update(value)
            if self.adwin.drift_detected:
                self._log_event("ADWIN", True, False, value)
                triggered_detectors.append("ADWIN")
                
        # 2. DDM
        if self.ddm:
            is_error = 0
            if self.count > 30 and std_dev > 0:
                if abs(value - self.mean) > 2 * std_dev:
                    is_error = 1
            self.ddm.update(is_error)
            
            if self.ddm.drift_detected:
                self._log_event("DDM", True, False, is_error)
                triggered_detectors.append("DDM")
            elif self.ddm.warning_detected:
                self._log_event("DDM", False, True, is_error)
                
        # 3. KSWIN
        if self.kswin:
            self.kswin.update(value)
            if self.kswin.drift_detected:
                self._log_event("KSWIN", True, False, value)
                triggered_detectors.append("KSWIN")
                
        # Phase 3 Intelligence Trigger
        if len(triggered_detectors) > 0:
            self._trigger_intelligence(triggered_detectors)
