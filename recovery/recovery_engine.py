import os
import csv
from datetime import datetime, timezone
from config.settings import settings
from config.logging_config import setup_logger
from recovery.rule_engine import RuleEngine
from recovery.recovery_executor import RecoveryExecutor
from recovery.metrics_engine import MetricsEngine
from ppo.ppo_inference import PPOInferenceEngine

logger = setup_logger("RecoveryEngine", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "recovery_engine.log"))

class RecoveryEngine:
    """
    Central Orchestrator for Phase 4.
    Listens to drift events and system faults, queries the Rule Engine,
    commands the Executor, and tracks Metrics.
    Designed so that `RuleEngine` can be seamlessly swapped for a `PPOAgent` later.
    """
    def __init__(self):
        # The Decision Logic (Phase 4: Rules, Phase 5: PPO)
        self.rule_engine = RuleEngine()
        self.ppo_inference = PPOInferenceEngine() if settings.PPO_ENABLED else None
        
        self.executor = RecoveryExecutor()
        self.metrics = MetricsEngine()
        
        # Initialize SHAP engine linking to PPO inference
        from explainability.shap_engine import ShapEngine
        self.shap_engine = ShapEngine(ppo_inference=self.ppo_inference)

        
        self.csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recovery_history.csv")
        self._init_csv()
        logger.info("Recovery Engine Initialized.")
        
    def _init_csv(self):
        if not os.path.exists(self.csv_path):
            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'trigger_type', 'drift_type', 'severity', 'fault_type', 'recovery_action', 'execution_time', 'success'])

    def _log_recovery(self, trigger_type, drift_type, severity, fault_type, action, exec_time, success):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, trigger_type, drift_type, severity, fault_type, action, round(exec_time,2), success])

    def _build_state(self, drift_type, severity, fault_type):
        """Constructs the dictionary expected by PPO state encoder based on the current trigger."""
        state = {
            "drift_type": drift_type,
            "severity_score": severity,
            "consumer_health": 1 if fault_type == "Consumer Failure" else 0,
            "latency": 5.0 if fault_type == "High Latency" else 0.2,
            "throughput": 0.0 if fault_type == "Consumer Failure" else 450.0,
            "success_rate": self.metrics.successful_recoveries / max(1, self.metrics.total_recoveries),
            "drift_frequency": min(1.0, self.metrics.total_recoveries / 100.0),
            "error_rate": 0.5 if fault_type == "Data Processing Failure" else 0.01
        }
        return state

    def handle_drift_event(self, drift_type, severity_score):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return
            
        logger.info(f"Received DRIFT Trigger: {drift_type} (Severity: {severity_score})")
        
        # 1. Select Action
        if settings.PPO_ENABLED and self.ppo_inference and self.ppo_inference.model is not None:
            logger.info("🤖 Using PPO Agent for Recovery Decision...")
            state = self._build_state(drift_type, severity_score, "None")
            action = self.ppo_inference.predict_action(state)
            
            # SHAP Explanation
            encoded_state = self.ppo_inference.state_encoder.encode(state)
            self.shap_engine.explain_action(state, encoded_state, action)
        else:
            logger.info("⚙️ Using Baseline Rule Engine for Recovery Decision...")
            action = self.rule_engine.select_action_for_drift(drift_type, severity_score)
            
        logger.info(f"Selected Action: {action}")
        
        # 2. Execute Action
        success, exec_time = self.executor.execute(action)
        
        # 3. Record Metrics and Log
        self.metrics.record_recovery(action, exec_time, success)
        self._log_recovery("DRIFT", drift_type, severity_score, "None", action, exec_time, success)

    def handle_fault(self, fault_type):
        if not settings.RECOVERY_ENGINE_ENABLED or not settings.AUTO_RECOVERY_ENABLED:
            return
            
        logger.warning(f"Received FAULT Trigger: {fault_type}")
        
        # 1. Select Action
        if settings.PPO_ENABLED and self.ppo_inference and self.ppo_inference.model is not None:
            logger.info("🤖 Using PPO Agent for Recovery Decision...")
            state = self._build_state("None", 100.0, fault_type) # faults are max severity
            action = self.ppo_inference.predict_action(state)
            
            # SHAP Explanation
            encoded_state = self.ppo_inference.state_encoder.encode(state)
            self.shap_engine.explain_action(state, encoded_state, action)
        else:
            logger.info("⚙️ Using Baseline Rule Engine for Recovery Decision...")
            action = self.rule_engine.select_action_for_fault(fault_type)
            
        logger.info(f"Selected Action: {action}")
        
        # 2. Execute Action
        success, exec_time = self.executor.execute(action)
        
        # 3. Record Metrics and Log
        self.metrics.record_recovery(action, exec_time, success)
        self._log_recovery("FAULT", "None", 0, fault_type, action, exec_time, success)

    def get_metrics(self):
        return self.metrics.get_metrics()
