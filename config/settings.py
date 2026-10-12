import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    KAFKA_BROKER = os.getenv("KAFKA_BROKER", "localhost:9092")
    TOPIC_NAME = os.getenv("TOPIC_NAME", "adrift_events")
    STREAM_DELAY = float(os.getenv("STREAM_DELAY", 0.1))
    DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 8501))
    DB_PATH = os.getenv("DB_PATH", "sqlite:///../data/adrift_metrics.db")
    
    # Optional dataset configurations
    DATASET_PATH = os.getenv("DATASET_PATH", "")
    DATA_SOURCE = os.getenv("DATA_SOURCE", "simulated") # Options: simulated, electricity, cicids2017

    # Phase 2: Drift Detection Configurations
    DRIFT_DETECTOR_ENABLED = os.getenv("DRIFT_DETECTOR_ENABLED", "True").lower() in ("true", "1", "yes")
    ADWIN_ENABLED = os.getenv("ADWIN_ENABLED", "True").lower() in ("true", "1", "yes")
    DDM_ENABLED = os.getenv("DDM_ENABLED", "True").lower() in ("true", "1", "yes")
    KSWIN_ENABLED = os.getenv("KSWIN_ENABLED", "True").lower() in ("true", "1", "yes")
    DRIFT_SCENARIO = os.getenv("DRIFT_SCENARIO", "Sudden")
    
    # Phase 3: Classification and Severity
    DRIFT_CLASSIFICATION_ENABLED = os.getenv("DRIFT_CLASSIFICATION_ENABLED", "True").lower() in ("true", "1", "yes")
    SEVERITY_SCORING_ENABLED = os.getenv("SEVERITY_SCORING_ENABLED", "True").lower() in ("true", "1", "yes")
    RECURRING_DRIFT_MEMORY_ENABLED = os.getenv("RECURRING_DRIFT_MEMORY_ENABLED", "True").lower() in ("true", "1", "yes")

    # Phase 4: Auto Recovery
    RECOVERY_ENGINE_ENABLED = os.getenv("RECOVERY_ENGINE_ENABLED", "True").lower() in ("true", "1", "yes")
    FAULT_MONITOR_ENABLED = os.getenv("FAULT_MONITOR_ENABLED", "True").lower() in ("true", "1", "yes")
    AUTO_RECOVERY_ENABLED = os.getenv("AUTO_RECOVERY_ENABLED", "True").lower() in ("true", "1", "yes")

    # Phase 5: PPO Agent
    PPO_ENABLED = os.getenv("PPO_ENABLED", "False").lower() in ("true", "1", "yes")
    TRAINING_MODE = os.getenv("TRAINING_MODE", "False").lower() in ("true", "1", "yes")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/ppo_agent.zip")
    REWARD_STRATEGY = os.getenv("REWARD_STRATEGY", "balanced")

settings = Config()
