import csv
import os
import uuid
from datetime import datetime, timezone

from config.settings import settings
from config.logging_config import setup_logger

logger = setup_logger("DatasetAdapter", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "producer.log"))

class DatasetAdapter:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name.lower()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.processed_dir = os.path.join(self.base_dir, "datasets", "processed")
        
        self.step = 0
        self.current_mean = 0.0 # for compatibility with existing logging
        
        if self.dataset_name == "electricity":
            self.filepath = os.path.join(self.processed_dir, "electricity.csv")
            self.value_col = "nswprice" # Primary value for standard drift detection
        elif self.dataset_name == "cicids2017":
            self.filepath = os.path.join(self.processed_dir, "cicids2017.csv")
            self.value_col = "Flow Duration" # E.g., Use flow duration as primary univariate drift target
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
        if not os.path.exists(self.filepath):
            logger.error(f"Dataset file not found: {self.filepath}. Please run data_pipeline/preprocess.py first.")
            raise FileNotFoundError(self.filepath)
            
        self.file_handle = open(self.filepath, 'r')
        self.csv_reader = csv.DictReader(self.file_handle)
        logger.info(f"Initialized DatasetAdapter for {self.dataset_name}")

    def generate_event(self):
        try:
            row = next(self.csv_reader)
            self.step += 1
            
            try:
                primary_val = float(row.get(self.value_col, 0.0))
            except ValueError:
                primary_val = 0.0
                
            self.current_mean = primary_val
                
            label = str(row.get("label", row.get("Label", row.get("class", "NORMAL"))))
            is_anomaly = label.upper() not in ["NORMAL", "BENIGN", "UP"]
            
            event = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_id": str(uuid.uuid4()),
                "value": primary_val,
                "source": self.dataset_name,
                "status": "ANOMALY" if is_anomaly else "OK",
                "simulated_fault": label if is_anomaly else None,
                "payload": row # Multivariate payload
            }
            return event
            
        except StopIteration:
            logger.info("End of dataset reached. Rewinding to start.")
            self.file_handle.seek(0)
            self.csv_reader = csv.DictReader(self.file_handle)
            return self.generate_event()

    def __del__(self):
        if hasattr(self, 'file_handle') and self.file_handle:
            self.file_handle.close()
