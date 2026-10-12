import csv
import os
from datetime import datetime, timezone

class ResultLogger:
    """
    Handles persisting individual experiment trial results to CSV for later analysis.
    """
    def __init__(self):
        self.results_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(self.results_dir, exist_ok=True)
        self.csv_path = os.path.join(self.results_dir, "results.csv")
        self._init_csv()

    def _init_csv(self):
        # We overwrite at the start of a fresh research evaluation suite run
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'experiment_id', 'scenario', 'system_type', 
                'mttr', 'success_rate', 'uptime', 'latency', 'throughput'
            ])

    def log_result(self, exp_id, scenario, sys_type, mttr, success_rate, uptime, latency, throughput):
        timestamp = datetime.now(timezone.utc).isoformat()
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, exp_id, scenario, sys_type, 
                round(mttr, 4), round(success_rate, 2), round(uptime, 2), 
                round(latency, 2), round(throughput, 2)
            ])
