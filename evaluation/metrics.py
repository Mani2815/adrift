import time

class EvaluationMetricsTracker:
    """
    Precise tracker for calculating research metrics: MTTR, Success Rate, and Uptime.
    """
    def __init__(self):
        self.system_start_time = time.time()
        self.fault_start_times = {}
        self.recoveries = []
        
        self.total_recovery_attempts = 0
        self.successful_recoveries = 0
        self.total_downtime = 0.0

    def reset_for_trial(self):
        self.fault_start_times.clear()
        self.recoveries.clear()
        self.total_recovery_attempts = 0
        self.successful_recoveries = 0
        self.total_downtime = 0.0
        self.system_start_time = time.time()

    def record_fault_start(self, fault_id):
        self.fault_start_times[fault_id] = time.time()

    def record_recovery_attempt(self, fault_id, success):
        self.total_recovery_attempts += 1
        recovery_end_time = time.time()
        
        start_time = self.fault_start_times.get(fault_id, recovery_end_time)
        mttr = recovery_end_time - start_time
        
        if success:
            self.successful_recoveries += 1
            
        self.total_downtime += mttr
        self.recoveries.append(mttr)
        
        return mttr

    def get_summary(self):
        if not self.recoveries:
            return {
                "avg_mttr": 0.0, "min_mttr": 0.0, "max_mttr": 0.0,
                "success_rate": 100.0, "uptime_percent": 100.0, "total_downtime": 0.0
            }
            
        avg_mttr = sum(self.recoveries) / len(self.recoveries)
        min_mttr = min(self.recoveries)
        max_mttr = max(self.recoveries)
        
        success_rate = (self.successful_recoveries / self.total_recovery_attempts) * 100
        
        total_runtime = time.time() - self.system_start_time
        # Uptime is bounded between 0 and 100
        uptime_percent = max(0.0, ((total_runtime - self.total_downtime) / total_runtime) * 100) if total_runtime > 0 else 100.0
        
        return {
            "avg_mttr": avg_mttr,
            "min_mttr": min_mttr,
            "max_mttr": max_mttr,
            "success_rate": success_rate,
            "uptime_percent": uptime_percent,
            "total_downtime": self.total_downtime
        }
