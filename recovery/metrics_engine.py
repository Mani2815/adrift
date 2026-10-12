import time

class MetricsEngine:
    """
    Calculates operational recovery metrics including MTTR, 
    Uptime, and Recovery Success Rate.
    """
    def __init__(self):
        self.total_recoveries = 0
        self.successful_recoveries = 0
        self.total_mttr = 0.0
        self.action_distribution = {
            "Retrain Model": 0,
            "Restart Consumer": 0,
            "Rollback Previous Model": 0,
            "Scale Processing Resources": 0,
            "Ignore Event": 0
        }
        self.start_time = time.time()
        self.downtime = 0.0

    def record_recovery(self, action, execution_time, success):
        self.total_recoveries += 1
        if success:
            self.successful_recoveries += 1
            
        self.total_mttr += execution_time
        
        # Assume the pipeline is "down" while executing critical actions
        if action != "Ignore Event":
            self.downtime += execution_time
            
        if action in self.action_distribution:
            self.action_distribution[action] += 1
        else:
            self.action_distribution[action] = 1

    def get_metrics(self):
        mttr = self.total_mttr / self.total_recoveries if self.total_recoveries > 0 else 0
        success_rate = self.successful_recoveries / self.total_recoveries if self.total_recoveries > 0 else 1.0
        
        total_runtime = time.time() - self.start_time
        uptime = (total_runtime - self.downtime) / total_runtime if total_runtime > 0 else 1.0
        
        return {
            "mttr": round(mttr, 2),
            "success_rate": round(success_rate, 2),
            "uptime": round(uptime, 4),
            "total_recoveries": self.total_recoveries,
            "action_distribution": self.action_distribution
        }
