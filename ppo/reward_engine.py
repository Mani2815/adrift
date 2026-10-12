class RewardEngine:
    """
    Calculates the Reinforcement Learning reward signal based on the outcome 
    of a recovery action, following the strict positive/negative rules requested.
    """
    def calculate_reward(self, action_str: str, success: bool, pre_metrics: dict, post_metrics: dict) -> float:
        reward = 0.0
        
        if action_str == "Ignore Event":
            # If we correctly ignored a low severity event
            if pre_metrics.get("severity_score", 0) < 30.0:
                reward += 20.0
            # If we ignored a critical fault or drift, heavily penalize
            elif pre_metrics.get("consumer_health", 0) == 1 or pre_metrics.get("severity_score", 0) > 70.0:
                reward -= 100.0
        else:
            # Active recovery taken
            if success:
                reward += 100.0
                
                # Bonus for improved throughput
                if post_metrics.get("throughput", 0) > pre_metrics.get("throughput", 0):
                    reward += 30.0
                    
                # Maintained Uptime (implied by success)
                reward += 20.0
            else:
                reward -= 100.0
                reward -= 75.0 # Increased downtime penalty
                
            # Repeated recovery attempts / unnecessary acting
            if pre_metrics.get("severity_score", 0) < 20.0 and pre_metrics.get("consumer_health", 0) == 0:
                reward -= 25.0
                
        # High Latency penalty applies regardless of action
        if post_metrics.get("latency", 0) > 2.0:
            reward -= 50.0
            
        return reward
