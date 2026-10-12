import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os
import sys

# Ensure imports work from the root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppo.state_encoder import StateEncoder
from ppo.action_mapper import ActionMapper
from ppo.reward_engine import RewardEngine

class PipelineEnv(gym.Env):
    """
    Custom Environment that follows gymnasium interface.
    Simulates the ADRIFT Kafka pipeline states and fault scenarios to train the PPO agent offline.
    """
    metadata = {"render_modes": ["console"]}

    def __init__(self):
        super().__init__()
        
        self.state_encoder = StateEncoder()
        self.action_mapper = ActionMapper()
        self.reward_engine = RewardEngine()
        
        # Action space: 0-4
        self.action_space = spaces.Discrete(self.action_mapper.get_action_space_size())
        
        # Observation space: 8 continuous features
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.state_encoder.state_dim,), dtype=np.float32
        )
        
        self.max_steps = 50
        self.current_step = 0
        self.state_dict = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.current_step = 0
        
        # Initialize a healthy pipeline
        self.state_dict = {
            "drift_type": "None",
            "severity_score": random.uniform(0.0, 10.0),
            "latency": random.uniform(0.1, 0.5),
            "throughput": random.uniform(400.0, 500.0),
            "consumer_health": 0,
            "success_rate": 1.0,
            "drift_frequency": 0.0,
            "error_rate": random.uniform(0.0, 0.01)
        }
        
        # Immediately trigger a random scenario so the agent has a problem to solve
        self._inject_random_scenario()
        
        observation = self.state_encoder.encode(self.state_dict)
        return observation, self.state_dict

    def _inject_random_scenario(self):
        scenario = random.choice([
            "Sudden Drift", "Gradual Drift", "Incremental Drift", "Recurring Drift", 
            "Consumer Crash", "High Latency", "Healthy"
        ])
        
        if scenario in ["Sudden Drift", "Recurring Drift"]:
            self.state_dict["drift_type"] = scenario
            self.state_dict["severity_score"] = random.uniform(70.0, 100.0)
            self.state_dict["error_rate"] = random.uniform(0.1, 0.5)
        elif scenario == "Gradual Drift":
            self.state_dict["drift_type"] = scenario
            self.state_dict["severity_score"] = random.uniform(40.0, 60.0)
            self.state_dict["error_rate"] = random.uniform(0.05, 0.2)
        elif scenario == "Incremental Drift":
            self.state_dict["drift_type"] = scenario
            self.state_dict["severity_score"] = random.uniform(20.0, 40.0)
        elif scenario == "Consumer Crash":
            self.state_dict["consumer_health"] = 1
            self.state_dict["latency"] = 10.0
            self.state_dict["throughput"] = 0.0
            self.state_dict["severity_score"] = 100.0
        elif scenario == "High Latency":
            self.state_dict["latency"] = random.uniform(2.0, 5.0)
            self.state_dict["throughput"] = random.uniform(50.0, 200.0)
            self.state_dict["severity_score"] = random.uniform(50.0, 80.0)
        else: # Healthy
            self.state_dict["drift_type"] = "None"
            self.state_dict["severity_score"] = random.uniform(0.0, 20.0)
            self.state_dict["latency"] = random.uniform(0.1, 0.5)
            self.state_dict["throughput"] = random.uniform(400.0, 500.0)
            self.state_dict["consumer_health"] = 0

    def step(self, action_idx):
        self.current_step += 1
        
        pre_metrics = self.state_dict.copy()
        action_str = self.action_mapper.get_action_string(action_idx)
        
        # Simulate execution success probability
        success = False
        if action_str == "Retrain Model":
            success = random.random() < 0.85
        elif action_str == "Restart Consumer":
            success = random.random() < 0.95
        elif action_str == "Rollback Previous Model":
            success = random.random() < 0.90
        elif action_str == "Scale Processing Resources":
            success = random.random() < 0.80
        elif action_str == "Ignore Event":
            success = True # Ignore always "succeeds" execution
            
        # Apply transition dynamics based on action
        if action_str != "Ignore Event" and success:
            # Action succeeded, pipeline returns to healthy
            self.state_dict["drift_type"] = "None"
            self.state_dict["severity_score"] = random.uniform(0.0, 15.0)
            self.state_dict["consumer_health"] = 0
            self.state_dict["latency"] = random.uniform(0.1, 0.5)
            self.state_dict["throughput"] = random.uniform(400.0, 500.0)
            self.state_dict["error_rate"] = random.uniform(0.0, 0.01)
        elif action_str == "Ignore Event" and pre_metrics["severity_score"] > 50.0:
            # Ignored a severe fault, pipeline degrades rapidly
            self.state_dict["severity_score"] = min(100.0, self.state_dict["severity_score"] + 20.0)
            self.state_dict["latency"] += 1.0
            self.state_dict["throughput"] = max(0.0, self.state_dict["throughput"] - 100.0)
        
        # Track success rate feature
        self.state_dict["success_rate"] = (self.state_dict["success_rate"] * 0.9) + (0.1 if success else 0.0)
        
        # Calculate Reward
        reward = self.reward_engine.calculate_reward(action_str, success, pre_metrics, self.state_dict)
        
        # Determine termination
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        if success and action_str != "Ignore Event":
            # Successful recovery! End episode to reward the agent
            terminated = True
            
        # If pipeline gets completely destroyed
        if self.state_dict["latency"] > 15.0 or (self.state_dict["severity_score"] == 100.0 and self.current_step > 5):
            terminated = True
            reward -= 200 # Heavy penalty for total pipeline collapse

        observation = self.state_encoder.encode(self.state_dict)
        
        # If not terminated, inject a new scenario randomly
        if not terminated and random.random() < 0.1:
            self._inject_random_scenario()
            
        return observation, reward, terminated, truncated, self.state_dict
