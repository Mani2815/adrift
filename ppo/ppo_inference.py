import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from ppo.state_encoder import StateEncoder
from ppo.action_mapper import ActionMapper
from config.settings import settings
import logging

logger = logging.getLogger("PPO_Inference")

class PPOInferenceEngine:
    """
    Loads a trained PPO agent and uses it to predict recovery actions in the live pipeline.
    """
    def __init__(self):
        self.state_encoder = StateEncoder()
        self.action_mapper = ActionMapper()
        self.model = None
        self._load_model()
        
    def _load_model(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(base_dir, settings.MODEL_PATH)
        
        # SB3 automatically appends .zip if not provided, but we'll check it explicitly
        if not model_path.endswith('.zip'):
            model_path += '.zip'
            
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            logger.info(f"✅ Successfully loaded PPO model from {model_path}")
        else:
            logger.error(f"❌ PPO model not found at {model_path}. You must run 'python ppo/train_ppo.py' first!")

    def predict_action(self, pipeline_state: dict) -> str:
        """
        Receives the raw state dictionary from the RecoveryEngine, encodes it into the 
        continuous vector space, and asks the Neural Network for the optimal action.
        """
        if self.model is None:
            logger.warning("PPO model not loaded. Falling back to 'Ignore Event'.")
            return "Ignore Event"
            
        encoded_state = self.state_encoder.encode(pipeline_state)
        
        # Predict returns the action index. deterministic=True ensures we always take the best action
        action_idx, _ = self.model.predict(encoded_state, deterministic=True)
        
        action_str = self.action_mapper.get_action_string(action_idx)
        return action_str
