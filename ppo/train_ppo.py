import os
import sys

# Ensure imports work from the root path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from ppo.pipeline_env import PipelineEnv
from config.logging_config import setup_logger
import numpy as np
import csv

logger = setup_logger("PPO_Trainer", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "ppo_agent.log"))

class LearningCurveLogger(BaseCallback):
    def __init__(self, eval_env, log_file, eval_freq=5000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_file = log_file
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestep', 'mean_reward', 'success_rate'])

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            rewards = []
            successes = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                done = False
                ep_reward = 0.0
                ep_success = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    ep_reward += reward
                    done = terminated or truncated
                    if done:
                        # Infer success based on final healthy pipeline metrics
                        ep_success = info.get("latency", 0) < 2.0 and info.get("severity_score", 0) < 50.0
                rewards.append(ep_reward)
                successes.append(1.0 if ep_success else 0.0)
                
            mean_reward = np.mean(rewards)
            success_rate = np.mean(successes) * 100
            
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, round(mean_reward, 2), round(success_rate, 2)])
        return True

def train():
    logger.info("Initializing Custom Pipeline Environment for PPO Training...")
    env = PipelineEnv()
    eval_env = PipelineEnv()
    
    ppo_config = {
        "learning_rate": 0.0003,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "seed": 42
    }
    
    model = PPO("MlpPolicy", env, verbose=1, **ppo_config)
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    
    import json
    with open(os.path.join(base_dir, "ppo", "ppo_configuration.json"), "w") as f:
        json.dump(ppo_config, f, indent=4)
        
    tex_content = """\\begin{table}[h]
\\caption{PPO Agent Hyperparameters}
\\label{tab:ppo_hyperparameters}
\\centering
\\begin{tabular}{@{}ll@{}}
\\toprule
\\textbf{Parameter} & \\textbf{Value} \\\\ \\midrule
"""
    for k, v in ppo_config.items():
        k_tex = k.replace('_', '\\_')
        tex_content += f"{k_tex} & {v} \\\\\n"
    tex_content += """\\bottomrule
\\end{tabular}
\\end{table}
"""
    with open(os.path.join(base_dir, "ppo", "ppo_configuration.tex"), "w") as f:
        f.write(tex_content)
    model_dir = os.path.join(base_dir, "models")
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    log_dir = os.path.join(base_dir, "logs")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=checkpoint_dir, name_prefix="ppo_model")
    
    learning_curve_csv = os.path.join(log_dir, "ppo_learning_curve.csv")
    curve_logger = LearningCurveLogger(eval_env, learning_curve_csv, eval_freq=5000)
                                 
    logger.info("Starting PPO offline training for 100,000 timesteps...")
    
    try:
        model.learn(total_timesteps=100000, callback=[checkpoint_callback, curve_logger])
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return
        
    model_path = os.path.join(model_dir, "ppo_agent")
    model.save(model_path)
    logger.info(f"✅ Training Complete. Model successfully saved to {model_path}.zip")
    
if __name__ == "__main__":
    train()
