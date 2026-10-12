"""
multi_seed_trainer.py
=====================
Camera-Ready Phase 2: Independent PPO Training-Seed Workflow.

Trains multiple independent PPO policies across configurable random seeds
(default: [42, 43, 44, 45, 46]) with 100,000 timesteps per seed, isolated
artifact storage, learning curves, and SHA-256 checkpoint hashing.
"""

import os
import sys
import time
import json
import csv
import hashlib
import random
from datetime import datetime, timezone
import numpy as np
import torch
import gymnasium as gym

# Ensure project root in sys.path
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BASE_DIR not in sys.path:
    sys.path.insert(0, _BASE_DIR)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from ppo.pipeline_env import PipelineEnv
from config.logging_config import setup_logger

logger = setup_logger("MultiSeedTrainer", os.path.join(_BASE_DIR, "logs", "multi_seed_trainer.log"))

PHASE2_RESULTS_DIR = os.path.join(_BASE_DIR, "evaluation", "results", "camera_ready_phase2")

DEFAULT_TRAINING_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_TOTAL_TIMESTEPS = 100000

PPO_HYPERPARAMETERS = {
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
}


class IsolatedLearningCurveLogger(BaseCallback):
    """Logs evaluation reward and success rate at periodic intervals during training."""
    def __init__(self, eval_env, log_file, eval_freq=5000, n_eval_episodes=5, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_file = log_file
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes

        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        with open(self.log_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "mean_reward", "success_rate"])

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
                        ep_success = (
                            info.get("latency", 0) < 2.0
                            and info.get("severity_score", 0) < 50.0
                        )
                rewards.append(ep_reward)
                successes.append(1.0 if ep_success else 0.0)

            mean_reward = float(np.mean(rewards))
            success_rate = float(np.mean(successes) * 100.0)

            with open(self.log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([self.num_timesteps, round(mean_reward, 2), round(success_rate, 2)])
        return True


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def train_single_seed(
    seed: int,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    base_output_dir: str = PHASE2_RESULTS_DIR,
    force_retrain: bool = False,
) -> dict:
    """
    Trains an independent PPO agent under a specific random seed.
    Saves isolated checkpoint, logs, and manifest in seed-specific directory.
    """
    seed_dir = os.path.join(base_output_dir, f"training_seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    checkpoint_path = os.path.join(seed_dir, "ppo_agent.zip")
    manifest_path = os.path.join(seed_dir, "training_manifest.json")
    learning_curve_csv = os.path.join(seed_dir, "learning_curve.csv")

    if not force_retrain and os.path.exists(checkpoint_path) and os.path.exists(manifest_path):
        logger.info(f"Seed {seed} already has trained checkpoint. Checking manifest...")
        with open(manifest_path, "r") as f:
            existing_manifest = json.load(f)
        if existing_manifest.get("status") == "COMPLETED":
            logger.info(f"Reusing verified completed checkpoint for seed {seed}")
            return existing_manifest

    logger.info(f"=== Starting PPO Training for Seed {seed} ({total_timesteps} steps) ===")
    start_time_iso = datetime.now(timezone.utc).isoformat()
    t0 = time.time()

    # 1. Seed all RNG sources
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 2. Setup environments
    env = PipelineEnv()
    eval_env = PipelineEnv()
    env.reset(seed=seed)
    eval_env.reset(seed=seed + 10000)

    # 3. Configure PPO
    config = PPO_HYPERPARAMETERS.copy()
    config["seed"] = seed

    model = PPO("MlpPolicy", env, verbose=0, **config)

    curve_logger = IsolatedLearningCurveLogger(
        eval_env, learning_curve_csv, eval_freq=5000, n_eval_episodes=5
    )

    # 4. Train
    try:
        model.learn(total_timesteps=total_timesteps, callback=[curve_logger])
        status = "COMPLETED"
        err_msg = ""
    except Exception as e:
        status = "FAILED"
        err_msg = str(e)
        logger.error(f"Training failed for seed {seed}: {e}")

    t1 = time.time()
    end_time_iso = datetime.now(timezone.utc).isoformat()
    duration_s = t1 - t0

    # 5. Save model and compute hash
    if status == "COMPLETED":
        model_save_base = os.path.join(seed_dir, "ppo_agent")
        model.save(model_save_base)
        sha256_hash = compute_file_sha256(checkpoint_path)
    else:
        sha256_hash = ""

    # 6. Build Manifest
    manifest = {
        "training_seed": seed,
        "total_timesteps": total_timesteps,
        "status": status,
        "error_message": err_msg,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "duration_seconds": round(duration_s, 2),
        "steps_per_second": round(total_timesteps / max(1.0, duration_s), 1),
        "checkpoint_file": checkpoint_path,
        "checkpoint_sha256": sha256_hash,
        "learning_curve_file": learning_curve_csv,
        "hyperparameters": config,
        "environment": {
            "name": "PipelineEnv",
            "observation_dim": int(env.observation_space.shape[0]),
            "action_space_size": int(env.action_space.n),
            "max_steps": int(env.max_steps),
        },
        "software_versions": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "gymnasium": gym.__version__,
            "numpy": np.__version__,
        },
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=4)

    logger.info(
        f"✅ Seed {seed} Training {status} in {duration_s:.1f}s ({manifest['steps_per_second']} steps/s). "
        f"SHA256: {sha256_hash[:16]}..."
    )
    return manifest


def train_all_seeds(
    seeds=DEFAULT_TRAINING_SEEDS,
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS,
    base_output_dir: str = PHASE2_RESULTS_DIR,
    force_retrain: bool = False,
) -> list:
    """Trains all specified seeds sequentially, returning list of manifests."""
    manifests = []
    for s in seeds:
        m = train_single_seed(
            seed=s,
            total_timesteps=total_timesteps,
            base_output_dir=base_output_dir,
            force_retrain=force_retrain,
        )
        manifests.append(m)

    summary_manifest_path = os.path.join(base_output_dir, "all_seeds_training_manifest.json")
    with open(summary_manifest_path, "w") as f:
        json.dump(manifests, f, indent=4)

    return manifests


if __name__ == "__main__":
    train_all_seeds()
