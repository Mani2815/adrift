#!/usr/bin/env python3
"""
docker_recovery_runner.py
=========================
Harness for ADRIFT Camera-Ready Phase 4: Real Docker-Based Recovery Validation.

This module conducts genuine container-level recovery experiments:
- Experiment A: Actual consumer container termination and restart with message streaming.
- Experiment B: Actuator and controller validation across edge cases and unsupported actions.
- Experiment C: Policy-in-the-loop recovery across all 5 PPO training seeds (42-46) and Rule Engine.

At least 10 controlled real restart trials are executed and recorded with
high-resolution monotonic timestamps (time.perf_counter()), wall-clock alignment,
downtime accounting, and offset/message-count data integrity indicators.
"""

import os
import sys
import time
import json
import uuid
import random
import socket
import logging
import argparse
import subprocess
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from ppo.state_encoder import StateEncoder
from ppo.action_mapper import ActionMapper
from recovery.rule_engine import RuleEngine
from recovery.docker_actuator import DockerRecoveryActuator, ActuationResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("DockerRecoveryRunner")

try:
    from kafka import KafkaProducer, KafkaAdminClient
    from kafka.admin import NewTopic
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    logger.warning("kafka-python not available on host.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results", "camera_ready_phase4_docker")
CHECKPOINTS_BASE = os.path.join(BASE_DIR, "evaluation", "results", "camera_ready_phase2")
PPO_SEEDS = [42, 43, 44, 45, 46]

DEFAULT_CONTAINER_NAME = "adrift-consumer"
DEFAULT_TOPIC = "adrift_events"
DEFAULT_KAFKA_HOST = "localhost:9092"
DEFAULT_READINESS_URL = "http://localhost:8080/ready"
DEFAULT_METRICS_URL = "http://localhost:8080/metrics"


class DockerEnvironmentManager:
    """Manages Docker Compose services and verifies container readiness."""

    def __init__(self, compose_file: str = "docker-compose.yml"):
        self.compose_file = os.path.join(BASE_DIR, compose_file)

    def is_docker_available(self) -> bool:
        try:
            res = subprocess.run(["docker", "info"], capture_output=True, timeout=5.0)
            return res.returncode == 0
        except Exception:
            return False

    def wait_for_port(self, host: str, port: int, timeout_s: float = 30.0) -> bool:
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < timeout_s:
            try:
                with socket.create_connection((host, port), timeout=1.0):
                    return True
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(0.5)
        return False

    def ensure_kafka_ready(self, timeout_s: float = 40.0) -> bool:
        logger.info("Verifying Kafka broker connectivity on localhost:9092...")
        ready = self.wait_for_port("localhost", 9092, timeout_s)
        if ready:
            logger.info("Kafka broker port 9092 is open and accepting TCP connections.")
            time.sleep(2.0)  # Brief settle time for broker initialization
            return True
        logger.error(f"Kafka broker failed to become ready within {timeout_s}s.")
        return False

    def ensure_topic(self, topic: str = DEFAULT_TOPIC):
        if not KAFKA_AVAILABLE:
            return
        try:
            admin = KafkaAdminClient(bootstrap_servers=[DEFAULT_KAFKA_HOST], request_timeout_ms=5000)
            existing = admin.list_topics()
            if topic not in existing:
                logger.info(f"Creating Kafka topic '{topic}'...")
                admin.create_topics([NewTopic(name=topic, num_partitions=1, replication_factor=1)])
                time.sleep(1.0)
            admin.close()
        except Exception as e:
            logger.warning(f"Topic creation check: {e}")

    def start_consumer_container(self, timeout_s: float = 20.0) -> bool:
        logger.info(f"Starting consumer container '{DEFAULT_CONTAINER_NAME}' via docker compose...")
        res = subprocess.run(
            ["docker", "compose", "-f", self.compose_file, "up", "-d", "consumer"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=30.0
        )
        if res.returncode != 0:
            logger.error(f"Failed to start consumer container: {res.stderr}")
            return False

        # Wait for HTTP readiness
        actuator = DockerRecoveryActuator()
        ready, elapsed, _ = actuator.wait_for_readiness(timeout_s)
        if ready:
            logger.info(f"Consumer container is READY on port 8080 ({elapsed:.2f}s).")
            return True
        else:
            logger.error(f"Consumer container failed HTTP readiness check within {timeout_s}s.")
            return False

    def stop_consumer_container(self) -> bool:
        logger.info(f"Stopping container '{DEFAULT_CONTAINER_NAME}' via docker stop...")
        res = subprocess.run(
            ["docker", "stop", DEFAULT_CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=15.0
        )
        return res.returncode == 0

    def kill_consumer_container(self) -> bool:
        logger.info(f"Killing container '{DEFAULT_CONTAINER_NAME}' via docker kill...")
        res = subprocess.run(
            ["docker", "kill", DEFAULT_CONTAINER_NAME],
            capture_output=True,
            text=True,
            timeout=15.0
        )
        return res.returncode == 0


class ControlledMessageProducer:
    """Produces verified test message batches to Kafka topic."""

    def __init__(self, bootstrap_servers: str = DEFAULT_KAFKA_HOST, topic: str = DEFAULT_TOPIC):
        self.topic = topic
        self.producer = None
        if KAFKA_AVAILABLE:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=[bootstrap_servers],
                    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                    retries=3,
                    request_timeout_ms=5000
                )
            except Exception as e:
                logger.warning(f"Could not initialize KafkaProducer: {e}")

    def produce_batch(self, count: int = 10, mean_val: float = 50.0) -> List[str]:
        if not self.producer:
            logger.warning("Producer not initialized; skipping message production.")
            return []

        event_ids = []
        for _ in range(count):
            eid = str(uuid.uuid4())
            payload = {
                "event_id": eid,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "value": round(random.gauss(mean_val, 5.0), 2),
                "source": "docker_recovery_test"
            }
            self.producer.send(self.topic, value=payload)
            event_ids.append(eid)
        self.producer.flush()
        return event_ids

    def close(self):
        if self.producer:
            try:
                self.producer.close(timeout=2.0)
            except Exception:
                pass


class DockerRecoveryEvaluator:
    """Orchestrates Phase 4 Experiments A, B, and C."""

    def __init__(self):
        self.env_mgr = DockerEnvironmentManager()
        self.actuator = DockerRecoveryActuator()
        self.state_encoder = StateEncoder()
        self.action_mapper = ActionMapper()
        self.rule_engine = RuleEngine()
        self.producer = ControlledMessageProducer()
        os.makedirs(RESULTS_DIR, exist_ok=True)

    def load_ppo_policy(self, seed: int):
        model_path = os.path.join(CHECKPOINTS_BASE, f"training_seed_{seed}", "ppo_agent.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint for seed {seed} not found at {model_path}")
        return PPO.load(model_path)

    # -------------------------------------------------------------------------
    # Experiment A: Single Controlled Container Termination & Restart Flow
    # -------------------------------------------------------------------------
    def run_experiment_a_baseline(self) -> Dict[str, Any]:
        """Validates end-to-end container crash, observation, restart, and resumed streaming."""
        logger.info("=== Running Experiment A: Controlled Termination & Restart Validation ===")
        
        # 1. Ensure consumer is running and ready
        if not self.actuator.check_readiness()[0]:
            self.env_mgr.start_consumer_container()

        # 2. Baseline message production
        pre_ids = self.producer.produce_batch(count=15, mean_val=50.0)
        time.sleep(1.0)
        metrics_pre = self.actuator.query_metrics() or {}
        pre_processed = metrics_pre.get("messages_processed", 0)

        # 3. Inject controlled failure: Kill container
        t_fault_injected = time.perf_counter()
        fault_wall_clock = datetime.now(timezone.utc).isoformat()
        logger.info("Injecting failure: terminating consumer container...")
        self.env_mgr.kill_consumer_container()

        # 4. Observe failure
        t_observed_start = time.perf_counter()
        ready_state = True
        while ready_state and (time.perf_counter() - t_observed_start < 5.0):
            ready_state, _ = self.actuator.check_readiness()
            if ready_state:
                time.sleep(0.05)
        t_fault_observed = time.perf_counter()
        observation_latency = t_fault_observed - t_fault_injected
        logger.info(f"Failure observed via HTTP poll in {observation_latency:.4f}s")

        # 5. Actuate restart
        t_act_start = time.perf_counter()
        act_res = self.actuator.restart_consumer(fault_timestamp=t_fault_injected)
        t_act_end = time.perf_counter()

        # 6. Post-recovery message production
        post_ids = self.producer.produce_batch(count=15, mean_val=50.0)
        t_poll_start = time.perf_counter()
        metrics_post = {}
        while time.perf_counter() - t_poll_start < 4.0:
            metrics_post = self.actuator.query_metrics() or {}
            if metrics_post.get("messages_processed", 0) > 0:
                break
            time.sleep(0.3)
        post_processed = metrics_post.get("messages_processed", 0)

        total_produced = len(pre_ids) + len(post_ids)
        downtime = act_res.end_to_end_recovery_s or (t_act_end - t_fault_injected)

        res = {
            "experiment": "Experiment A (Baseline Restart)",
            "success": act_res.success,
            "status": act_res.status,
            "fault_injected_wall_clock": fault_wall_clock,
            "observation_latency_s": round(observation_latency, 4),
            "docker_restart_duration_s": round(act_res.docker_restart_duration_s, 4),
            "readiness_wait_duration_s": round(act_res.readiness_wait_duration_s, 4),
            "end_to_end_recovery_s": round(downtime, 4),
            "service_downtime_s": round(downtime, 4),
            "messages_produced": total_produced,
            "messages_processed_post": post_processed,
            "last_offset": metrics_post.get("last_offset"),
            "container_id": act_res.container_id,
        }
        logger.info(f"Experiment A result: {res}")
        return res

    # -------------------------------------------------------------------------
    # Experiment B: Actuator & Controller Edge-Case Validation
    # -------------------------------------------------------------------------
    def run_experiment_b_validation(self) -> List[Dict[str, Any]]:
        """Validates actuator error handling, edge cases, and closed failures."""
        logger.info("=== Running Experiment B: Actuator & Controller Boundary Validation ===")
        results = []

        # Case 1: Nonexistent container (whitelisted, but container does not exist in Docker Engine)
        logger.info("Case 1: Restart nonexistent container...")
        act_nonexistent = DockerRecoveryActuator(
            container_name="nonexistent_container_xyz",
            allowed_containers=[DEFAULT_CONTAINER_NAME, "nonexistent_container_xyz"]
        )
        res_nonexistent = act_nonexistent.restart_consumer()
        results.append({
            "test_case": "Nonexistent Container",
            "target": "nonexistent_container_xyz",
            "expected_status": "CONTAINER_NOT_FOUND",
            "actual_status": res_nonexistent.status,
            "success": res_nonexistent.success,
            "passed": (res_nonexistent.status == "CONTAINER_NOT_FOUND" and not res_nonexistent.success)
        })

        # Case 2: Invalid container regex injection
        logger.info("Case 2: Security check - invalid container name with semicolon...")
        res_regex = self.actuator.restart_consumer(target_container="consumer; rm -rf /")
        results.append({
            "test_case": "Container Regex Validation",
            "target": "consumer; rm -rf /",
            "expected_status": "INVALID_TARGET",
            "actual_status": res_regex.status,
            "success": res_regex.success,
            "passed": (res_regex.status == "INVALID_TARGET" and not res_regex.success)
        })

        # Case 3: Whitelist authorization check
        logger.info("Case 3: Security check - unauthorized container targeting...")
        res_auth = self.actuator.restart_consumer(target_container="some_valid_looking_container")
        results.append({
            "test_case": "Actuator Whitelist Authorization",
            "target": "some_valid_looking_container",
            "expected_status": "PERMISSION_DENIED",
            "actual_status": res_auth.status,
            "success": res_auth.success,
            "passed": (res_auth.status == "PERMISSION_DENIED" and not res_auth.success)
        })

        # Case 4: Readiness Timeout handling
        logger.info("Case 4: Readiness check timeout handling...")
        short_actuator = DockerRecoveryActuator(readiness_timeout_s=0.001)
        res_timeout = short_actuator.restart_consumer()
        results.append({
            "test_case": "Readiness Timeout Enforcement",
            "target": DEFAULT_CONTAINER_NAME,
            "expected_status": "TIMEOUT",
            "actual_status": res_timeout.status,
            "success": res_timeout.success,
            "passed": (res_timeout.status == "TIMEOUT" and not res_timeout.success)
        })

        # Case 5-7: Unsupported actions fail closed
        for unsupp_action in ["Retrain Model", "Rollback Previous Model", "Scale Processing Resources"]:
            logger.info(f"Case: Unsupported action '{unsupp_action}' fail-closed check...")
            res_unsupp = self.actuator.execute(unsupp_action)
            results.append({
                "test_case": f"Unsupported Action: {unsupp_action}",
                "target": DEFAULT_CONTAINER_NAME,
                "expected_status": "UNSUPPORTED_ACTION",
                "actual_status": res_unsupp.status,
                "success": res_unsupp.success,
                "passed": (res_unsupp.status == "UNSUPPORTED_ACTION" and not res_unsupp.success)
            })

        # Case 8: Ignore Event documented NO_OP
        logger.info("Case: Ignore Event documented NO_OP check...")
        res_ignore = self.actuator.execute("Ignore Event")
        results.append({
            "test_case": "Ignore Event NO_OP",
            "target": DEFAULT_CONTAINER_NAME,
            "expected_status": "NO_OP",
            "actual_status": res_ignore.status,
            "success": res_ignore.success,
            "passed": (res_ignore.status == "NO_OP" and res_ignore.success)
        })

        # Ensure container is restored and ready after tests
        self.actuator.restart_consumer()

        logger.info(f"Experiment B completed: {len(results)} validation checks performed.")
        return results

    # -------------------------------------------------------------------------
    # Experiment C: Controlled Policy-in-the-Loop & Rule-Engine Matrix
    # -------------------------------------------------------------------------
    def run_controlled_trial(
        self,
        trial_id: int,
        engine_type: str,
        training_seed: Optional[int],
        action_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes a single controlled real-container recovery trial."""
        logger.info(f"--- Running Trial {trial_id}: Engine={engine_type}, Seed={training_seed} ---")
        
        # 1. Verify consumer initial health
        if not self.actuator.check_readiness()[0]:
            self.env_mgr.start_consumer_container()

        # 2. Produce pre-fault stream
        pre_ids = self.producer.produce_batch(count=10, mean_val=50.0)
        time.sleep(0.5)

        # 3. Inject failure: stop container
        t_fault_injected = time.perf_counter()
        fault_wall_clock = datetime.now(timezone.utc).isoformat()
        self.env_mgr.stop_consumer_container()

        # 4. Observe failure
        t_obs_start = time.perf_counter()
        while self.actuator.check_readiness()[0] and (time.perf_counter() - t_obs_start < 5.0):
            time.sleep(0.05)
        t_fault_observed = time.perf_counter()
        observation_latency = t_fault_observed - t_fault_injected

        # 5. Build schema-valid observation matching StateEncoder
        # consumer_health=1 (failed), severity_score=100.0, throughput=0.0, latency=5.0, error_rate=0.5
        state_dict = {
            "drift_type": "None",
            "severity_score": 100.0,
            "consumer_health": 1,
            "latency": 5.0,
            "throughput": 0.0,
            "success_rate": 1.0,
            "drift_frequency": 0.0,
            "error_rate": 0.5
        }
        encoded_state = self.state_encoder.encode(state_dict)

        # 6. Action selection & decision latency
        t_dec_start = time.perf_counter()
        raw_probs = None

        if engine_type == "Rule Engine":
            selected_action = self.rule_engine.select_action_for_fault("Consumer Failure")
            t_dec_end = time.perf_counter()
        elif engine_type == "PPO Policy":
            model = self.load_ppo_policy(training_seed)
            obs_tensor, _ = model.policy.obs_to_tensor(encoded_state)
            dist = model.policy.get_distribution(obs_tensor)
            raw_probs = dist.distribution.probs.detach().cpu().numpy()[0].tolist()
            action_idx, _ = model.predict(encoded_state, deterministic=True)
            selected_action = self.action_mapper.get_action_string(action_idx)
            t_dec_end = time.perf_counter()
        else:
            raise ValueError(f"Unknown engine: {engine_type}")

        if action_override:
            selected_action = action_override

        decision_latency = t_dec_end - t_dec_start

        # 7. Actuate recovery
        t_act_start = time.perf_counter()
        actuator_start_latency = t_act_start - t_dec_end
        
        act_res = self.actuator.execute(
            action=selected_action,
            fault_timestamp=t_fault_injected
        )
        t_act_finish = time.perf_counter()

        # 8. Post-recovery verification & streaming
        post_ids = []
        metrics_post = {}
        if act_res.success and act_res.status == "SUCCESS":
            post_ids = self.producer.produce_batch(count=10, mean_val=50.0)
            t_poll_start = time.perf_counter()
            while time.perf_counter() - t_poll_start < 4.0:
                metrics_post = self.actuator.query_metrics() or {}
                if metrics_post.get("messages_processed", 0) > 0:
                    break
                time.sleep(0.3)

        # 9. Compute metrics
        e2e_recovery = (t_act_finish - t_fault_injected) if act_res.success else None
        downtime = e2e_recovery

        record = {
            "trial_id": trial_id,
            "engine_type": engine_type,
            "training_seed": training_seed if training_seed is not None else "N/A",
            "wall_clock_timestamp": fault_wall_clock,
            "selected_action": selected_action,
            "action_supported": selected_action in DockerRecoveryActuator.SUPPORTED_ACTIONS,
            "action_status": act_res.status,
            "success": act_res.success,
            "observation_latency_s": round(observation_latency, 5),
            "decision_latency_s": round(decision_latency, 5),
            "actuator_start_latency_s": round(actuator_start_latency, 5),
            "docker_restart_duration_s": round(act_res.docker_restart_duration_s, 5),
            "readiness_wait_duration_s": round(act_res.readiness_wait_duration_s, 5),
            "end_to_end_recovery_s": round(e2e_recovery, 5) if e2e_recovery else None,
            "service_downtime_s": round(downtime, 5) if downtime else None,
            "messages_produced_before_fault": len(pre_ids),
            "messages_produced_after_fault": len(post_ids),
            "messages_processed_total": metrics_post.get("messages_processed", 0),
            "last_offset": metrics_post.get("last_offset"),
            "container_id": act_res.container_id,
            "error_reason": act_res.error_reason,
            "raw_action_probs": [round(p, 5) for p in raw_probs] if raw_probs else None,
        }
        return record

    def run_all_experiments(self, repetitions_per_config: int = 2) -> Dict[str, Any]:
        """
        Executes full suite:
        - 1 baseline Experiment A trial
        - Experiment B boundary suite (8 tests)
        - Experiment C matrix: 2 trials for Rule Engine + 2 trials for each of 5 PPO seeds (12 trials total).
        Grand total: 13 real container restart trials executed!
        """
        logger.info("=================================================================")
        logger.info("STARTING REAL DOCKER RECOVERY EXPERIMENTAL SUITE (PHASE 4)")
        logger.info("=================================================================")

        # Step 0: Ensure Docker and Kafka infrastructure
        if not self.env_mgr.is_docker_available():
            raise RuntimeError("Docker Engine is not available.")
        
        self.env_mgr.ensure_kafka_ready()
        self.env_mgr.ensure_topic()
        self.env_mgr.start_consumer_container()

        # Step 1: Run Experiment A
        exp_a_res = self.run_experiment_a_baseline()

        # Step 2: Run Experiment B
        exp_b_res = self.run_experiment_b_validation()

        # Step 3: Run Experiment C (Policy-in-the-Loop matrix)
        trials = []
        trial_id = 1

        # Rule Engine repetitions
        for r in range(repetitions_per_config):
            trial_rec = self.run_controlled_trial(
                trial_id=trial_id,
                engine_type="Rule Engine",
                training_seed=None
            )
            trials.append(trial_rec)
            trial_id += 1

        # PPO Seeds 42-46 repetitions
        for seed in PPO_SEEDS:
            for r in range(repetitions_per_config):
                trial_rec = self.run_controlled_trial(
                    trial_id=trial_id,
                    engine_type="PPO Policy",
                    training_seed=seed
                )
                trials.append(trial_rec)
                trial_id += 1

        # Step 4: Write CSV and JSON results
        csv_path = os.path.join(RESULTS_DIR, "raw_docker_trials.csv")
        self._write_csv(trials, csv_path)

        summary = self._compute_summary(trials, exp_a_res, exp_b_res)
        summary_path = os.path.join(RESULTS_DIR, "docker_recovery_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        exp_b_path = os.path.join(RESULTS_DIR, "actuator_validation_results.json")
        with open(exp_b_path, "w") as f:
            json.dump(exp_b_res, f, indent=2)

        logger.info(f"Phase 4 real Docker experiments successfully completed!")
        logger.info(f"Raw trial CSV saved to: {csv_path}")
        logger.info(f"Summary JSON saved to: {summary_path}")
        logger.info(f"Validation JSON saved to: {exp_b_path}")

        return summary

    def _write_csv(self, trials: List[Dict[str, Any]], path: str):
        import csv
        if not trials:
            return
        fieldnames = list(trials[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in trials:
                # Format raw_action_probs as json string for clean csv storage
                row = dict(t)
                if row.get("raw_action_probs") is not None:
                    row["raw_action_probs"] = json.dumps(row["raw_action_probs"])
                writer.writerow(row)

    def _compute_summary(
        self,
        trials: List[Dict[str, Any]],
        exp_a: Dict[str, Any],
        exp_b: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        total_trials = len(trials)
        successful_trials = sum(1 for t in trials if t["success"])
        failed_trials = sum(1 for t in trials if not t["success"])
        timed_out_trials = sum(1 for t in trials if t["action_status"] == "TIMEOUT")

        e2e_latencies = [t["end_to_end_recovery_s"] for t in trials if t["end_to_end_recovery_s"] is not None]
        docker_durations = [t["docker_restart_duration_s"] for t in trials if t["docker_restart_duration_s"] is not None]
        readiness_waits = [t["readiness_wait_duration_s"] for t in trials if t["readiness_wait_duration_s"] is not None]
        decision_latencies = [t["decision_latency_s"] for t in trials if t["decision_latency_s"] is not None]
        obs_latencies = [t["observation_latency_s"] for t in trials if t["observation_latency_s"] is not None]

        import numpy as np

        def stats_dict(vals):
            if not vals:
                return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
            return {
                "mean": round(float(np.mean(vals)), 4),
                "std": round(float(np.std(vals)), 4),
                "median": round(float(np.median(vals)), 4),
                "min": round(float(np.min(vals)), 4),
                "max": round(float(np.max(vals)), 4),
            }

        per_engine_summary = {}
        # Rule Engine stats
        rule_e2e = [t["end_to_end_recovery_s"] for t in trials if t["engine_type"] == "Rule Engine" and t["end_to_end_recovery_s"]]
        per_engine_summary["Rule Engine"] = {
            "trials": len([t for t in trials if t["engine_type"] == "Rule Engine"]),
            "success_rate": 1.0 if rule_e2e else 0.0,
            "e2e_recovery_stats": stats_dict(rule_e2e)
        }

        # PPO seed stats
        for s in PPO_SEEDS:
            seed_e2e = [t["end_to_end_recovery_s"] for t in trials if t["training_seed"] == s and t["end_to_end_recovery_s"]]
            per_engine_summary[f"PPO_Seed_{s}"] = {
                "trials": len([t for t in trials if t["training_seed"] == s]),
                "success_rate": 1.0 if seed_e2e else 0.0,
                "e2e_recovery_stats": stats_dict(seed_e2e)
            }

        return {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": "Phase 4 Real Docker Recovery Validation",
                "container_name": DEFAULT_CONTAINER_NAME,
                "topic": DEFAULT_TOPIC,
                "kafka_broker": DEFAULT_KAFKA_HOST,
                "docker_api": "1.56",
            },
            "trial_counts": {
                "total_controlled_trials": total_trials,
                "successful_trials": successful_trials,
                "failed_trials": failed_trials,
                "timed_out_trials": timed_out_trials,
                "success_rate": round(successful_trials / max(1, total_trials), 4)
            },
            "aggregate_metrics": {
                "observation_latency_s": stats_dict(obs_latencies),
                "decision_latency_s": stats_dict(decision_latencies),
                "docker_restart_duration_s": stats_dict(docker_durations),
                "readiness_wait_duration_s": stats_dict(readiness_waits),
                "end_to_end_recovery_s": stats_dict(e2e_latencies),
                "service_downtime_s": stats_dict(e2e_latencies),
            },
            "per_engine_summary": per_engine_summary,
            "experiment_a_baseline": exp_a,
            "experiment_b_validation_count": len(exp_b),
            "experiment_b_all_passed": all(t["passed"] for t in exp_b)
        }


def main():
    parser = argparse.ArgumentParser(description="ADRIFT Phase 4 Docker Recovery Runner")
    parser.add_argument("--repetitions", type=int, default=2, help="Repetitions per configuration (default: 2 -> 12 total trials)")
    args = parser.parse_args()

    evaluator = DockerRecoveryEvaluator()
    evaluator.run_all_experiments(repetitions_per_config=args.repetitions)


if __name__ == "__main__":
    main()
