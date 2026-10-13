#!/usr/bin/env python3
"""
docker_actuator.py
==================
Real Docker container recovery actuator for ADRIFT Phase 4.

Key Design Principles:
1. Genuine Container Operations: Invokes real Docker Engine operations via subprocess (shell=False)
   to restart or recover containerized streaming consumers.
2. Verified Readiness: Recovery is NOT declared successful when the Docker API returns;
   it requires the consumer to satisfy an HTTP readiness check (/ready -> 200 OK).
3. Monotonic Precision: Uses time.perf_counter() for exact monotonic duration accounting,
   preserving ISO-8601 wall-clock timestamps for log alignment.
4. Fail-Closed for Unsupported Actions: ADRIFT action space contains 5 actions:
   - Ignore Event: Documented no-op with audit logging.
   - Restart Consumer: Genuine container restart via Docker Engine.
   - Retrain Model: Refused with UNSUPPORTED_ACTION (no real retraining pipeline).
   - Rollback Previous Model: Refused with UNSUPPORTED_ACTION (no model version actuator).
   - Scale Processing Resources: Refused with UNSUPPORTED_ACTION (no orchestration autoscaler).
5. Operational Safety: Enforces strict container name whitelisting, prevents arbitrary execution,
   and avoids root-level or privileged assumptions.
"""

import os
import re
import time
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
import urllib.request
import urllib.error

logger = logging.getLogger("DockerRecoveryActuator")

ALLOWED_CONTAINER_REGEX = re.compile(r"^[a-zA-Z0-9_\-]+$")
DEFAULT_CONSUMER_CONTAINER = "adrift-consumer"
DEFAULT_READINESS_URL = "http://localhost:8080/ready"
DEFAULT_METRICS_URL = "http://localhost:8080/metrics"


@dataclass
class ContainerState:
    exists: bool
    status: str            # e.g., "running", "exited", "restarting", "not_found"
    running: bool
    restarting: bool
    exit_code: Optional[int] = None
    container_id: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class ActuationResult:
    action: str
    status: str            # SUCCESS, FAILED, TIMEOUT, UNSUPPORTED_ACTION, CONTAINER_NOT_FOUND, DOCKER_UNAVAILABLE, NO_OP
    success: bool
    wall_clock_timestamp: str
    request_timestamp: float
    docker_restart_duration_s: float
    readiness_wait_duration_s: float
    total_actuation_duration_s: float
    preflight_inspect_duration_s: float = 0.0
    end_to_end_recovery_s: Optional[float] = None
    container_id: Optional[str] = None
    error_reason: Optional[str] = None
    telemetry_after_recovery: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DockerRecoveryActuator:
    """
    Actuator for real Docker infrastructure recovery operations.
    Completely separated from the simulated RecoveryExecutor to ensure
    original Protocol B simulation benchmarks remain reproducible.
    """

    SUPPORTED_ACTIONS = {"Restart Consumer", "Ignore Event"}
    UNSUPPORTED_ACTIONS = {
        "Retrain Model": "Action not supported: Real model retraining pipeline is not deployed to local infrastructure.",
        "Rollback Previous Model": "Action not supported: Model rollback requires versioned deployment pipeline not present in local prototype.",
        "Scale Processing Resources": "Action not supported: Horizontal container scaling requires multi-replica orchestration not present in local prototype."
    }

    def __init__(
        self,
        container_name: str = DEFAULT_CONSUMER_CONTAINER,
        readiness_url: str = DEFAULT_READINESS_URL,
        metrics_url: str = DEFAULT_METRICS_URL,
        readiness_timeout_s: float = 15.0,
        poll_interval_s: float = 0.1,
        allowed_containers: Optional[List[str]] = None,
    ):
        if not ALLOWED_CONTAINER_REGEX.match(container_name):
            raise ValueError(f"Invalid container name '{container_name}': must match alphanumeric/dash/underscore.")
        
        self.container_name = container_name
        self.readiness_url = readiness_url
        self.metrics_url = metrics_url
        self.readiness_timeout_s = readiness_timeout_s
        self.poll_interval_s = poll_interval_s
        
        self.allowed_containers = set(allowed_containers or [DEFAULT_CONSUMER_CONTAINER])
        if self.container_name not in self.allowed_containers:
            self.allowed_containers.add(self.container_name)

    # -------------------------------------------------------------------------
    # Docker Engine Inspection and Management
    # -------------------------------------------------------------------------

    def inspect_container(self, target_container: Optional[str] = None) -> ContainerState:
        """Inspects real container state via docker inspect."""
        target = target_container or self.container_name
        self._validate_container_target(target)

        try:
            res = subprocess.run(
                ["docker", "inspect", target],
                capture_output=True,
                text=True,
                timeout=10.0,
                check=False
            )
        except subprocess.TimeoutExpired:
            return ContainerState(exists=False, status="timeout", running=False, restarting=False, error_message="Docker inspect timed out")
        except FileNotFoundError:
            return ContainerState(exists=False, status="docker_missing", running=False, restarting=False, error_message="Docker CLI binary not found")
        except Exception as e:
            return ContainerState(exists=False, status="error", running=False, restarting=False, error_message=str(e))

        if res.returncode != 0:
            return ContainerState(
                exists=False,
                status="not_found",
                running=False,
                restarting=False,
                error_message=res.stderr.strip() or "Container not found"
            )

        try:
            data = json.loads(res.stdout)
            if not data or not isinstance(data, list):
                return ContainerState(exists=False, status="parse_error", running=False, restarting=False)
            
            c_info = data[0]
            state = c_info.get("State", {})
            return ContainerState(
                exists=True,
                status=state.get("Status", "unknown"),
                running=bool(state.get("Running", False)),
                restarting=bool(state.get("Restarting", False)),
                exit_code=state.get("ExitCode"),
                container_id=c_info.get("Id", "")[:12],
                error_message=state.get("Error") or None
            )
        except Exception as e:
            return ContainerState(exists=False, status="parse_error", running=False, restarting=False, error_message=str(e))

    def _validate_container_target(self, target: str):
        if not ALLOWED_CONTAINER_REGEX.match(target):
            raise ValueError(f"Security violation: Container '{target}' violates naming regex.")
        if target not in self.allowed_containers:
            raise PermissionError(f"Security violation: Container '{target}' is not in authorized actuator whitelist.")

    # -------------------------------------------------------------------------
    # Readiness Verification
    # -------------------------------------------------------------------------

    def check_readiness(self) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Queries HTTP readiness endpoint. Returns (is_ready, response_json)."""
        req = urllib.request.Request(self.readiness_url, headers={"User-Agent": "ADRIFT-Actuator/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    try:
                        body = json.loads(resp.read().decode("utf-8"))
                        return True, body
                    except Exception:
                        return True, {}
                return False, None
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionRefusedError, TimeoutError, OSError):
            return False, None

    def query_metrics(self) -> Optional[Dict[str, Any]]:
        """Queries HTTP metrics endpoint."""
        req = urllib.request.Request(self.metrics_url, headers={"User-Agent": "ADRIFT-Actuator/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                if resp.status == 200:
                    return json.loads(resp.read().decode("utf-8"))
        except Exception:
            pass
        return None

    def wait_for_readiness(self, timeout_s: float) -> Tuple[bool, float, Optional[Dict[str, Any]]]:
        """Polls readiness endpoint until 200 OK or timeout. Returns (success, elapsed_s, telemetry)."""
        t_start = time.perf_counter()
        while time.perf_counter() - t_start < timeout_s:
            ready, telemetry = self.check_readiness()
            if ready:
                elapsed = time.perf_counter() - t_start
                return True, elapsed, telemetry
            time.sleep(self.poll_interval_s)
        return False, time.perf_counter() - t_start, None

    # -------------------------------------------------------------------------
    # Core Actuator Operations
    # -------------------------------------------------------------------------

    def restart_consumer(
        self,
        target_container: Optional[str] = None,
        fault_timestamp: Optional[float] = None
    ) -> ActuationResult:
        """
        Executes a real Docker restart operation against the streaming consumer.
        Verifies container presence, issues docker restart/start, and polls /ready.
        """
        target = target_container or self.container_name
        wall_clock = datetime.now(timezone.utc).isoformat()
        t_req = time.perf_counter()

        try:
            self._validate_container_target(target)
        except PermissionError as pe:
            logger.error(f"Target authorization failure: {pe}")
            return ActuationResult(
                action="Restart Consumer",
                status="PERMISSION_DENIED",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=0.0,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=time.perf_counter() - t_req,
                error_reason=str(pe)
            )
        except ValueError as ve:
            logger.error(f"Target validation failure: {ve}")
            return ActuationResult(
                action="Restart Consumer",
                status="INVALID_TARGET",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=0.0,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=time.perf_counter() - t_req,
                error_reason=str(ve)
            )

        # Step 1: Pre-restart Container State Verification
        t_inspect_start = time.perf_counter()
        initial_state = self.inspect_container(target)
        t_inspect_end = time.perf_counter()
        inspect_duration = t_inspect_end - t_inspect_start

        if not initial_state.exists:
            logger.error(f"Cannot restart container '{target}': container does not exist.")
            return ActuationResult(
                action="Restart Consumer",
                status="CONTAINER_NOT_FOUND",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=0.0,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=time.perf_counter() - t_req,
                preflight_inspect_duration_s=inspect_duration,
                error_reason=f"Container '{target}' not found: {initial_state.error_message}"
            )

        # Step 2: Issue genuine Docker command
        # If the container is stopped/exited, 'docker start' is faster and more reliable;
        # if running, 'docker restart' stops and starts it.
        cmd = ["docker", "restart", target] if initial_state.running else ["docker", "start", target]
        logger.info(f"Issuing real Docker command: {' '.join(cmd)}")
        
        t_cmd_start = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=15.0, check=False)
            t_cmd_end = time.perf_counter()
            docker_duration = t_cmd_end - t_cmd_start
        except subprocess.TimeoutExpired:
            t_cmd_end = time.perf_counter()
            logger.error(f"Docker command timed out after 15s")
            return ActuationResult(
                action="Restart Consumer",
                status="TIMEOUT",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=t_cmd_end - t_cmd_start,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=time.perf_counter() - t_req,
                container_id=initial_state.container_id,
                error_reason="Docker restart command execution timed out (15s)"
            )
        except Exception as e:
            t_cmd_end = time.perf_counter()
            logger.error(f"Failed to execute Docker command: {e}")
            return ActuationResult(
                action="Restart Consumer",
                status="DOCKER_UNAVAILABLE",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=t_cmd_end - t_cmd_start,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=time.perf_counter() - t_req,
                container_id=initial_state.container_id,
                error_reason=f"Docker CLI invocation error: {str(e)}"
            )

        if proc.returncode != 0:
            logger.error(f"Docker command failed (exit code {proc.returncode}): {proc.stderr.strip()}")
            return ActuationResult(
                action="Restart Consumer",
                status="FAILED",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=docker_duration,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=time.perf_counter() - t_req,
                container_id=initial_state.container_id,
                error_reason=f"Docker exited with code {proc.returncode}: {proc.stderr.strip()}"
            )

        # Step 3: Wait for verified service readiness
        logger.info(f"Container restarted in {docker_duration:.3f}s. Polling readiness at {self.readiness_url}...")
        ready_ok, ready_duration, telemetry = self.wait_for_readiness(self.readiness_timeout_s)
        t_finish = time.perf_counter()
        total_actuation = t_finish - t_req

        e2e_recovery = (t_finish - fault_timestamp) if fault_timestamp is not None else None

        if not ready_ok:
            logger.error(f"Readiness check timed out after {ready_duration:.3f}s (timeout={self.readiness_timeout_s}s)")
            return ActuationResult(
                action="Restart Consumer",
                status="TIMEOUT",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=docker_duration,
                readiness_wait_duration_s=ready_duration,
                total_actuation_duration_s=total_actuation,
                end_to_end_recovery_s=e2e_recovery,
                container_id=initial_state.container_id,
                error_reason=f"Consumer service failed to satisfy HTTP readiness check within {self.readiness_timeout_s}s"
            )

        # Step 4: Verify post-recovery telemetry
        final_metrics = self.query_metrics() or telemetry
        logger.info(
            f"✅ Container '{target}' verifiably RECOVERED! "
            f"Docker restart: {docker_duration:.3f}s, Readiness wait: {ready_duration:.3f}s, "
            f"Total actuation: {total_actuation:.3f}s"
        )

        return ActuationResult(
            action="Restart Consumer",
            status="SUCCESS",
            success=True,
            wall_clock_timestamp=wall_clock,
            request_timestamp=t_req,
            docker_restart_duration_s=docker_duration,
            readiness_wait_duration_s=ready_duration,
            total_actuation_duration_s=total_actuation,
            preflight_inspect_duration_s=inspect_duration,
            end_to_end_recovery_s=e2e_recovery,
            container_id=initial_state.container_id,
            error_reason=None,
            telemetry_after_recovery=final_metrics
        )

    # -------------------------------------------------------------------------
    # Universal Action Dispatcher (Fail-Closed)
    # -------------------------------------------------------------------------

    def execute(
        self,
        action: str,
        target_container: Optional[str] = None,
        fault_timestamp: Optional[float] = None
    ) -> ActuationResult:
        """
        Universal entrypoint for ADRIFT action execution.
        Enforces strict fail-closed semantics for unsupported actions.
        """
        wall_clock = datetime.now(timezone.utc).isoformat()
        t_req = time.perf_counter()

        if action == "Restart Consumer":
            return self.restart_consumer(target_container, fault_timestamp)

        elif action == "Ignore Event":
            logger.info("ADRIFT Decision: Ignore Event -> Verified NO_OP recorded.")
            return ActuationResult(
                action="Ignore Event",
                status="NO_OP",
                success=True,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=0.0,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=0.0,
                end_to_end_recovery_s=None,
                error_reason=None,
            )

        elif action in self.UNSUPPORTED_ACTIONS:
            reason = self.UNSUPPORTED_ACTIONS[action]
            logger.warning(f"ADRIFT Decision '{action}' is UNSUPPORTED on real infrastructure: {reason}")
            return ActuationResult(
                action=action,
                status="UNSUPPORTED_ACTION",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=0.0,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=0.0,
                end_to_end_recovery_s=None,
                error_reason=reason
            )

        else:
            logger.error(f"ADRIFT Decision '{action}' is completely UNRECOGNIZED.")
            return ActuationResult(
                action=action,
                status="UNKNOWN_ACTION",
                success=False,
                wall_clock_timestamp=wall_clock,
                request_timestamp=t_req,
                docker_restart_duration_s=0.0,
                readiness_wait_duration_s=0.0,
                total_actuation_duration_s=0.0,
                end_to_end_recovery_s=None,
                error_reason=f"Action '{action}' is not in the ADRIFT 5-action space."
            )
