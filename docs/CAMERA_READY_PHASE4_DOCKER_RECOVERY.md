# ADRIFT Camera-Ready Revision — Phase 4: Real Docker-Based Recovery Validation

**Accepted Paper:** *ADRIFT: Adaptive Drift-Aware Recovery Framework for Intelligent Streaming Pipelines* (ICDDS 2026, Track II)  
**Date:** October 12, 2026  
**Status / Phase Verdict:** **PASS** (Genuine Containerized Streaming Consumer, Docker Engine Actuator, Policy-in-the-Loop Orchestration, 13 Real Container Restarts, 100% Verifiable Recovery)  
**Deliverable Document:** `docs/CAMERA_READY_PHASE4_DOCKER_RECOVERY.md`  

---

## 1. Executive Summary & Phase 4 Verdict

Phase 4 establishes an authentic container-level recovery prototype for ADRIFT, replacing previous simulated `time.sleep()` operational mocks with real Docker Engine operations against a containerized Kafka streaming consumer.

### Phase 4 Completion Criteria Verification Table

| Criterion | Required Target | Delivered Evidence | Verdict |
| :--- | :--- | :--- | :---: |
| **Real Containerized Consumer** | Dedicated container running Kafka consumer on stream | `consumer/streaming_consumer.py` inside `adrift-consumer:latest` Docker container, connected to `kafka:29092` | **PASS** |
| **Observable Readiness & Telemetry** | HTTP health & readiness probes | Ports `8080` exposing `/health` (200), `/ready` (200 when assigned partition), and `/metrics` (JSON telemetry) | **PASS** |
| **Genuine Docker Actuator** | Issue real Docker Engine restart, no `time.sleep` | `recovery/docker_actuator.py` executes `docker start`/`docker restart` via subprocess with high-resolution monotonic timing | **PASS** |
| **Verified Service Restoration** | Actuator waits for actual HTTP readiness, not CLI exit | Actuator polls `http://localhost:8080/ready` until 200 OK or timeout; records true readiness recovery duration | **PASS** |
| **Controlled Failure Ingestion** | Inject container termination and verify observation | Consumer container stopped/killed; controller detects HTTP connection refusal via documented signal path | **PASS** |
| **Policy-in-the-Loop Integration** | Connect Rule Engine & all 5 PPO seeds (42–46) | Evaluated Rule Engine + PPO Seeds 42, 43, 44, 45, 46 on schema-valid 8-feature observations | **PASS** |
| **Fail-Closed Unsupported Actions** | Retrain, Rollback, Scale must refuse without fake success | Strict `UNSUPPORTED_ACTION` recorded; zero fallback substitutions | **PASS** |
| **Actuator Edge-Case Testing** | Nonexistent containers, timeouts, regex injection, auth | 8/8 Experiment B edge-case validation checks executed and passing | **PASS** |
| **Minimum Real Restart Trials** | $\ge 10$ controlled restart trials | **13 real container restarts completed** (1 Experiment A baseline + 12 Experiment C controlled trials) | **PASS** |
| **Measured Recovery & Downtime** | True wall-clock & monotonic latency breakdown | Mean Observation: 1.09s, Docker Restart: 0.074s, Readiness Wait: 9.19s, End-to-End Downtime: 10.43s | **PASS** |
| **Data Integrity Verification** | Offset progression & message accounting | Monotonic offset advancement (offsets 339 $\to$ 579), 0 dropped messages, 0 duplicates detected | **PASS** |
| **Automated Test Suite** | Unit and integration test coverage | 13/13 tests in `test_docker_recovery.py` passed; 39/39 tests in full repo passed | **PASS** |
| **Preservation of Research Boundaries** | No edits to manuscript, Phase 1–3 artifacts untouched | `main.tex`, PDF, Phase 1 Protocol B CSVs, Phase 2 checkpoints, and Phase 3 SHAP data 100% preserved | **PASS** |

**OVERALL PHASE 4 VERDICT:** **PASS**

---

## 2. STAGE 1 — Repository and Runtime Audit

### 2.1 Audit of Baseline Repository State

Prior to Phase 4 implementation, an exhaustive inspection of the ADRIFT codebase was performed:
1. **Docker Compose Configuration (`docker-compose.yml`)**:
   - Contained services for ZooKeeper (`confluentinc/cp-zookeeper:7.5.0` on port 2181) and Kafka (`confluentinc/cp-kafka:7.5.0` on ports 9092 and 29092).
   - **Critical Gap Verified**: No containerized streaming consumer service existed in `docker-compose.yml`. The consumer ran solely as a host Python process (`consumer/consumer.py`).
2. **Recovery Actuator Implementation (`recovery/recovery_executor.py`)**:
   - Contained purely synthetic logic: `time.sleep(random.uniform(0.5, 1.5))` for `Restart Consumer` and random boolean success draws (`random.random() <= 0.95`).
   - **Critical Gap Verified**: No Docker Engine integration existed in the original codebase; all recovery was in-memory simulation.
3. **PPO Checkpoint & Inference Architecture**:
   - Five independent PPO checkpoints trained for 100,000 timesteps during Phase 2 exist at `evaluation/results/camera_ready_phase2/training_seed_{42..46}/ppo_agent.zip`.
   - Feature encoder (`ppo/state_encoder.py`) defines an 8-dimensional continuous observation vector.
   - Action mapper (`ppo/action_mapper.py`) maps discrete integers 0–4 to `Ignore Event`, `Retrain Model`, `Restart Consumer`, `Rollback Previous Model`, and `Scale Processing Resources`.

### 2.2 Execution Environment Specifications

The real Docker validation environment was audited and verified:
- **Operating System**: macOS (Darwin 24.6.0, `darwin/arm64`, Apple Silicon)
- **Docker Engine Client**: Version 29.8.0, API version 1.56, Go version `go1.26.8`
- **Docker Engine Server**: Docker Desktop 4.91.0 (239619), API 1.56, `linux/arm64` container VM
- **Docker Compose**: Version `v5.5.1`
- **Docker UNIX Socket**: `/var/run/docker.sock` (symlinked from `/Users/mani/.docker/run/docker.sock`)
- **Python Environment**: Python 3.9.6, PyTorch 2.8.0, Gymnasium 1.1.1, Stable-Baselines3 2.7.1, kafka-python 2.0.2, requests 2.32.5

---

## 3. STAGE 2 — Containerized Streaming Consumer Service

### 3.1 Architectural Design (`consumer/streaming_consumer.py` & `consumer/Dockerfile`)

A real, standalone streaming consumer service was implemented to run within its own containerized runtime environment.

```
+-----------------------------------------------------------------------------------------+
|                               adrift-consumer Container                                 |
|                                                                                         |
|  +-----------------------------+          +------------------------------------------+  |
|  |     Telemetry HTTP Server   |          |          Kafka Consumer Thread           |  |
|  |       (0.0.0.0:8080)        |          |                                          |  |
|  |                             |          |  1. Connects to kafka:29092              |  |
|  |  GET /health  -> 200 UP     |          |  2. Subscribes to topic: adrift_events   |  |
|  |  GET /ready   -> 200 READY  |<========>|  3. Participates in group rebalance      |  |
|  |  GET /metrics -> Telemetry  |  State   |  4. Assigned Partition 0                 |  |
|  |                  JSON       |  Tracker |  5. Consumes JSON payloads               |  |
|  +-----------------------------+          |  6. Computes rolling mean/variance       |  |
|                                           |  7. Tracks last_offset & message count   |  |
|                                           +------------------------------------------+  |
+-----------------------------------------------------------------------------------------+
```

### 3.2 Delivery and Offset-Commit Semantics
- **Offset Tracking**: Configured with `auto_offset_reset='earliest'`, `enable_auto_commit=True`, and `auto_commit_interval_ms=1000`.
- **Delivery Guarantee**: At-least-once delivery semantics under consumer failure.
- **Partition Assignment & Readiness Gate**:
  - The telemetry server reports `503 Service Unavailable` during startup, network detachment, or container shutdown.
  - The endpoint reports `200 OK` (`"status": "READY"`) only when the Kafka broker connection is established, the group coordinator has completed partition rebalancing, and partition 0 is actively assigned to the consumer thread.

### 3.3 Container Specification (`consumer/Dockerfile`)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
RUN pip install --no-cache-dir kafka-python==2.0.2 six
COPY consumer/streaming_consumer.py /app/streaming_consumer.py
EXPOSE 8080
ENV KAFKA_BROKER=kafka:29092 \
    TOPIC_NAME=adrift_events \
    GROUP_ID=adrift_consumer_group \
    HTTP_PORT=8080 \
    AUTO_OFFSET_RESET=earliest \
    PYTHONUNBUFFERED=1
CMD ["python", "streaming_consumer.py"]
```

---

## 4. STAGE 3 — Real Docker Recovery Actuator

### 4.1 Interface and Responsibilities (`recovery/docker_actuator.py`)

The `DockerRecoveryActuator` is decoupled from the legacy simulated `RecoveryExecutor`:
- **Real Infrastructure Operations**: Interacts directly with Docker Engine via subprocess execution (`docker inspect`, `docker start`, `docker restart`).
- **Verified Readiness Gate**: Recovery success is determined exclusively by HTTP readiness polling against `http://localhost:8080/ready`. An operation is never marked successful merely because the CLI process exited with code 0.
- **Monotonic Clock Precision**: Monotonic durations are measured with `time.perf_counter()`; ISO-8601 wall-clock timestamps are recorded for distributed log alignment.

### 4.2 Fail-Closed Semantics for Unsupported Actions

ADRIFT defines a 5-action space. The local Docker prototype implements real operations only where actual container infrastructure exists:

| Action Name | Prototype Implementation | Status Code | Rationale & Security Boundary |
| :--- | :--- | :--- | :--- |
| **Restart Consumer** | Real Docker Container Restart | `SUCCESS` / `FAILED` | Validated against `adrift-consumer` with HTTP readiness verification. |
| **Ignore Event** | Documented Verified No-Op | `NO_OP` | Action deliberately skips recovery; fault remains unaddressed in telemetry. |
| **Retrain Model** | Refused / Fail Closed | `UNSUPPORTED_ACTION` | No containerized model training pipeline in local prototype. |
| **Rollback Previous Model** | Refused / Fail Closed | `UNSUPPORTED_ACTION` | No model registry or artifact rollback actuator in local prototype. |
| **Scale Processing Resources** | Refused / Fail Closed | `UNSUPPORTED_ACTION` | Requires multi-replica orchestration (e.g. Swarm/Kubernetes); not present in single-container prototype. |

> **Critical Research Rule**: The actuator never secretly substitutes `Restart Consumer` when an unsupported action is selected. Unsupported decisions fail closed with status `UNSUPPORTED_ACTION` and `success=False`.

### 4.3 Operational Safety & Access Controls
- **Whitelisting**: Target container identifiers are strictly validated against regex `^[a-zA-Z0-9_\-]+$` and restricted to `self.allowed_containers` (defaulting strictly to `adrift-consumer`).
- **Permission Boundaries**: Unauthorized container targets are rejected immediately with `PERMISSION_DENIED`.
- **Non-Privileged Execution**: The Docker actuator executes standard client commands; it does not mount the host root filesystem or run containers in `--privileged` mode.

---

## 5. STAGE 4 — Fault Detection and Policy-in-the-Loop Integration

### 5.1 End-to-End Control Flow

```
[ Kafka Stream: topic 'adrift_events' ]
                 │
                 ▼
[ Containerized Consumer ('adrift-consumer') ] ─── Telemetry Port 8080
                 │
                 ▼ (Controlled Failure: docker stop / kill)
[ Container Offline: HTTP /ready Refuses Connections ]
                 │
                 ▼ (Observation Latency: Mean 1.09s)
[ ADRIFT Controller: Detects Failure Signal ]
                 │
                 ▼
[ State Encoder: Builds Schema-Valid 8-Feature Vector ]
  - drift_type: None (0.0)
  - severity_score: 100.0 (1.0 normalized)
  - latency: 5.0 (1.0 normalized)
  - throughput: 0.0 (0.0 normalized)
  - consumer_health: 1 (Failed)
  - success_rate: 1.0
  - drift_frequency: 0.0
  - error_rate: 0.5
                 │
                 ▼ (Decision Latency: Mean 0.05s)
[ Decision Engine: Rule Engine or PPO Policy (Seeds 42–46) ]
  - Action Selected: Action 2 ("Restart Consumer")
  - Action Probability: P(Restart) > 0.996 across all PPO policies
                 │
                 ▼
[ Docker Recovery Actuator: Genuine 'docker start' Invocation ]
                 │ (Docker Restart Duration: Mean 0.074s)
                 ▼
[ Service Readiness Wait: HTTP /ready Polling ]
                 │ (Readiness Wait: Mean 9.19s, incl. Kafka Group Join)
                 ▼
[ Verifiably Recovered: 200 OK + Resumed Message Streaming ]
```

### 5.2 Verification Across All Five PPO Checkpoints

Each checkpoint from Phase 2 was independently loaded and verified against the failure observation schema:

| Policy Checkpoint | Training Seed | Checkpoint SHA-256 | Action 2 Probability | Selected Action | Actuator Execution |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `seed_42/ppo_agent.zip` | 42 | `ba8f2ca573d6...` | **0.99791** | `Restart Consumer` | Real Docker Restart |
| `seed_43/ppo_agent.zip` | 43 | `1e93c47c0978...` | **0.99629** | `Restart Consumer` | Real Docker Restart |
| `seed_44/ppo_agent.zip` | 44 | `b3be1e873582...` | **0.99922** | `Restart Consumer` | Real Docker Restart |
| `seed_45/ppo_agent.zip` | 45 | `8bfd118731d5...` | **0.99713** | `Restart Consumer` | Real Docker Restart |
| `seed_46/ppo_agent.zip` | 46 | `fb4a0924f927...` | **0.99665** | `Restart Consumer` | Real Docker Restart |
| `RuleEngine` (Baseline) | N/A | Static Rule 4 | 1.00000 | `Restart Consumer` | Real Docker Restart |

All five PPO policies, despite random seed variations during training, decisively identify that a consumer failure requires container restart ($P > 99.6\%$).

---

## 6. STAGE 5 & 6 — Controlled Experiments and Empirical Evidence

### 6.1 Metric Definitions and Formulas

| Metric Name | Formula / Definition | Clock Source | Unit |
| :--- | :--- | :--- | :---: |
| **Observation Latency** | $t_{\text{observed}} - t_{\text{fault\_injected}}$ | Monotonic (`perf_counter`) | Seconds |
| **Decision Latency** | $t_{\text{decision\_end}} - t_{\text{decision\_start}}$ | Monotonic (`perf_counter`) | Seconds |
| **Actuator Start Latency** | $t_{\text{actuator\_start}} - t_{\text{decision\_end}}$ | Monotonic (`perf_counter`) | Seconds |
| **Docker Restart Duration** | Elapsed time of `docker start` CLI execution | Monotonic (`perf_counter`) | Seconds |
| **Readiness Wait Duration** | $t_{\text{ready\_200}} - t_{\text{docker\_cmd\_end}}$ (includes group join) | Monotonic (`perf_counter`) | Seconds |
| **End-to-End Recovery Time** | $t_{\text{ready\_200}} - t_{\text{fault\_injected}}$ | Monotonic (`perf_counter`) | Seconds |
| **Service Downtime** | Total duration service was unavailable | Monotonic (`perf_counter`) | Seconds |
| **Data Integrity Delta** | Messages Produced vs. Messages Processed & Offsets | Kafka Broker / Offsets | Integer |

### 6.2 Experiment A: Baseline Consumer Kill & Resumed Streaming
- **Procedure**: Container killed with `docker kill adrift-consumer`. Failure detected in 0.1477s. Docker restart executed in 0.0773s. Readiness confirmed after group join in 9.7105s.
- **Continuous Streaming**: 15 pre-fault messages produced; 15 post-recovery messages produced.
- **Result**: Post-recovery messages immediately processed; final offset reached 339 with 100% data integrity.

### 6.3 Experiment B: Actuator & Controller Boundary Suite Results

Raw results from `evaluation/results/camera_ready_phase4_docker/actuator_validation_results.json`:

| Test ID | Target Container / Action | Expected Status | Actual Status | Success | Verdict |
| :---: | :--- | :--- | :--- | :---: | :---: |
| **B1** | `nonexistent_container_xyz` | `CONTAINER_NOT_FOUND` | `CONTAINER_NOT_FOUND` | False | **PASS** |
| **B2** | `consumer; rm -rf /` | `INVALID_TARGET` | `INVALID_TARGET` | False | **PASS** |
| **B3** | `some_valid_looking_container` | `PERMISSION_DENIED` | `PERMISSION_DENIED` | False | **PASS** |
| **B4** | `readiness_timeout_s=0.001` | `TIMEOUT` | `TIMEOUT` | False | **PASS** |
| **B5** | `Retrain Model` | `UNSUPPORTED_ACTION` | `UNSUPPORTED_ACTION` | False | **PASS** |
| **B6** | `Rollback Previous Model` | `UNSUPPORTED_ACTION` | `UNSUPPORTED_ACTION` | False | **PASS** |
| **B7** | `Scale Processing Resources` | `UNSUPPORTED_ACTION` | `UNSUPPORTED_ACTION` | False | **PASS** |
| **B8** | `Ignore Event` | `NO_OP` | `NO_OP` | True | **PASS** |

### 6.4 Experiment C: Controlled Policy-in-the-Loop Trial Matrix (12 Trials)

Raw trial records from `evaluation/results/camera_ready_phase4_docker/raw_docker_trials.csv`:

| Trial ID | Engine / Seed | Selected Action | Status | Observation Latency (s) | Decision Latency (s) | Docker Restart (s) | Readiness Wait (s) | End-to-End Downtime (s) | Post Offset | Dropped / Duplicates |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | Rule Engine | Restart Consumer | SUCCESS | 0.6328 | 0.00001 | 0.0877 | 8.5801 | 9.3366 | 359 | 0 / 0 |
| **2** | Rule Engine | Restart Consumer | SUCCESS | 1.1326 | 0.00000 | 0.0654 | 9.7631 | 10.9923 | 379 | 0 / 0 |
| **3** | PPO Seed 42 | Restart Consumer | SUCCESS | 1.1101 | 0.50052 | 0.0952 | 9.0160 | 10.7488 | 399 | 0 / 0 |
| **4** | PPO Seed 42 | Restart Consumer | SUCCESS | 1.1366 | 0.02134 | 0.0714 | 9.0063 | 10.2620 | 419 | 0 / 0 |
| **5** | PPO Seed 43 | Restart Consumer | SUCCESS | 1.1215 | 0.01035 | 0.0663 | 9.3383 | 10.5625 | 439 | 0 / 0 |
| **6** | PPO Seed 43 | Restart Consumer | SUCCESS | 1.1532 | 0.00910 | 0.0714 | 9.2753 | 10.5375 | 459 | 0 / 0 |
| **7** | PPO Seed 44 | Restart Consumer | SUCCESS | 1.1409 | 0.01433 | 0.0786 | 9.0162 | 10.2788 | 479 | 0 / 0 |
| **8** | PPO Seed 44 | Restart Consumer | SUCCESS | 1.1380 | 0.00790 | 0.0680 | 9.3178 | 10.5569 | 499 | 0 / 0 |
| **9** | PPO Seed 45 | Restart Consumer | SUCCESS | 1.1308 | 0.00943 | 0.0680 | 9.2398 | 10.4731 | 519 | 0 / 0 |
| **10** | PPO Seed 45 | Restart Consumer | SUCCESS | 1.1290 | 0.00936 | 0.0737 | 9.1869 | 10.4265 | 539 | 0 / 0 |
| **11** | PPO Seed 46 | Restart Consumer | SUCCESS | 1.1345 | 0.01091 | 0.0745 | 9.3169 | 10.5616 | 559 | 0 / 0 |
| **12** | PPO Seed 46 | Restart Consumer | SUCCESS | 1.1364 | 0.00852 | 0.0700 | 9.2009 | 10.4421 | 579 | 0 / 0 |

### 6.5 Aggregate Statistical Summary

Computed across all 12 controlled real-container trials:

```
+-----------------------------------------------------------------------------------------+
|                               Aggregate Metrics Summary                                 |
+--------------------------+----------+----------+----------+----------+----------+-------+
| Metric                   | Mean     | Std      | Median   | Min      | Max      | Unit  |
+--------------------------+----------+----------+----------+----------+----------+-------+
| Failure Observation Time | 1.0914   | 0.1386   | 1.1336   | 0.6328   | 1.1532   | sec   |
| Decision Latency         | 0.0501   | 0.1359   | 0.0094   | 0.0000   | 0.5005   | sec   |
| Docker Restart Duration  | 0.0742   | 0.0086   | 0.0714   | 0.0654   | 0.0952   | sec   |
| Readiness Wait Duration  | 9.1881   | 0.2675   | 9.2204   | 8.5801   | 9.7631   | sec   |
| End-to-End Downtime      | 10.4315  | 0.3796   | 10.5053  | 9.3366   | 10.9923  | sec   |
+--------------------------+----------+----------+----------+----------+----------+-------+
```

### 6.6 Performance Comparison: Rule Engine vs. PPO Policies

| Engine / Agent | Mean Decision Latency (s) | Mean Docker Restart (s) | Mean Readiness Wait (s) | Mean End-to-End Downtime (s) | Success Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rule Engine** | 0.000005 | 0.0765 | 9.1716 | **10.1644** $\pm$ 0.83 | **100%** (2/2) |
| **PPO Seed 42** | 0.260930 | 0.0833 | 9.0112 | **10.5054** $\pm$ 0.24 | **100%** (2/2) |
| **PPO Seed 43** | 0.009725 | 0.0688 | 9.3068 | **10.5500** $\pm$ 0.01 | **100%** (2/2) |
| **PPO Seed 44** | 0.011115 | 0.0733 | 9.1670 | **10.4179** $\pm$ 0.14 | **100%** (2/2) |
| **PPO Seed 45** | 0.009395 | 0.0708 | 9.2133 | **10.4498** $\pm$ 0.02 | **100%** (2/2) |
| **PPO Seed 46** | 0.009715 | 0.0722 | 9.2589 | **10.5018** $\pm$ 0.06 | **100%** (2/2) |

**Key Finding**:
The container restart operation itself (`docker start`) takes merely **$\sim 74$ milliseconds**. The dominant contributor to distributed streaming recovery is the **Kafka Consumer Group partition assignment and rebalancing**, which requires **$\sim 9.19$ seconds**. In-memory simulated benchmarks that assume recovery is instantaneous or sub-second overlook this fundamental distributed systems latency.

---

## 7. STAGE 7 — Automated Reliability and Integration Tests

### 7.1 Test Suite Structure (`tests/test_docker_recovery.py`)

1. **Unit Tests (Offline, Zero Infrastructure Dependencies)**:
   - `test_container_name_regex_and_security`: Blocks command injection (`bad;name`, `cat`, backticks).
   - `test_unauthorized_container_targeting`: Rejects non-whitelisted container names (`PERMISSION_DENIED`).
   - `test_unsupported_actions_fail_closed`: Verifies `Retrain`, `Rollback`, and `Scale` fail closed.
   - `test_ignore_event_documented_noop`: Confirms `Ignore Event` records a verified no-op.
   - `test_unknown_action_fails`: Unrecognized action space strings fail closed.
   - `test_simulated_executor_separation`: Confirms strict class and runtime decoupling.
   - `test_all_five_checkpoints_exist`: Validates disk presence of seeds 42–46 checkpoints.
   - `test_inference_on_consumer_failure_state`: Validates $P(\text{Restart}) > 0.95$ across all 5 seeds.
   - `test_prior_reports_exist`: Asserts Phase 1, Phase 2, and Phase 3 reports remain intact.
   - `test_manuscript_unchanged`: Asserts `main.tex` is unaltered.
2. **Integration Tests (Live Docker Engine and Kafka Active)**:
   - `test_container_inspection`: Validates live Docker state query via `docker inspect`.
   - `test_http_readiness_and_metrics`: Validates HTTP status code 200 on `/ready` and telemetry payload on `/metrics`.
   - `test_actuator_restart_trial`: Executes an actual live restart of `adrift-consumer` and measures monotonic durations.

### 7.2 Execution Command and Results
```bash
venv/bin/python3 -m unittest discover -s tests -p "test_*.py" -v
```
**Output**:
```
Ran 39 tests in 14.327s
OK
```
All 39 tests across the entire repository (basic tests, Phase 1 Protocol B regression, Phase 2 seed robustness, Phase 3 SHAP regression, and Phase 4 Docker recovery) passed cleanly.

---

## 8. STAGE 8 — Research Integrity and Security Disclosures

1. **No Edits to Submitted Manuscript**: `main.tex`, PDF files, and legacy conference artifacts were preserved without any modification.
2. **No Data Fabrication**: Every single latency metric reported in this phase originated from monotonic clock measurements (`time.perf_counter()`) on real Docker container lifecycles.
3. **No Kubernetes or Cloud Claims**: This phase validates a **local Docker prototype** (Docker Compose with Kafka and a single containerized consumer). It does not claim Kubernetes pod orchestration, multi-replica horizontal autoscaling, or production cloud resilience.
4. **Security Limitations**: The Docker actuator requires access to the Docker daemon socket (`/var/run/docker.sock`). While strict name whitelisting is enforced, deploying actuators with daemon socket access in untrusted multi-tenant environments presents privilege escalation risks. Production implementations should use restricted container orchestrator APIs (e.g. Kubernetes RBAC) rather than raw Docker sockets.

---

## 9. Reproduction Instructions

To reproduce all Phase 4 results from scratch:

```bash
# 1. Start Kafka and ZooKeeper infrastructure
docker compose up -d zookeeper kafka

# 2. Build and start the containerized streaming consumer
docker build -t adrift-consumer:latest -f consumer/Dockerfile .
docker compose up -d consumer

# 3. Verify consumer readiness and telemetry
curl -s http://localhost:8080/ready
curl -s http://localhost:8080/metrics

# 4. Run the automated unit and integration test suite
venv/bin/python3 -m unittest discover -s tests -p "test_*.py"

# 5. Run the full Phase 4 Docker recovery experimental matrix
venv/bin/python3 evaluation/docker_recovery_runner.py --repetitions 2
```

All trial-by-trial logs and CSV summaries will be saved to:
- `evaluation/results/camera_ready_phase4_docker/raw_docker_trials.csv`
- `evaluation/results/camera_ready_phase4_docker/docker_recovery_summary.json`
- `evaluation/results/camera_ready_phase4_docker/actuator_validation_results.json`
