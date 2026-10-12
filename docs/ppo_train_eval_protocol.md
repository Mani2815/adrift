# PPO Training and Evaluation Protocol — ADRIFT

## Overview

This document traces the exact PPO training and evaluation environments to
establish whether a train/evaluation separation exists, and to what degree.

---

## Training Environment

**Source file:** `ppo/train_ppo.py`, `ppo/pipeline_env.py`

The PPO agent is trained using a **custom synthetic Gymnasium environment**
(`PipelineEnv`). This environment does **not** replay recorded dataset rows.
Instead, it parametrically generates scenarios at episode reset via
`_inject_random_scenario()`, which uniformly samples from seven scenario types:

- Sudden Drift
- Gradual Drift
- Incremental Drift
- Recurring Drift
- Consumer Crash
- High Latency
- Healthy

Each scenario sets synthetic values for `severity_score`, `latency`,
`throughput`, `consumer_health`, and `error_rate` using uniform random
draws within scenario-specific ranges. No real dataset rows are used.

**Training seed:** Fixed `seed=42` for the PPO model itself.  
**Training callbacks:** `LearningCurveLogger` uses a separate `PipelineEnv()`
instance (not dataset-backed) for mid-training evaluation.  
**Timesteps:** 100,000.

---

## Evaluation Environment

**Source file:** `evaluation/experiment_runner.py`

The experiment runner evaluates both the Baseline Rule Engine and PPO Agent
across the **same seven scenario types** listed above, using:

- `FaultSimulator(scenario=scenario)` to generate 250 synthetic events per trial
- 10 trials per scenario, seeded `random.seed(trial)` and `np.random.seed(trial)` (seeds 0–9)
- 7 scenarios × 2 systems × 10 trials = 140 total trials

The PPO agent is loaded from the saved model file (`models/ppo_agent.zip`)
and queried via `PPOInferenceEngine.predict_action()`.

---

## Separation Analysis

| Dimension | Assessment |
|-----------|-----------|
| Data source | Training: `PipelineEnv` (synthetic parametric). Evaluation: `FaultSimulator` (also synthetic parametric). |
| Scenario types | **Same 7 scenario types** used in both training and evaluation. |
| Random seeds | Different: training uses seed=42; evaluation uses seeds 0–9. |
| Row-level disjointness | Not applicable — no real dataset rows used in either protocol. |
| Hold-out generalization | **Not present.** Both training and evaluation use the same synthetic scenario framework. |
| Leakage risk | Moderate: the agent is trained on the same scenario distribution it is evaluated on. The evaluation is in-distribution with respect to the training environment. |

**Conclusion:**  
The PPO training and evaluation environments are both derived from the same
parametric synthetic scenario framework. The evaluation therefore measures
in-distribution policy performance rather than generalization to held-out
scenarios or unseen datasets.

This is an expected and honest property of the prototype: the PPO agent was
trained offline on the same simulation that is used to measure its recovery
performance. The distinction from the baseline rule engine is that the rule
engine is purely deterministic (no training), so both systems face exactly
the same evaluation conditions.

---

## Training-Time Success Curve

The 100% training-time success rate (Fig. 11b) is measured by the
`LearningCurveLogger` callback during training, using `eval_env.reset()` +
`model.predict()` on a separate-but-identical `PipelineEnv` instance.
Success is defined as `latency < 2.0 AND severity_score < 50.0` at episode
end. This is a training-time metric, not the same as the post-training
experiment-runner evaluation (99.29%).

---

## Supporting Source Files

| File | Role |
|------|------|
| `ppo/train_ppo.py` | PPO training loop; seed=42; LearningCurveLogger |
| `ppo/pipeline_env.py` | Training Gymnasium env; synthetic scenario injection |
| `ppo/ppo_inference.py` | Loads trained model; queries action at evaluation time |
| `evaluation/experiment_runner.py` | Evaluation loop; same scenario types; seeds 0–9 |
| `producer/fault_simulator.py` | Generates events for evaluation scenarios |
| `evaluation/multi_seed_evaluator.py` | Aggregates results; computes Wilcoxon stats |
