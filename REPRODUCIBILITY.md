# ADRIFT Reproducibility Document

This document outlines the exact execution parameters required to reproduce the empirical evaluation presented in the ADRIFT study.

## Hardware Specifications
- **Operating System**: macOS
- **CPU**: (Standard Evaluation Hardware)

## Software Versions & Dependencies
- **Python**: 3.9+
- See `requirements.txt` for exact package versions, including:
  - `stable-baselines3`
  - `river`
  - `pandas`
  - `matplotlib`
  - `seaborn`

## Configuration & Seeds
All empirical evaluations and agent training utilize fixed random seeds to ensure deterministic execution unless explicitly assessing variance.
- **Global Seed**: `42`
- **Trial Seeds**: Increment by 1 per trial (e.g., `np.random.seed(trial)`)

## Execution Order & Commands

To recreate the full evaluation from scratch, execute the following commands in the root of the repository (`/ADRIFT`):

1. **Train the PPO Agent and Export Hyperparameters**
   ```bash
   python ppo/train_ppo.py
   ```
   *Expected Output*: Saved model checkpoint in `models/ppo_agent.zip`, learning curve in `logs/ppo_learning_curve.csv`, and configurations in `ppo/ppo_configuration.json` and `.tex`.

2. **Run Chronological Dataset Evaluation**
   ```bash
   python evaluation/dataset_runner.py
   ```
   *Expected Output*: `evaluation/results/dataset_drifts.csv` and `dataset_throughput.csv`.

3. **Run Multi-Scenario Experiment Simulator**
   ```bash
   python evaluation/experiment_runner.py
   ```
   *Expected Output*: `evaluation/results/results.csv` tracking MTTR, Success Rate, and Uptime across scenarios.

4. **Run Detector Ablation**
   ```bash
   python evaluation/detector_ablation.py
   ```
   *Expected Output*: `evaluation/results/detector_ablation.csv` and corresponding plots assessing the performance of ADWIN, DDM, KSWIN, and the fused ensemble.

5. **Run Recovery Action Analysis**
   ```bash
   python evaluation/recovery_analysis.py
   ```
   *Expected Output*: `evaluation/results/recovery_analysis.csv` distinguishing between `Observed` and `Simulated` metrics.

6. **Generate Multi-Seed Statistics**
   ```bash
   python evaluation/multi_seed_evaluator.py
   ```
   *Expected Output*: `evaluation/results/multi_seed_stats.csv`

7. **Generate Tables and Figures**
   ```bash
   python evaluation/generate_tables.py
   python evaluation/generate_publication_figures.py
   python evaluation/regen_ieee_figures.py
   ```
   *Expected Output*: All LaTeX tables (`.tex`) and high-resolution figures (`.png`, `.pdf`, `.svg`) placed in `evaluation/results/ieee_figures/` and synced with the manuscript directory.
