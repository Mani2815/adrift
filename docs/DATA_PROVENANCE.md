# ADRIFT Data Provenance and Traceability

This document provides a verifiable mapping for every published figure and table in the ADRIFT research project. It guarantees that all numerical results and visual artifacts are generated dynamically from experimental outputs rather than hardcoded assumptions.

## Figures

### Fig1_Architecture
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source Data**: Architectural diagram (No external CSV data needed)
*   **Implementation Module**: N/A

### Fig2_Workflow
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source Data**: Architectural diagram (No external CSV data needed)
*   **Implementation Module**: N/A

### Fig3_Classification_Workflow
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source Data**: Architectural diagram (No external CSV data needed)
*   **Implementation Module**: N/A

### Fig4_Severity_Workflow
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source Data**: Architectural diagram (No external CSV data needed)
*   **Implementation Module**: N/A

### Fig5_Electricity_Results
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source CSV**: `evaluation/results/dataset_drifts.csv`
*   **Source Experiment**: `evaluation/dataset_runner.py` (Electricity Dataset)
*   **Implementation Module**: `consumer/drift_classifier.py`

### Fig6_CICIDS_Results
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source CSV**: `evaluation/results/dataset_drifts.csv`
*   **Source Experiment**: `evaluation/dataset_runner.py` (CICIDS-2017 Dataset)
*   **Implementation Module**: `consumer/drift_classifier.py`

### Fig7_PPO_Evaluation
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source CSV**: `evaluation/results/comparison_report.csv`
*   **Source Experiment**: `evaluation/experiment_runner.py`
*   **Implementation Module**: `ppo/train_ppo.py`, `recovery/recovery_engine.py`

### Fig8_Dataset_Comparison
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source CSV**: `evaluation/results/dataset_drifts.csv`, `evaluation/results/dataset_throughput.csv`
*   **Source Experiment**: `evaluation/dataset_runner.py`
*   **Implementation Module**: `consumer/drift_detector.py`

### Fig9_Dataset_Pipeline
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source Data**: Architectural diagram (No external CSV data needed)
*   **Implementation Module**: N/A

### Fig10_Evaluation_Workflow
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source Data**: Architectural diagram (No external CSV data needed)
*   **Implementation Module**: N/A

### Fig11_Experimental_Setup
*   **Generation Script**: `evaluation/generate_publication_figures.py`
*   **Source Data**: Architectural diagram (No external CSV data needed)
*   **Implementation Module**: N/A

---

## Regenerated Training Curves

### ppo_reward_curve.png
*   **Generation Script**: `evaluation/regen_ieee_figures.py`
*   **Source CSV**: `logs/ppo_learning_curve.csv`
*   **Source Experiment**: `ppo/train_ppo.py`
*   **Implementation Module**: `ppo/reward_engine.py`

### ppo_success_curve.png
*   **Generation Script**: `evaluation/regen_ieee_figures.py`
*   **Source CSV**: `logs/ppo_learning_curve.csv`
*   **Source Experiment**: `ppo/train_ppo.py`
*   **Implementation Module**: `ppo/reward_engine.py`

---

## Tables

### Table I: Performance Comparison (tab:performance_comparison)
*   **Generation Script**: `evaluation/generate_tables.py`
*   **Source CSV**: `evaluation/results/comparison_report.csv`
*   **Source Experiment**: `evaluation/experiment_runner.py`
*   **Implementation Module**: `recovery/metrics_engine.py`, `evaluation/metrics.py`

### Table II: Experimental Dataset Statistics (tab:dataset_stats)
*   **Generation Script**: `evaluation/generate_tables.py`
*   **Source CSV**: `evaluation/results/dataset_drifts.csv`, `evaluation/results/dataset_throughput.csv`
*   **Source Experiment**: `evaluation/dataset_runner.py`
*   **Implementation Module**: `consumer/drift_detector.py`, `consumer/severity_engine.py`

### Table III: Drift Classification Distribution (tab:drift_dist)
*   **Generation Script**: `evaluation/generate_tables.py`
*   **Source CSV**: `evaluation/results/dataset_drifts.csv`
*   **Source Experiment**: `evaluation/dataset_runner.py`
*   **Implementation Module**: `consumer/drift_classifier.py`
