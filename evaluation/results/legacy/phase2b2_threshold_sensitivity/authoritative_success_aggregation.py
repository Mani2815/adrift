import pandas as pd
import numpy as np

def calculate_authoritative_success(raw_results_df):
    """
    Reproduces the exact trial-level aggregation methodology of the ADRIFT authoritative benchmark.
    
    In original benchmark:
    - metrics_engine.py rounds success_rate = (successful_recoveries / total_recoveries) to 2 decimal places.
    - If total_recoveries == 0, success_rate = 1.0 (100%).
    - experiment_runner.py logs (success_rate * 100) per trial.
    - pandas mean() averages these 70 percentages.
    
    This function applies the exact same mathematical pipeline to the raw event-level log.
    """
    if raw_results_df.empty:
        return 100.0
        
    trial_rates = []
    trials_present = raw_results_df['trial_id'].unique()
    
    for trial_id in trials_present:
        trial_actions = raw_results_df[raw_results_df['trial_id'] == trial_id]
        
        # Mean of boolean success is equivalent to successful / total
        fraction = trial_actions['success'].mean()
        
        # Crucial rounding applied by original metrics_engine.py
        rounded_fraction = round(fraction, 2)
        trial_rates.append(rounded_fraction * 100)
        
    # The original benchmark ALWAYS runs exactly 70 trials.
    # Any trial that generated 0 actionable events (and thus has no rows in raw_results_df)
    # would have reported 100.0% success per the metrics_engine.py fallback logic.
    missing_trials_count = 70 - len(trial_rates)
    for _ in range(missing_trials_count):
        trial_rates.append(100.0)
        
    return np.mean(trial_rates)
