# ADRIFT Severity Normalization Engine

The `SeverityEngine` calculates the severity of a concept drift event based on three core dimensions: Magnitude, Detector Consensus, and Rapidity (Frequency).

## Formula
The total severity score is the sum of three independently bounded sub-scores:
**Total Score = Magnitude Score + Consensus Score + Frequency Score**

The Total Score is rigorously bounded between `[0, 100]`.

### 1. Magnitude Score (Max 50 points)
Represents the standardized magnitude of the drift.
- **Normalization**: Z-score calculation: `|magnitude_shift| / std_dev`
- **Scaling**: `z_score * 10`
- **Thresholds**: Capped at a maximum of `50`. If `std_dev` is 0 but magnitude shift > 0, it defaults to a score of `10`.

### 2. Detector Consensus Score (Max 30 points)
Represents the agreement between multiple anomaly detectors.
- **Normalization**: Counts the number of independent detectors that triggered.
- **Scaling**: `detectors_count * 10`
- **Thresholds**: Capped at a maximum of `30`.

### 3. Frequency / Rapidity Score (Max 20 points)
Represents the instability of the data stream, penalizing drifts that occur in rapid succession.
- **Normalization**: Time elapsed in events since the last drift (`steps_since_last_drift`).
- **Scaling**: `20 - (steps_since_last_drift * 20 / 1000)`
- **Thresholds**: 
  - If steps > 1000 or no prior drift, score = `0`.
  - Capped at a maximum of `20` for highly rapid drifts.

## Severity Levels (Score Range)
The final severity score `[0, 100]` maps to discrete severity levels used by the Recovery Engine and the PPO Agent to dictate intervention aggressiveness.

- **0 - 25**: `Low`
- **26 - 50**: `Moderate`
- **51 - 75**: `High`
- **76 - 100**: `Critical`
