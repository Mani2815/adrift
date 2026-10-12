# ADRIFT Final Post-Rerun Manuscript Consistency Audit

This document serves as the final forensic consistency audit verifying that the `main.tex` manuscript is 100% synchronized with the fresh, authoritative rerun data for both the MTTR experiment and the PPO training success curves.

## 1. Fresh Experiment Values

- **Rule Engine MTTR:** 0.00 s
- **PPO MTTR:** 0.00 s
- **Rule Engine Success Rate:** 97.17%
- **PPO Success Rate:** 99.29%
- **Success Rate Wilcoxon p-value:** 0.206
- **Rule Engine Uptime:** 99.997%
- **PPO Uptime:** 99.981%
- **Uptime Wilcoxon p-value:** 3.68e-12
- **PPO Training Success Rate:** 100.0% consistently (no dips)

## 2. Old Values Discovered

During the previous round of edits, most stale values had already been updated via the `adrift_macros.tex` modifications. However, the **Abstract** still contained the phrase: 
*“the PPO agent attained a statistically significantly higher recovery success rate”*
This was identified and fixed during this audit.

## 3. Manuscript Corrections Made

The following corrections were made over the course of the rerun alignment:

1. **Results Section:** Removed claims that PPO's success rate was statistically significantly higher, reflecting the true p-value of 0.206.
2. **PPO Training Section:** Rewrote the false claim of an "~80% success rate dip" to truthfully state that success rate remained 100%, and attributed the observed dips to the reward curve.
3. **Figure Captions:** Updated the caption for the PPO curves to remove mention of the transient success rate dip.
4. **Multi-Seed Section:** Clarified that the numerically higher success rate is *not* statistically significant.
5. **Discussion Section:** Removed claims of statistical significance for success rate, maintaining that the PPO agent demonstrated high numerical reliability.
6. **Conclusion Section:** Replaced the claim of a "statistically significantly higher success rate" with a numerical comparison and the p-value.
7. **Abstract:** (Fixed during this final audit) Removed the "statistically significantly higher recovery success rate" phrasing.

## 4. Statistical Interpretation

The p-value of **0.206** for the recovery success rate dictates that the difference between the PPO agent (99.29%) and the Rule Engine (97.17%) is **not statistically significant**. The manuscript has been heavily audited to ensure no claims of statistical superiority remain for this metric. The paper now responsibly frames the PPO performance as "numerically higher" or "competitive" rather than statistically superior. 

For pipeline uptime, the difference remains statistically significant (p = 3.68e-12) but is correctly characterized in the manuscript as "practically negligible" (99.997% vs 99.981%).

## 5. Figure Verification

- **Figure 5 (MTTR Comparison):** Verified. The regenerated figure displays 0.00 s explicitly without invisible/misleading bars. The caption explicitly explains that this represents "raw deterministic execution in simulation".
- **Figure 11 (PPO Training Curves):** Verified. The success curve is completely flat at 100.0%. The caption correctly distinguishes between the reward variance and the stable success rate.

## 6. Numerical Provenance

| Metric | Paper Value | Source File | Experiment | Status |
|---|---|---|---|---|
| Rule Success | 97.17% | `multi_seed_stats.csv` | 140-trial evaluation | Verified |
| PPO Success | 99.29% | `multi_seed_stats.csv` | 140-trial evaluation | Verified |
| Rule MTTR | 0.00 s | `multi_seed_stats.csv` | 140-trial evaluation | Verified |
| PPO MTTR | 0.00 s | `multi_seed_stats.csv` | 140-trial evaluation | Verified |
| Rule Uptime | 99.997% | `multi_seed_stats.csv` | 140-trial evaluation | Verified |
| PPO Uptime | 99.981% | `multi_seed_stats.csv` | 140-trial evaluation | Verified |
| Success p-value | 0.206 | `wilcoxon_test.csv` | 140-trial evaluation | Verified |
| Uptime p-value | 3.68e-12 | `wilcoxon_test.csv` | 140-trial evaluation | Verified |
| MTTR p-value | 1.0 | `wilcoxon_test.csv` | 140-trial evaluation | Verified |
| PPO Training Success | 100.0% | `ppo_learning_curve.csv` | PPO training | Verified |
| PPO Timesteps | 100,000 | `ppo_learning_curve.csv` | PPO training | Verified |

*Note: Electricity and CICIDS drift counts (82 and 369) are preserved from earlier valid dataset evaluations and were not affected by the targeted recovery rerun.*

## 7. Compilation Status

The `main.tex` file has been compiled with `tectonic`.
- **Undefined references:** 0
- **Undefined citations:** 0
- **Figure rendering:** All figures render successfully.
- **IEEE Layout:** The two-column layout is fully intact.

## 8. Remaining Warnings

There are some standard LaTeX warnings (e.g., `Underfull \hbox` and `Overfull \vbox`) native to IEEE two-column formatting, but absolutely no warnings regarding missing data, missing files, or broken references.

## 9. Remaining Blocker

None. 

## FINAL VERDICT

**PASS — MANUSCRIPT FULLY SYNCHRONIZED WITH FRESH EXPERIMENTS**
