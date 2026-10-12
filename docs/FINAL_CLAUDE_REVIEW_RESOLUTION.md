# ADRIFT — Final Claude Review Resolution Report

Date: 2026-08-11
Manuscript: main.tex
Protocol doc: ADRIFT/docs/ppo_train_eval_protocol.md

## Issue-by-Issue Resolution

| # | Claude Issue | Verdict | Evidence Source | Action |
|---|---|---|---|---|
| 1 | PPO train/eval separation | TRUE — same framework | pipeline_env.py, experiment_runner.py | Added in-distribution limitation |
| 2 | Severity formula not reproducible | TRUE | severity_engine.py | Added exact Eq.(5) with weights |
| 3 | Uptime inconsistency Table VII vs XII | PARTIALLY TRUE — different protocols | multi_seed_evaluator.py vs per_dataset_breakdown.py | Added protocol clarification note |
| 4 | "comparable reliability" overclaim | TRUE | Wilcoxon p=0.206 | Replaced with honest non-significance wording |
| 5 | MTTR Wilcoxon p=1.0 | TRUE | wilcoxon_test.csv + code line 47-48 | Replaced with "not applicable — all diffs zero" |
| 6 | CICIDS subsampling undocumented | PARTIALLY TRUE | dataset_adapter.py sequential read | Added contiguous subset + no stratification limitation |
| 7 | Ablation overgeneralized | TRUE | detector_ablation.py: only Sudden Drift | Scoped to sudden-drift injection |
| 8 | Gradual drift unvalidated | TRUE | 0 gradual events in dataset_drifts.csv | Added limitation |
| 9 | Dataset table conflates full/evaluated | TRUE | Table had 2,830,743 but 283,074 processed | Updated caption + added Evaluated Records row |
| 10 | Reward equation incomplete | FALSE | reward_engine.py matches exactly | Added "applied additively" only |
| 11 | "Comprehensive evaluation" overclaim | TRUE | Two datasets, one ablation type | Changed to "Evaluation on two established..." |
| 12 | Global claim audit | PARTIALLY TRUE | Handled by issues 1-11 | See individual fixes |
| 13 | Compilation and audit | PASS | tectonic exit 0 | Verified |

## FINAL STATUS

PASS — NO SCIENTIFIC BLOCKERS

All genuine issues resolved or documented as limitations.
No experiments rerun. No CSVs modified. No values invented.
