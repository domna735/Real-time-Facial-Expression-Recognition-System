# 04 — Metrics & Acceptance Gates

## Webcam-mini (deployment-aligned)

Primary:
- `metrics.smoothed.macro_f1_present`
- `metrics.smoothed.accuracy`

Stability:
- flip-rate / jitter metrics from `scripts/score_live_results.py`

Rule (tentative):
- Pass if smoothed macro-F1_present improves vs baseline (or non-worse with better stability)

## Offline eval-only (safety gate)

Dataset:
- `Training_data_cleaned/classification_manifest_eval_only.csv`

Primary:
- macro-F1 (raw)
- per-class F1 (watch minority)

Calibration:
- ECE/NLL (raw and TS)

Rule (tentative):
- Pass if macro-F1 drop is within tolerance (define tolerance in the report)

## Feb 2026 evidence update (KD baseline vs KD+LP)

On 2026-02-05 we ran two short-budget KD screenings and recorded gate metrics for both eval-only and ExpW. These are useful as **current baselines** for the next domain-shift steps.

Eval-only (test):

- KD baseline: raw macro-F1=0.4385411 (acc=0.5162321)
- KD + LP: raw macro-F1=0.4411229 (acc=0.5207738)

ExpW (test):

- KD baseline: raw macro-F1=0.4595847 (acc=0.6311145)
- KD + LP: raw macro-F1=0.4583109 (acc=0.6356902)

Note: these numbers come from `outputs/evals/students/*/reliabilitymetrics.json` dated 2026-02-05 (see evaluation plan for exact paths).

## Promotion policy

Only promote / deploy adapted checkpoint if:
1) Webcam-mini improves (or improves stability without harming F1), AND
2) Offline eval-only does not regress beyond tolerance, AND
3) Stability does not get worse.
