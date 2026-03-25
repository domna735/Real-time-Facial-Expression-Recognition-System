# Evaluation Plan — Field Transfer / Domain Shift

Last updated: 2026-02-05

This document defines *how we will evaluate* domain-shift improvements so results are comparable across runs.

---

## 1) Evaluation targets (what distributions we care about)

### A) Offline “broad safety” gate (must-not-regress)
- Manifest: `Training_data_cleaned/classification_manifest_eval_only.csv`
- Purpose: prevent promoting a model that overfits a narrow target buffer
- Output artifact: `outputs/evals/students/<run>/reliabilitymetrics.json`

### B) Offline “target domain” benchmark (repeatable)
- Primary: `Training_data_cleaned/expw_full_manifest.csv` (evaluate on `test`)
- Purpose: controlled in-the-wild domain shift benchmark
- Output artifacts:
  - per-run: `outputs/evals/students/<run>/reliabilitymetrics.json`
  - compare table: `outputs/evals/_compare_*_domainshift_expw_full_manifest_test.md`

### C) Live webcam behavior (deployment-facing)
- Inputs: labeled webcam runs under `demo/outputs/<run_stamp>/`
- Required artifacts:
  - `per_frame.csv`
  - `score_results.json`

---

## 2) Metrics to report (fixed set)

### Offline classification metrics
Report (at minimum):
- Accuracy
- Macro-F1
- Per-class F1
- Minority-F1 (lowest-3 classes by F1)
- Calibration (raw + temperature-scaled): ECE, NLL

### Live (real-time) behavior metrics
From `score_results.json`:
- `metrics.raw` and `metrics.smoothed`
  - accuracy
  - `macro_f1_present`
  - `minority_f1_present` (lowest-3 among present classes)
- `jitter_flips_per_min`

Stabilizer policy:
- For comparability, all “main” live runs should use the same stabilizer settings chosen in Experiment 2.

---

## 3) Pass / fail rules (pre-registered)

### Safety gate (eval-only)
A candidate method is **FAIL** if any of the following occur:
- Macro-F1 drops by more than Δ = 0.01 vs the baseline checkpoint on eval-only
- Minority-F1 (lowest-3) drops by more than Δ = 0.02 vs baseline on eval-only

(These thresholds are initial defaults; if you want stricter/looser, change them *before* running experiments.)

### Target benchmark (ExpW)
A candidate is considered a **WIN** if:
- Minority-F1 improves by at least +0.01 on ExpW test, and
- Macro-F1 does not drop by more than 0.01 on ExpW test

### Live behavior
Live runs are treated as behavior evidence (supports differ). For a “live win” claim:
- Compare on the *same* labeled session re-scored with identical labels.
- Require:
  - smoothed `macro_f1_present` improves, and
  - flips/min does not increase beyond +10%.

---

## 4) Streaming adaptation evaluation protocol (for Exp 3 / Exp 4)

### Required logging per adaptation run
Store a small `adapt_meta.json` (or include fields inside existing `eval_meta.json`) containing:
- base checkpoint path
- which parameters were unfrozen (e.g., norm affine only)
- learning rate / optimizer / steps per chunk
- micro-batch size (frames per update)
- entropy threshold $E_0$ (if used)
- sharpness-aware settings (ρ) if used
- recovery/reset settings (moving average window, $e_0$)
- what buffer source was used (which webcam run(s) or which manifest)

### Required evaluation points
- Before adaptation (baseline)
- After each adaptation chunk (or after a fixed number of chunks)

For each point:
- Run eval-only gate
- Run ExpW test

---

## 5) Result reporting convention

For every experiment, produce:
- one compare markdown table (human-readable)
- plus the underlying JSON metrics per run

This keeps the final report consistent with the repo’s evidence-first rule.

---

## 6) Current baselines (2026-02-05)

These are the most recent KD baselines evaluated on the two fixed offline gates:

Checkpoints:

- KD baseline: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/best.pt`
- KD + LP-loss: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt`

Eval-only (test):

- KD baseline: acc=0.5162321, macro-F1=0.4385411
- KD + LP: acc=0.5207738, macro-F1=0.4411229

ExpW (test):

- KD baseline: acc=0.6311145, macro-F1=0.4595847
- KD + LP: acc=0.6356902, macro-F1=0.4583109

Source artifacts are the four `outputs/evals/students/*20260205*/reliabilitymetrics.json` files created by the post-eval hook.
