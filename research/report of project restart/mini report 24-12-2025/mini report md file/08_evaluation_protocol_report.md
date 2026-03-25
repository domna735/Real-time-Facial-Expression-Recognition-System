# 8) Evaluation Protocol Report（評估流程報告）
Date: 2025-12-24

## Goal
Define a consistent evaluation protocol for teachers, ensembles, and students.

## Datasets / splits
Common evaluation sources used in this project:
- HQ training manifest evaluation:
  - `Training_data_cleaned/classification_manifest_hq_train.csv`
- Mixed-source robustness test:
  - `Training_data_cleaned/test_all_sources.csv`
- RAF-derived benchmarks:
  - `Training_data_cleaned/rafdb_basic_only.csv`
  - additional RAF compound/RAF-ML CSVs under `Training_data_cleaned/`

## Metrics reported
Per run, store metrics in JSON files under the run folder:
- Classification metrics:
  - Accuracy
  - Macro-F1
  - Per-class F1
- Reliability / calibration metrics:
  - Negative Log-Likelihood (NLL)
  - Expected Calibration Error (ECE)
  - Brier score
  - Temperature scaling results (global temperature, post-calibration NLL/ECE)

## Where metrics come from
- Reliability computation helper:
  - `scripts/compute_reliability.py`
- Live/demo scoring helper (if used for demo logs):
  - `scripts/score_live_results.py`

## Temperature scaling
- Apply post-hoc temperature scaling on validation logits to reduce miscalibration.
- Report both raw and temperature-scaled NLL/ECE.

## Ensemble evaluation
- Evaluate ensembles in logit space:
  - weighted logit fusion
  - best weights selected from a target benchmark (e.g., `test_all_sources.csv`)

## Output artifacts per evaluation
Expected in a complete run folder:
- `history.json`
- `reliabilitymetrics.json`
- `calibration.json`
- `best.pt`

## Next steps
- Standardize one command/script to evaluate any checkpoint on a chosen manifest and always emit the same JSON schema.

---

## Feb 2026 addendum (paper-style evaluation + protocol matching)

Later work implemented an ordered offline benchmark suite and exported a single CSV for all model×dataset results:

- `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`

Protocol rule for fair paper comparison:

- Report **accuracy** when papers report accuracy (common for FER2013 / RAF-DB).
- Also report **macro-F1 + per-class F1** to expose minority-class fragility.
- When comparing to a paper, match the dataset’s **official split** (especially FER2013 public test) if the paper uses it.

Interpretation note (important): `test_all_sources.csv` is a **mixed-source stress test**. It is a good choice for:

- catching regressions (deployment-realistic mixture)
- selecting ensemble weights that do not overfit to a single dataset

But it is **not** the best choice for paper-style SOTA comparison, because the mixture composition and label noise/domain shift can dominate the score. For paper comparisons, prefer a single dataset’s fixed official test split (e.g., RAF-DB test, FERPlus test, FER2013 PublicTest/PrivateTest).

Paper comparison note (local PDFs):

- `research/final report/final report.md` → **9.3.6**

One-page paper table (Feb-2026):

- `research/paper_vs_us__20260208.md`

Supervisor-aligned analytical comparison framing:

- `research/final report/final report.md` → Section **6.1**

FER2013 official-split evaluation prerequisite:

- Requires a local `fer2013.csv` (license-restricted; not included in this repo). Use `tools/data/convert_fer2013_csv_to_manifest.py` to generate official manifests before evaluation.
