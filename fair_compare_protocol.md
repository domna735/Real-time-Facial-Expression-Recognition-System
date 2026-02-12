# Fair Comparison Protocol (start) — Paper Methods in This FER System

Last updated: 2026-02-08

Goal

- Create a **fair, apples-to-apples comparison** of different paper-inspired methods inside this repo by holding constant:
  - dataset + split definitions
  - label space (7 classes)
  - student backbone and input resolution
  - training budget / optimizer where applicable
  - evaluation manifests and reported metrics
  - artifact outputs (JSON + compare tables)

Important scope note

- This protocol is for **fair internal comparison** (within this codebase). It does not claim reproduction of every paper’s exact architecture/training recipe.

---

## 1) Fixed definitions (must be identical across methods)

### 1.1 Canonical label space

- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### 1.2 Student backbone (deployment model)

- Default: `mobilenetv3_large_100` (timm)
- Input: img224
- Seed: 1337 (or a fixed small set like {1337, 2026} if we later want variance)

### 1.3 Must-run evaluation manifests (the “gates”)

These are the minimum set to report for every method.

- In-domain / training mixture reference:
  - `Training_data_cleaned/classification_manifest_hq_train.csv` (use its split column)
- Offline safety gate (broad regression check):
  - `Training_data_cleaned/classification_manifest_eval_only.csv`
- Cross-dataset proxy for “in-the-wild” shift:
  - `Training_data_cleaned/expw_full_manifest.csv`

Additional single-dataset benchmarks (optional but recommended for paper comparisons)

- RAF-DB (basic 7) only:
  - `Training_data_cleaned/rafdb_basic_only.csv`
  - Optional fixed test manifest: `Training_data_cleaned/test_rafdb_basic.csv`
- FERPlus only:
  - Optional fixed test manifest: `Training_data_cleaned/test_ferplus.csv`
- FER2013 uniform 7 only:
  - Optional fixed test manifest: `Training_data_cleaned/test_fer2013_uniform_7.csv`
- AffectNet full balanced:
  - Optional fixed test manifest: `Training_data_cleaned/test_affectnet_full_balanced.csv`

Recommended “paper fairness anchor” choice (do one-by-one)

- Yes — testing these datasets one by one is a good choice **if** we keep the protocol identical (same backbone, same resolution, same training budget, same evaluation manifests, same metrics).
- Suggested order for cleanest interpretation:
  1) RAF-DB basic (often used in FER papers; relatively clean labels)
  2) FER2013 uniform 7 (classic baseline; useful for cross-paper comparison)
  3) FERPlus (good, but watch label mapping / ambiguity)
  4) AffectNet full balanced (stronger domain variety; also the most “different” from lab-like datasets)

Dataset-specific gotchas (so comparisons stay fair)

- RAF-DB basic:
  - Good as an “anchor” because many FER papers report on it.
  - Make sure we always evaluate with the same manifest/split (prefer `test_rafdb_basic.csv` for reporting).
- FER2013 uniform 7:
  - Good for paper comparisons, but preprocessing conventions vary across papers.
  - Keep our preprocessing fixed and document it (img224, same transforms) so the comparison is at least internally fair.
- FERPlus:
  - Original FERPlus has 8 emotions (includes contempt) + label distributions; many papers use special remaps.
  - In this repo, ensure we are truly evaluating the **7-class mapped** version; do not mix settings across runs.
- AffectNet full balanced:
  - More in-the-wild variation; good stress test, but cross-paper comparability is harder because papers may use different subsets, face alignment, and filtering.
  - Still valuable as a “robustness” benchmark if we keep our evaluation fixed.

---

## 2) Methods to compare (paper-inspired) — what counts as a “method” here

We compare methods by changing exactly one “idea knob” at a time, always relative to a documented baseline.

### 2.1 Baselines

- CE student training (baseline)
- KD student training (baseline)
- DKD student training (baseline)

### 2.2 Training-time regularizers / objectives

- LP-loss (Deep Locality-Preserving) — optional auxiliary loss
- NegL (complementary-label negative learning) — gated auxiliary loss

### 2.3 Domain shift methods

- Self-Learning (pseudo-label fine-tune on webcam buffer)
- Self-Learning + NegL (main track)
- DANN (domain-adversarial training with GRL)

### 2.4 Test-time adaptation (TTA)

- TENT-style: entropy minimization with restricted parameter updates
- SAR-lite: reliable-sample filtering + recovery/reset (safety rails)

---

## 3) Compute budgets (to keep comparisons fair)

We use two budgets:

### 3.1 Screening budget (fast)

- 5 epochs (or a small fixed step budget) to detect regressions quickly.
- Purpose: choose candidates and eliminate unstable configurations.

### 3.2 Full budget (final comparison)

- A fixed full training budget (to be decided once, then locked).
- Purpose: produce the final tables used in the report.

Rule

- Never compare a 5-epoch screening run to a full-budget run.

---

## 4) Required artifacts (evidence-first)

For every run that is included in a comparison table:

- Training artifact directory under `outputs/students/...` containing at least:
  - `best.pt` (or equivalent)
  - `history.json`
  - `reliabilitymetrics.json` (if training produced validation metrics)

- Evaluation artifacts under `outputs/evals/students/...`:
  - `reliabilitymetrics.json` for:
    - eval-only gate
    - ExpW
    - (optional) RAF-DB / FERPlus / FER2013 / AffectNet test

- For real-time claims (stability):
  - `demo/outputs/<run_id>/per_frame.csv`
  - `demo/outputs/<run_id>/score_results.json`

---

## 5) Reporting format (so tables are comparable)

### 5.1 Metrics to report (minimum)

- Accuracy
- Macro-F1
- Per-class F1
- Raw ECE / Raw NLL
- TS ECE / TS NLL (temperature-scaled)

### 5.2 Real-time metrics to report (minimum)

- Raw vs smoothed accuracy / macro-F1 on the labeled replay session
- Jitter flips/min

Rule

- Real-time comparisons must use the **same labeled session** and fixed demo parameters.

---

## 6) Temperature policy (to avoid unfair stability comparisons)

Because EMA/hysteresis uses probabilities, temperature scaling can change stability even if argmax is unchanged.

For any replay-based comparison, pick one policy and keep it fixed across checkpoints:

- Policy A: force `--temperature 1.0` for all checkpoints
- Policy B: force one fixed temperature $T$ for all checkpoints

Do not mix “auto from calibration.json” across checkpoints when comparing stability.

---

## 7) First milestone (start the fair comparison)

Step 1 (mandatory): replay-based A/B on one labeled session

- Compare: CE vs KD+LP vs DKD (and any other candidate)
- Output: `demo/outputs/*/score_results.json`

Step 2 (offline fairness table)

- For the same set of checkpoints, generate eval artifacts on:
  - `Training_data_cleaned/classification_manifest_eval_only.csv`
  - `Training_data_cleaned/expw_full_manifest.csv`
  - (optional) `Training_data_cleaned/test_rafdb_basic.csv`

Step 3 (paper-method queue)

- Implement/enable one method knob at a time (e.g., SAR-lite) and repeat Step 1–2.

---

## 8) Open choices (to confirm with supervisor)

- Which single-dataset benchmark should be the “paper fairness anchor” in addition to ExpW?
  - RAF-DB basic vs FERPlus vs FER2013 vs AffectNet
- What full-budget training schedule should we lock for the final comparison table?
- How many seeds (1 vs 2+) do we require before claiming a consistent effect?

Practical next step (what to do next in this repo)

1) Pick the first anchor dataset (I suggest RAF-DB basic).
2) Run evaluation for your baseline checkpoints (CE / KD / DKD) on that dataset’s fixed test manifest.
3) Freeze the resulting table format (metrics + artifact paths) and reuse it for every paper method.

---

## 9) Completed baseline offline benchmark (evidence)

Full suite (includes the two gates + the 4 single-dataset tests)

- Output directory:
  - `outputs/benchmarks/offline_suite__20260208_192604/`
- Summary table (Markdown):
  - `outputs/benchmarks/offline_suite__20260208_192604/benchmark_summary.md`
- Machine-readable index (JSON):
  - `outputs/benchmarks/offline_suite__20260208_192604/benchmark_index.json`

Models included

- Teachers (best.pt):
  - `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/best.pt`
  - `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/best.pt`
  - `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/best.pt`
- Students (auto-picked “best run” inside each folder by training `raw.macro_f1`):
  - `outputs/students/CE/*/best.pt`
  - `outputs/students/KD/*/best.pt`
  - `outputs/students/DKD/*/best.pt`

Dataset order (as executed)

1) `Training_data_cleaned/classification_manifest_eval_only.csv` (offline safety gate)
2) `Training_data_cleaned/expw_full_manifest.csv` (in-the-wild proxy)
3) `Training_data_cleaned/test_rafdb_basic.csv`
4) `Training_data_cleaned/test_fer2013_uniform_7.csv`
5) `Training_data_cleaned/test_ferplus.csv`
6) `Training_data_cleaned/test_affectnet_full_balanced.csv`

Command used

```powershell
C:/Real-time-Facial-Expression-Recognition-System_v2_restart/.venv/Scripts/python.exe scripts/run_offline_benchmark_suite.py `
  --teacher-run outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224 `
  --teacher-run outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224 `
  --teacher-run outputs/teachers/RN18_resnet18_seed1337_stageA_img224 `
  --student-group outputs/students/CE `
  --student-group outputs/students/KD `
  --student-group outputs/students/DKD `
  --manifest Training_data_cleaned/classification_manifest_eval_only.csv `
  --manifest Training_data_cleaned/expw_full_manifest.csv `
  --manifest Training_data_cleaned/test_rafdb_basic.csv `
  --manifest Training_data_cleaned/test_fer2013_uniform_7.csv `
  --manifest Training_data_cleaned/test_ferplus.csv `
  --manifest Training_data_cleaned/test_affectnet_full_balanced.csv
```
