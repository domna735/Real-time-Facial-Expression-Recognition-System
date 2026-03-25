# Negative Learning and Complementary Learning Experiments  
### Extension Study for Real-Time FER System
## Nested Learning and Negative Learning report
---

## Abstract

This project evaluates whether auxiliary Negative Learning (NL) and Complementary Learning (NegL) improve the offline performance and calibration of a distilled real-time facial expression recognition (FER) student model. Building on a baseline CE → KD → DKD pipeline with ensemble teacher softlabels, we conducted controlled short-budget screening runs to examine the effects of NL/NegL on Accuracy, Macro-F1, Minority-F1 (lowest-3), and calibration metrics (ECE/NLL, including temperature-scaled variants).

Terminology clarification (used consistently in this repo):

- **NL** here refers to *Nested Learning* (prototype-memory auxiliary mechanism with a gate/applied fraction), not “negative learning” in the generic sense.
- **NegL** refers to an entropy-gated *complementary-label negative learning* loss (a “not-this-class” auxiliary loss).

Across KD and DKD stages, NL(proto) was stable but did not consistently outperform the baseline under the tested configurations. NegL with entropy gating showed mixed effects: high thresholds applied too sparsely to strongly influence learning, while lower thresholds increased activation but frequently worsened temperature-scaled calibration in these runs. Synergy runs (NL + NegL) improved raw calibration/loss under DKD in the tested configuration but did not translate into gains in Macro-F1 or Minority-F1.

Overall, NL/NegL as currently tuned are not ready as drop-in improvements for the existing KD/DKD pipeline based on the offline evidence available in this repo. Future work should focus on safer weighting, more reliable gating, and adding deployment-facing stability metrics (e.g., flip-rate/jitter) that are not yet measured in these experiments.

## Introduction

Real-time FER systems require not only strong offline recognition metrics but also stable, reliable behavior under deployment constraints such as lighting changes, motion blur, and frame-to-frame variation. Knowledge Distillation (KD) and Decoupled KD (DKD) provide strong baselines for training lightweight student models, but they may inherit teacher confidence/miscalibration and can be sensitive on minority classes.

This project explores two auxiliary mechanisms:
- Negative Learning (NL), implemented here as prototype memory with a gated auxiliary objective.
- Complementary Learning / NegL, implemented here as an entropy-gated complementary-label loss.

We evaluate whether NL/NegL provide measurable gains for the current MobileNetV3-Large student model trained on the ExpW 7-class label space, using the repo’s standardized offline metrics (Accuracy, Macro-F1, per-class F1 where available, Minority-F1, ECE/NLL, and temperature-scaled variants). The experiments summarized here are short-budget KD and DKD resume runs intended to quickly detect regressions and characterize gating behavior.

## 1. Problem Statement

- **Background**  
  This project trains a real-time facial expression recognition (FER) student model using a standard distillation pipeline (CE → KD → DKD) and evaluates offline accuracy/F1 plus calibration (ECE/NLL), with the goal of reliable real-time behavior.

- **KD/DKD limitations (why look beyond KD/DKD)**  
  From the internal framework notes ([research/nl_negl_plan/01_problem_map.md](../01_problem_map.md)), the motivations include:
  - KD/DKD can inherit teacher overconfidence and miscalibration, affecting reliability (ECE/NLL) even when Macro-F1 is strong.
  - Minority classes can be fragile to training dynamics and loss balancing.
  - Pure logit-level supervision may not address stability goals (e.g., output jitter) without additional mechanisms.

- **Motivation for NL / NegL**  
  - **Negative Learning (NL)** was introduced as an auxiliary mechanism (prototype memory + gating) to preserve useful structure and selectively apply additional learning signals.
  - **Complementary Learning / Negative Learning loss (NegL)** was introduced as an auxiliary “not-this-class” loss to penalize confident mistakes and improve calibration with controlled application (gating / ratios).

- **Research question**  
  Under our current student pipeline and dataset, do NL and/or NegL:
  1) Improve calibration (lower ECE / NLL), and/or
  2) Improve (or at least not harm) overall recognition quality (Accuracy, Macro-F1, per-class F1), especially minority classes,
  while staying stable and compatible with the real-time FER constraints?

---

## 2. Methods

### 2.1 Dataset

- **Sources**
  - Evidence from the manifest validation indicates the core dataset used in these experiments is **ExpW** (Expression in-the-Wild):
    - Full manifest: `Training_data_cleaned/expw_full_manifest.csv` (rows_total = 91793) (see `outputs/manifest_validation_expw_full.json`).
    - A high-quality filtered subset also exists: `Training_data_cleaned/expw_hq_manifest.csv` (rows_kept = 33375) (see `outputs/expw_hq_import_report.json`).

- **Label space**  
  The canonical label set is 7 classes (from `outputs/manifest_validation_expw_full.json`):
  - Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

- **Characteristics (as reflected in manifests)**
  - Strong class imbalance (e.g., Happy/Neutral dominate counts in ExpW splits).
  - Manifest-driven train/val/test splits for reproducibility.

- **Dataset inventory in this workspace (available)**
  The folder `Training_data_cleaned/` contains multiple datasets/manifests that could be used for training or evaluation:
  - ExpW: `expw_full_manifest.csv`, `expw_hq_manifest.csv`, and `test_expw_full.csv`
  - RAF-DB: `rafdb_basic/`, `rafdb_compound_mapped/`, plus `test_rafdb_basic.csv`
  - FERPlus / FER2013: `ferplus/`, `fer2013_uniform_7/`, plus `test_ferplus.csv`, `test_fer2013_uniform_7.csv`
  - AffectNet: `affectnet_full_balanced/`, plus `test_affectnet_full_balanced.csv`
  - Multi-source evaluation manifests exist (e.g., `test_all_sources.csv`)

- **What was actually used in the NL/NegL runs summarized here**
  - The runs and compare tables referenced in Section 3 use the 7-class label space shown above (Angry/Disgust/Fear/Happy/Sad/Surprise/Neutral), consistent with ExpW manifests and the per-class keys in `reliabilitymetrics.json`.
  - This report therefore treats ExpW as the evidence-backed dataset for these NL/NegL conclusions; other datasets are present in the workspace but are not the primary basis of the results tables in Section 3.

### 2.2 Baseline Models

- **Teacher models (supervision source)**  
  Student KD uses precomputed soft labels from an ensemble (documented in [research/nl_negl_plan/NL_NGEL_study.md](../NL_NGEL_study.md)). Example referenced softlabels run:
  - `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856/`

- **Student model**
  - Backbone: `mobilenetv3_large_100`
  - Input size: 224
  - Seed (in the studied runs): 1337

- **Training pipeline**
  - Stage structure: CE → KD → DKD.
  - The “screening” experiments in this report mainly use short budgets (typically **5 epochs KD** and **DKD resume runs**) to quickly detect stability/regressions.

### 2.3 Negative Learning (NL)

- **Mechanism (as implemented)**  
  NL was implemented in two variants (from [research/nl_negl_plan/NL_NGEL_study.md](../NL_NGEL_study.md)):
  - `--nl-kind proto` (stable default): prototype memory (32–64 dim) + momentum smoothing + consistency-gated auxiliary loss.
  - `--nl-kind negl_gate` (legacy): learned per-sample gate used to scale NegL weights.

- **Implementation details (key knobs used in experiments)**
  - Prototype embedding source:
    - `--nl-embed logits` (legacy; often became nearly inactive)
    - `--nl-embed penultimate` (recommended; makes gating non-degenerate)
  - Gating styles observed:
    - Threshold-based gating (`--nl-consistency-thresh ...`) often decayed to near-zero applied fraction after early epochs.
    - Top-k gating (fraction-based) keeps NL active each epoch by construction.

### 2.4 Complementary Learning (NegL)

- **Mechanism (as implemented)**  
  NegL applies an auxiliary loss using complementary labels: the sample is treated as “not class k” for selected negative classes.

- **Implementation details (key knobs used in experiments)**
  - Loss weight: `--negl-weight` (commonly 0.05 in the runs)
  - Ratio: `--negl-ratio` (commonly 0.5)
  - Gating: entropy-based (`--negl-gate entropy`) with threshold `--negl-entropy-thresh`.
  - Observed behavior: high thresholds can be very selective (e.g., in `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251228_233720/history.json`, NegL `applied_frac` reaches 0.00583 at epoch 4).

### 2.5 Evaluation Metrics

- **Offline metrics** (reported by the compare tools and reliability JSONs)
  - Accuracy
  - Macro-F1
  - Per-class F1 (tracked in some runs)
  - Minority-F1 (defined in compare tables as “lowest-3”)
  - Calibration: ECE and NLL
  - Also reported: temperature-scaled metrics (TS ECE, TS NLL)

### 2.6 Experimental Protocol Notes (what “KD5” / “DKD5” mean here)

- **KD screening runs ("KD5")**
  - These are short-budget KD-stage experiments, typically 5 epochs.
  - Per-epoch metrics and gating behavior are recorded in `outputs/students/KD/.../history.json`.
  - Final headline metrics for tables are taken from `reliabilitymetrics.json`.

- **DKD resume runs ("DKD5")**
  - These runs resume DKD from a KD-trained checkpoint.
  - In the saved `history.json` for DKD runs, the logged epochs commonly appear as 5–9 (i.e., 5 additional epochs of DKD after the KD stage).
  - For this reason, applied-fraction evidence for DKD is reported for epochs 5→9.

---

## 3. Results

This section summarizes the results we actually have in the repo, primarily from:
- `outputs/students/_compare*.md`
- `outputs/students/*/*/reliabilitymetrics.json`
- `outputs/students/*/*/history.json`

### 3.1 NL Results

**KD-stage NL (5 epochs) vs KD baseline** (baseline: `KD_20251229_182119`)

| Run | NL setting | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| KD baseline (5ep) | NL off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 | `outputs/students/_compare_kd5_vs_negl5.md` |
| KD + NL(proto) | logits embed, thr=0.2, w=0.1 | 0.729573 | 0.728076 | 0.042676 | 0.796150 | 0.694379 | `outputs/students/_compare_kd5_nlproto_vs_kd5.md` |
| KD + NL(proto) | penultimate embed, thr=0.2, w=0.1 | 0.726689 | 0.724393 | 0.039511 | 0.799723 | 0.691421 | `outputs/students/_compare_kd5_nlproto_penultimate_thr0p2_vs_kd5.md` |
| KD + NL(proto) | penultimate, fixed thr=0.05, w=0.1 | 0.721527 | 0.718989 | 0.030271 | 0.807121 | 0.686280 | `outputs/students/_compare_20260101_084847_kd5_nlproto_penultimate_fixed_thr0p05_vs_kd5.md` |
| KD + NL(proto) | penultimate, top-k=0.1, w=0.1 | 0.723015 | 0.718769 | 0.040034 | 0.809448 | 0.686940 | `outputs/students/_compare_20260101_084847_kd5_nlproto_penultimate_topk0p1_vs_kd5.md` |
| KD + NL(proto) | penultimate, top-k=0.05, w=0.1 | 0.727759 | 0.725666 | 0.037482 | 0.797487 | 0.693276 | `outputs/students/_compare_20260101_153859_kd5_nlproto_penultimate_topk0p05_w0p1_vs_kd.md` |

**Key observations (NL)**
- NL(proto) runs were **stable** (no training collapse), but in these short KD budgets they did **not** show a consistent improvement over baseline on accuracy/Macro-F1/minority-F1.
- Threshold-based NL gating tended to become effectively inactive after early epochs (seen in the study notes via `history.json` applied fraction trends).
- Top-k gating successfully keeps NL “on” every epoch, but the measured offline metrics remain mixed/mostly worse than baseline in these screening runs.

**DKD-stage NL (resume DKD) vs DKD baseline** (baseline: `DKD_20251229_223722`)

| Run | NL setting | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| DKD baseline | NL off | 0.735711 | 0.736796 | 0.034764 | 0.783468 | 0.704458 | `outputs/students/_compare_dkd5_negl_vs_dkd5.md` |
| DKD + NL(proto) | top-k=0.05, w=0.1 | 0.719807 | 0.717861 | 0.045183 | 0.844715 | 0.688264 | `outputs/students/_compare_20260101_204953_dkd5_nlproto_penultimate_topk0p05_w0p1_vs_dkd.md` |

- Under DKD, NL-only (top-k=0.05, w=0.1) produced a **large regression** in accuracy/Macro-F1/minority-F1.

### 3.2 NegL Results

**KD-stage NegL (5 epochs) vs KD baseline**

| Run | NegL setting | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| KD baseline (5ep) | NegL off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 | `outputs/students/_compare_kd5_vs_negl5.md` |
| KD + NegL | entropy gate, ent=0.7, w=0.05, ratio=0.5 | 0.722364 | 0.719800 | 0.039770 | 0.808534 | 0.682749 | `outputs/students/_compare_kd5_vs_negl5.md` |
| KD + NegL | entropy gate, ent=0.4, w=0.05, ratio=0.5 | 0.723899 | 0.720618 | 0.039708 | 0.829301 | 0.690973 | `outputs/students/_compare_20260101_084847_kd5_negl_entropy_ent0p4_vs_kd5.md` |
| KD + NegL | entropy gate, ent=0.3, w=0.05, ratio=0.5 | 0.728177 | 0.726967 | 0.046010 | 0.827339 | 0.698288 | `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p3_vs_kd.md` |
| KD + NegL | entropy gate, ent=0.5, w=0.05, ratio=0.5 | 0.726782 | 0.725032 | 0.044099 | 0.824008 | 0.690081 | `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p5_vs_kd.md` |

**DKD-stage NegL (resume DKD) vs DKD baseline**

| Run | NegL setting | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 | Evidence |
|---|---|---:|---:|---:|---:|---:|---|
| DKD baseline | NegL off | 0.735711 | 0.736796 | 0.034764 | 0.783468 | 0.704458 | `outputs/students/_compare_dkd5_negl_vs_dkd5.md` |
| DKD + NegL | entropy gate, ent=0.7, w=0.05, ratio=0.5 | 0.735060 | 0.734752 | 0.034830 | 0.792553 | 0.702431 | `outputs/students/_compare_dkd5_negl_vs_dkd5.md` |
| DKD + NegL | entropy gate, ent=0.3, w=0.05, ratio=0.5 | 0.731479 | 0.730934 | 0.041676 | 0.812235 | 0.705310 | `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p3_vs_dkd.md` |
| DKD + NegL | entropy gate, ent=0.5, w=0.05, ratio=0.5 | 0.730410 | 0.729865 | 0.035637 | 0.805373 | 0.703345 | `outputs/students/_compare_20260101_204953_dkd5_negl_entropy_ent0p5_vs_dkd.md` |

**Synergy (NL + NegL)**

| Stage | Setting | Raw acc | Raw macro-F1 | Raw ECE | Raw NLL | TS ECE | TS NLL | Minority-F1 | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| KD | NL(top-k=0.05,w=0.1) + NegL(ent=0.4,w=0.05) | 0.725155 | 0.722802 | 0.201665 | 1.579906 | 0.042345 | 0.795800 | 0.686232 | `outputs/students/_compare_20260101_153859_kd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_kd.md` |
| DKD | NL(top-k=0.05,w=0.1) + NegL(ent=0.4,w=0.05) | 0.733712 | 0.733798 | 0.202779 | 1.412536 | 0.037443 | 0.786831 | 0.701544 | `outputs/students/_compare_20260101_204953_dkd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_dkd.md` |

**Key observations (NegL)**
- High entropy threshold (0.7) is selective in KD: in `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251228_233720/history.json`, NegL `applied_frac` drops from 0.03196 (epoch 0) to 0.00583 (epoch 4), and it did not improve baseline metrics.
- Lower thresholds (0.3–0.5) increased application rate, but the offline metric gains are still **not consistent** (TS calibration often worsened).
- Synergy runs showed **some improvement in raw calibration/loss** in DKD (lower Raw ECE and Raw NLL than DKD baseline), but did not improve accuracy/Macro-F1/minority-F1.

---

### 3.3 Per-class F1 Snapshots (real saved metrics)

Per-class F1 values below are pulled directly from each run’s `reliabilitymetrics.json`:

| Run | Raw acc | Raw macro-F1 | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KD baseline (`KD_20251229_182119`) | 0.728363 | 0.726648 | 0.7301 | 0.6612 | 0.7444 | 0.7620 | 0.7148 | 0.7581 | 0.7161 |
| DKD baseline (`DKD_20251229_223722`) | 0.735711 | 0.736796 | 0.7436 | 0.6919 | 0.7676 | 0.7604 | 0.7038 | 0.7725 | 0.7176 |
| DKD + NL only (`DKD_20260101_204953`) | 0.719807 | 0.717861 | 0.7123 | 0.6606 | 0.7372 | 0.7486 | 0.6919 | 0.7594 | 0.7151 |
| DKD + NL+NegL (`DKD_20260101_221602`) | 0.733712 | 0.733798 | 0.7342 | 0.6727 | 0.7627 | 0.7619 | 0.7164 | 0.7732 | 0.7155 |

**Stability counterexample (legacy learned gate)**

The legacy `--nl-kind negl_gate` run referenced in the study notes has a clear collapse signature in saved metrics:
- `KD_20251229_194408` (from `reliabilitymetrics.json`): Raw acc = 0.5402, macro-F1 = 0.5204, with large per-class collapse (e.g., Angry F1 = 0.1961, Neutral F1 = 0.3563).

### 3.4 Mechanism Sanity Signals (gating / applied fraction)

These signals are taken from each run’s `history.json` and show whether NL / NegL is actually being applied during training.

| Run (stage) | Mechanism | Config (from `history.json`) | Applied fraction evidence |
|---|---|---|---|
| `KD_20251230_004048` (KD) | NL(proto) | logits embed, thr=0.2, w=0.1 | NL `applied_frac`: 0.2068 (ep0) → 0.0089 (ep1) → 0.0000716 (ep4) |
| `KD_20251231_155841` (KD) | NL(proto) | penultimate embed, thr=0.2, w=0.1 | NL `applied_frac`: 0.0417 (ep0) → 0.000138 (ep1) → 0.0 (ep4) |
| `KD_20260101_091806` (KD) | NL(proto) | penultimate embed, top-k=0.1, w=0.1 | NL `applied_frac`: 0.109375 every epoch (ep0–ep4) |
| `KD_20260101_153900` (KD) | NL(proto) | penultimate embed, top-k=0.05, w=0.1 | NL `applied_frac`: 0.0625 every epoch (ep0–ep4) |
| `DKD_20260101_204953` (DKD) | NL(proto) | penultimate embed, top-k=0.05, w=0.1 | NL `applied_frac`: 0.0625 every logged epoch (ep5–ep9) |
| `KD_20251228_233720` (KD) | NegL | entropy gate, ent=0.7, w=0.05, ratio=0.5 | NegL `applied_frac`: 0.03196 (ep0) → 0.00583 (ep4) |
| `KD_20260101_094542` (KD) | NegL | entropy gate, ent=0.4, w=0.05, ratio=0.5 | NegL `applied_frac`: 0.16326 (ep0) → 0.04052 (ep4) |
| `DKD_20260101_212203` (DKD) | NegL | entropy gate, ent=0.3, w=0.05, ratio=0.5 | NegL `applied_frac`: 0.08850 (ep5) → 0.02854 (ep9) |
| `DKD_20260101_221602` (DKD) | NL + NegL | top-k=0.05 (NL) + ent=0.4 (NegL) | NL `applied_frac`: 0.0546875 constant (ep5–ep9); NegL `applied_frac`: 0.05647 (ep5) → 0.01491 (ep9) |


---

## 4. Analysis

This analysis is grounded in the repo’s “assumption check” summary ([research/nl_negl_plan/06_assumption_check_and_next_steps_2026-01-02.md](../06_assumption_check_and_next_steps_2026-01-02.md)) and the measured compare tables.

- **Interpretation of NL results**
  - Threshold-based NL(proto) frequently becomes inactive after early epochs (applied fraction decays), meaning it cannot influence later-stage performance.
  - Top-k NL gating solves the “inactive” problem, but in DKD it caused a large regression at the tested weight (w=0.1, top-k=0.05), suggesting the NL auxiliary objective is too strong or misaligned for DKD stage behavior.
  - The embedding source matters: logits-based embeddings can be too “easy” (high similarity), while penultimate features activate NL earlier, but still may collapse to near-zero once prototypes align.

- **Interpretation of NegL results**
  - Entropy-gated NegL is sensitive to the entropy threshold:
    - ent=0.7 often applies to too few samples to move metrics.
    - lower thresholds apply more broadly, but may degrade TS calibration and/or minority-F1 depending on stage.
  - In DKD synergy runs, Raw ECE/NLL improved while F1 did not, suggesting NegL may change confidence distribution without translating into improved classification boundaries under the current settings.

- **Dataset-level factors**
  - ExpW is strongly imbalanced; majority classes (Happy/Neutral) dominate. This can make auxiliary negative signals risky: they may disproportionately affect already-weak minority decision regions.
  - If the current evaluation is mostly in-domain and baseline is already strong, short-run 5-epoch screens may be underpowered to reveal small improvements.

- **Model-level factors**
  - Student capacity (MobileNetV3) may be sensitive to additional auxiliary losses; the DKD regression with NL suggests careful weighting is required.
  - Teacher softlabels may already encode strong structure; adding NegL can conflict with teacher targets (especially when gating selects uncertain regions that overlap minority confusions).

- **Why methods did not match the scenario (current evidence)**
  - The intended motivations include real-time stability (flip-rate/jitter), but current “success” judgments here are mainly offline F1 + calibration. It is possible NL/NegL could affect real-time stability without improving Macro-F1 in short runs, but that is not yet measured in these experiments.
  - Several tested settings are not yet in a “safe regime” (notably DKD+NL top-k at w=0.1), so conclusions about NL’s potential may be premature until a non-regressing configuration is found.

- **Limitations (grounded in what was actually run)**
  - Many results here are short-budget (5-epoch) KD and DKD-resume runs; small improvements may not reliably appear at this budget.
  - Runs referenced in tables use a single documented seed (1337) in the studied commands and run folders.
  - Reported metrics are offline (Accuracy/Macro-F1/ECE/NLL); real-time stability signals (flip-rate/jitter) are not yet part of the measured results included here.
  - Conclusions are evidence-backed primarily on ExpW (7-class label space); other datasets exist in `Training_data_cleaned/` but are not the dominant evidence basis for these NL/NegL tables.

---

## 5. Conclusion

- **Summary of findings**
  - NL(proto) is generally **stable** in KD screening runs, but did not yield consistent improvements over baseline metrics in the current short budgets.
  - NL-only under DKD with (top-k=0.05, w=0.1) is **not safe** (large regression in acc/Macro-F1/minority-F1).
  - NegL with entropy gating does not show a consistent offline win; high thresholds apply too rarely, and lower thresholds frequently worsen temperature-scaled calibration in these runs.
  - NL+NegL synergy can improve **raw** calibration/loss under DKD in the tested configuration, but does not improve Macro-F1/Minority-F1.
  - The evidence in this report is based on offline evaluation (Accuracy/F1 + calibration); real-time stability metrics (flip-rate/jitter) are not yet measured here.

- **Suitability of NL / NegL (current state)**
  - As currently tuned, NL/NegL are **not ready** as a drop-in improvement to the KD/DKD pipeline for offline performance metrics.
  - NegL may still be promising for calibration, but requires safer tuning and evaluation protocols aligned with deployment goals.

- **Key takeaways**
  - Gating behavior (inactive vs always-on) is a first-order issue for NL.
  - NegL effectiveness is highly threshold/weight dependent.
  - “Synergy” is not a free win; stability and non-regression must be established first.

---

## 6. Future Work

- **Make NL safe before expecting gains**
  - Reduce NL weight (e.g., try w=0.02–0.05) while keeping top-k gating, especially in DKD.
  - Re-check `history.json` NL applied fraction and ensure it stays active without dominating gradients.
  - Explore alternative gating strategies (e.g., confidence-based or adaptive thresholds) and verify with applied-fraction logs.

- **NegL tuning in a controlled sweep**
  - Keep entropy gating but sweep weights (e.g., w=0.01, 0.02, 0.05) at ent=0.3 or ent=0.4.
  - Consider class-aware ratios (see [research/nl_negl_plan/neglrules.md](../neglrules.md)) to protect minority recall.
  - Evaluate whether NegL effects become clearer under longer training budgets than the current 5-epoch screens.

- **Align evaluation with real-time goals**
  - Add deployment-facing metrics from demo logs (flip-rate, confidence stability, “confident wrong” rate) and use them as primary success signals when appropriate (see [research/nl_negl_plan/04_metrics_acceptance.md](../04_metrics_acceptance.md)).

- **Test where gains are more plausible (domain shift)**
  - Evaluate existing checkpoints on out-of-domain splits (e.g., ExpW full test manifest) using the existing `scripts/run_domain_shift_eval_oneclick.ps1` to see if NL/NegL improve robustness/minority stability under shift.

- **Synergy only after both components are non-regressing**
  - Re-test NL+NegL only once NL-only and NegL-only are each stable and non-regressing in the same stage.
