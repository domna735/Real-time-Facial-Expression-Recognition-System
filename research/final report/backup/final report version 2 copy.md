# Real-time Facial Expression Recognition System: Final Report

Project Title: Real-time Facial Expression Recognition System via Knowledge Distillation and Self-Learning + Negative Learning (NegL)

Author: Donovan Ma  
Institution: HKpolyU  
Supervisor: Prof. Lam  
Report Period: Aug 2025 – Feb 2026  
Document Date: Feb 21, 2026  
Report Version: 2

---

## Abstract

This project rebuilds a reproducible real-time facial expression recognition (FER) system and evaluates both offline classification quality and deployment-facing behavior under domain shift. The system targets a canonical 7-class label space and emphasizes artifact-grounded provenance: every run produces validated manifests, checkpoints, and metrics JSONs.

Key achievements (evidence-backed):

- Unified cleaned multi-source manifest validated at 466,284 samples with 0 missing paths and 0 bad labels (artifact: `outputs/manifest_validation_all_with_expw.json`).
- Teacher backbones trained with an ArcFace-style protocol (RN18/B3/CNXT), achieving Stage-A validation macro-F1 ≈ 0.781–0.791 on a fixed split of 18,165 samples (artifacts: `outputs/teachers/*/reliabilitymetrics.json` + `outputs/teachers/*/alignmentreport.json`).
- Official FER2013 split evaluation (from `fer2013.csv`, protocol-aware): best student (DKD_20251229_223722) reaches PublicTest accuracy **0.614099** (single-crop) / **0.609083** (ten-crop) and PrivateTest accuracy **0.608247** (single-crop) / **0.612148** (ten-crop) (summary artifact: `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`).
- A 3-teacher ensemble (RN18/B3/CNXT, weights 0.4/0.4/0.2) achieves macro-F1 0.6596075 on a mixed-source benchmark of 48,928 samples (artifact: `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json`).
- A deployable student (MobileNetV3-Large) trained via CE → KD → DKD, where KD/DKD improve temperature-scaled calibration (TS ECE) but do not surpass CE macro-F1 in the main HQ-train evaluation (artifacts: `outputs/students/**/reliabilitymetrics.json`).
- January 2026 extensions include: (1) NL/NegL screening for KD/DKD with mixed or negative macro-F1 impact under short-budget settings (artifacts: `outputs/students/_compare*.md` + per-run `reliabilitymetrics.json`), and (2) a webcam scoring protocol (raw vs smoothed + flip-rate/jitter) plus a safety-gated Self-Learning + NegL loop. A 2026-02-21 A/B attempt showed that (a) adaptation runs must match preprocessing (e.g., `use_clahe`) and must avoid BatchNorm running-stat drift during small-buffer tuning, and (b) even after passing the offline eval-only gate, a first conservative adapted checkpoint regressed on the recorded-session webcam score (artifacts: `demo/outputs/20260126_205446/score_results.json`, `demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`, and `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/` + `..._20260221_211119/`).

## Executive Summary (1 page)

**One-sentence summary:** A reproducible real-time FER pipeline is implemented using a teacher→student design (KD/DKD) and evaluated with protocol-aware offline benchmarks plus deployment-facing domain shift scoring, with artifact-backed reporting throughout.

Deliverables:

- Cleaned multi-source training/evaluation pipeline using CSV manifests + validation.
- Teacher training (RN18 / B3 / CNXT) + optional ensemble for robustness.
- Student distillation pipeline (CE → KD → DKD) targeting real-time constraints (MobileNetV3-Large).
- Real-time demo with logging and deployment-facing scoring (raw vs smoothed + flip-rate/jitter).
- Evidence index for “paper vs us” comparisons and protocol-aware reporting.

Key findings (bounded by stored artifacts):

- Mixed-source and cross-domain tests expose consistent fragility in minority classes (Fear/Disgust) under domain shift.
- KD/DKD often improve temperature-scaled calibration (TS ECE / TS NLL) but do not reliably improve macro-F1 on hard gates.
- “Fair compare” depends primarily on split and test-time protocol (single-crop vs ten-crop); official splits are necessary but not sufficient.

Evidence quick links:

- Offline suite (canonical table): `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
- FER2013 official split summary (single-crop + ten-crop): `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`
- Paper comparability table: `research/paper_vs_us__20260208.md`
- Paper training/protocol checklist (hyperparameter gap analysis): `research/paper_training_recipe_checklist__20260212.md`
- Fair-compare rules: `research/fair_compare_protocol.md`
- Live demo scoring outputs: `demo/outputs/*/score_results.json`

Next steps (high-level):

1) Continue fair comparison using **official** data/splits where papers claim “official protocols”; document access constraints (license / request) and the compute/storage requirements.
2) Treat different papers as different experimental systems unless protocol details are matched (preprocessing/alignment, label mapping, splits, metrics).
3) Close gaps via controlled ablations of likely setting differences (resolution, crop/TTA, augmentation, optimizer/schedule, KD/DKD temperatures and weights).
4) Continue domain shift improvement via **safety-gated** Self-Learning + NegL, promoting checkpoints only after eval-only/ExpW gates pass.

---

## Table of Contents

1. [Introduction and Background](#1-introduction-and-background)
2. [Literature Review](#2-literature-review)
3. [Methodology](#3-methodology)
4. [Results & Analysis](#4-results--analysis)
5. [Demo and Application](#5-demo-and-application)
6. [Discussion and Limitations](#6-discussion-and-limitations)
7. [Project Timeline (updated)](#7-project-timeline-updated)
8. [Lessons Learned from Development](#8-lessons-learned-from-development)
9. [Conclusion and Next Steps](#9-conclusion-and-next-steps)
10. [References](#10-references)
11. [Appendix](#11-appendix)

---

## 1. Introduction and Background

Real-time FER aims to classify facial expressions from a live stream while meeting constraints that are not captured by typical offline benchmarks:

- Latency and throughput (FPS) constraints
- Stability (avoid flickering predictions)
- Calibration (confidence should be meaningful)
- Domain shift (webcam lighting, sensor noise, motion blur, user-specific effects)

This project adopts a teacher → student design to reconcile accuracy and speed:

- Train strong teachers offline on a validated multi-source dataset
- Optionally ensemble teachers for robustness
- Distill knowledge into a compact student suitable for real-time inference
- Evaluate both offline metrics (macro-F1, per-class F1, calibration) and real-time metrics (smoothed vs raw performance, jitter flips/min)

## 1.1 Relationship to Interim Report v4 (Dec 25, 2025)

This final report is a continuation of the project’s Dec-2025 interim deliverable:

- Interim report v4: `research/Interim Report/version 4 Real-time-Facial-Expression-Recognition-System Interim Report (25-12-2025).md`

How it is integrated here:

- The interim report’s framing (deployment constraints, domain shift risk, and the need for artifact-grounded provenance) is preserved as the motivation for the pipeline design in Sections 1–3.
- All **numeric results** in this final report are still taken directly from run artifacts (JSON/compare tables) as listed in Sections 3–4.
- The Jan-2026 extensions (NL/NegL screening, webcam scoring protocol, and offline safety gate) extend beyond the interim report’s Dec-2025 scope.

## 2. Literature Review

This literature review expands the key research themes that motivate the system design choices in this project. The goal is not to reproduce any single paper’s method in full, but to explain the technical background needed to justify: (1) why the teacher→student pipeline is used, (2) why evaluation must be protocol-aware for “fair comparison”, and (3) why domain shift and deployment stability are first-class concerns in real-time FER.

## 2.1 Problem setting: FER is noisy, imbalanced, and ambiguous

Facial expression recognition (FER) typically maps a face crop to a discrete set of emotion categories (commonly 7 basic emotions). In practice, this mapping is inherently ambiguous:

- **Subtlety and intensity variation:** low-intensity expressions can be visually close to Neutral.
- **Class overlap:** Fear vs Surprise, Disgust vs Angry, and Sad vs Neutral are common confusions in-the-wild.
- **Annotation noise:** crowd-labeled in-the-wild datasets contain label noise and context ambiguity.

Because of these factors, accuracy alone can hide important failure modes when the dataset is class-imbalanced. For this reason, macro-F1 and per-class F1 are commonly used to reveal minority-class brittleness.

## 2.2 Datasets and protocols: why “fair comparison” is hard

FER papers often report results on one curated dataset and a specific official split (or a defined cross-validation protocol). Real deployments (and this project) face a different reality:

- **Training distribution is a mixture:** multi-source data can be larger and more diverse, but also noisier.
- **Test distribution may shift:** webcam usage differs in sensor, lighting, pose, motion blur, and cropping jitter.

This makes protocol details essential for fair comparison:

- **Split definition:** official train/test splits vs folder-packaged splits.
- **Label mapping:** 7-class vs 8-class (or compound classes).
- **Test-time protocol:** single-crop vs multi-crop / test-time augmentation (TTA).
- **Preprocessing:** face alignment, cropping policy, image resolution, and histogram normalization can all change results.

In this repo, the official FER2013 PublicTest/PrivateTest evaluation is treated as the strongest anchor for “paper-like” comparison on FER2013, while mixed-source gates (eval-only / ExpW / test_all_sources) are treated as deployment-aligned stress tests.

## 2.3 Backbone architectures and attention: capacity vs efficiency vs calibration

Modern FER systems commonly use CNN backbones pretrained on large-scale image datasets. This project uses multiple backbone families for different roles:

- **Teachers (capacity-first):** ResNet-18 / EfficientNet-B3 / ConvNeXt-Tiny are used as higher-capacity feature extractors.
- **Student (efficiency-first):** MobileNetV3-Large is used for real-time constraints.

Attention mechanisms are frequently used inside modern backbones to improve representational power:

- **Channel attention (SE-style):** reweights feature channels based on global context.
- **Spatial+channel attention (e.g., CBAM-style):** can further focus on salient regions.

These modules can improve accuracy, but they also interact with calibration: models that become “sharper” can become overconfident. This is one reason calibration metrics (ECE/NLL) and temperature scaling are important when the system uses confidence thresholds (e.g., in self-learning or in UI decisions).

## 2.4 Knowledge Distillation (KD) and Decoupled KD (DKD): teacher→student transfer

Knowledge distillation trains a compact student model by combining:

- **Hard labels (cross-entropy):** encourage correct classification on ground-truth labels.
- **Soft targets (teacher probabilities/logits):** transfer teacher knowledge, including class similarity structure.

In typical KD, the teacher logits are softened with a temperature $T$ during training to provide a smoother target distribution. This can help optimization and improve generalization of a smaller student.

However, vanilla KD can have trade-offs:

- The student may inherit teacher uncertainty patterns.
- Improvements in probability quality may not translate to macro-F1 gains on hard, shifted test sets.

Decoupled KD (DKD) modifies the distillation objective so that target-class and non-target-class contributions can be weighted differently. The motivation is that a student may need strong target-class learning while still benefiting from the teacher’s non-target similarity structure.

In this repo, KD/DKD are treated as engineering tools to improve efficiency and reliability, and are evaluated on multiple manifests to understand where the trade-offs appear.

## 2.5 Calibration and temperature scaling: why probability quality matters

Calibration measures whether predicted confidence aligns with empirical correctness. Two common metrics are:

- **ECE (Expected Calibration Error):** a binned estimate of confidence vs accuracy mismatch.
- **NLL (Negative Log-Likelihood):** penalizes overconfident wrong predictions more strongly.

Temperature scaling is a simple post-hoc calibration method that rescales logits by a single scalar $T$ (without changing argmax labels). It improves probability quality and is especially relevant when:

- The system uses confidence thresholds (e.g., pseudo-label acceptance for self-learning).
- The UI or downstream logic depends on probability margins.
- Real-time smoothing/hysteresis reacts differently to “peaky” vs “flat” distributions.

In this project, calibration is reported alongside classification metrics because “real-time usable” behavior often depends on the confidence profile, not only top-1 accuracy.

## 2.6 Real-time FER: temporal stabilization and deployment KPIs

Real-time FER differs from offline image classification because predictions are made on a stream. A model that is accurate frame-by-frame can still be frustrating if it flickers rapidly between classes due to noise.

Common stabilization techniques include:

- **Exponential moving average (EMA) smoothing** over probability vectors.
- **Hysteresis** to resist switching classes unless the new class is sufficiently stronger.
- **Voting windows** over recent predictions.

These techniques change what is being optimized and what should be measured. Therefore, reporting both **raw** and **smoothed** behavior (plus stability metrics like flip-rate/jitter) is important for deployment-aligned evaluation.

## 2.7 Domain shift and adaptation: self-learning and negative learning need safety rails

Cross-domain generalization is a key challenge for FER. Domain shift can be caused by:

- camera sensor differences and compression artifacts,
- lighting and white balance,
- pose and occlusion,
- face detector/crop jitter,
- subject-specific appearance and expression style.

Two families of techniques are commonly discussed for improving robustness:

- **Test-time adaptation (TTA):** update a subset of parameters at test time using unsupervised objectives (e.g., entropy minimization on confident predictions).
- **Self-learning / pseudo-labeling:** treat confident predictions as pseudo-labels and fine-tune on them.

Both approaches can fail under label noise and distribution drift. Negative (or complementary) learning adds auxiliary objectives that discourage probability mass on likely-wrong classes, but it can also destabilize training if applied too aggressively.

For this reason, this project treats adaptation as a safety-gated process:

- Only update on high-confidence or stable samples.
- Keep parameter updates conservative.
- Require passing an offline regression gate (e.g., eval-only / ExpW) before accepting an adapted checkpoint.

This framing supports the supervisor’s emphasis on fair comparison and analytical gap explanation: improvements must be evidenced by stored artifacts, and claims must respect protocol differences.

## 3. Methodology

This section summarizes the implemented pipeline and the artifacts that make every claim traceable.

## 3.0 Pipeline overview (figure)

Figure 3.0 (System overview; all metrics are artifact-backed and tied to manifests):

```mermaid
flowchart LR
  A[Raw datasets (multiple sources)] --> B[Clean + validate manifests]
  B -->|CSV manifests| C[Teacher training (RN18/B3/CNXT)
  ArcFace-style Stage-A]
  C --> D[Teacher selection / ensemble]
  D --> E[Softlabels export]
  E --> F[Student training (MobileNetV3)
  CE → KD → DKD]
  F --> G[Offline evaluation
  (manifests + reliabilitymetrics.json)]
  F --> H[Real-time demo
  webcam loop + stabilization]
  H --> I[Live scoring
  (raw vs smoothed + flip-rate)]
  I --> J[Adaptation candidates
  (Self-learning + NegL)]
  J --> K[Offline safety gate
  (eval-only / ExpW)]
  K -->|pass| L[Promote checkpoint]
  K -->|fail| M[Reject + tighten update]
```

## 3.1 Canonical label space

All training and evaluation in this report uses a 7-class mapping:

- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## 3.2 Data cleaning and manifest validation

Unified cleaned multi-source manifest:

- `Training_data_cleaned/classification_manifest.csv`

Integrity validation artifact:

- `outputs/manifest_validation_all_with_expw.json`

Verified split/label counts (computed directly from the CSV manifests):

- Script: `scripts/summarize_manifest_counts.py`
- Outputs: `outputs/manifest_counts_summary.md` and `outputs/manifest_counts_summary.json`

Dataset usage notes (Dec-24 mini-report pack):

- `research/report of project restart/mini report 24-12-2025/mini report md file/05_dataset_usage_report.md`

### 3.2.1 Dataset provenance snapshots (Kaggle / Drive packaging)

Some datasets used in this project are downloaded via Kaggle or shared drives, and their folder packaging may not exactly match the “official split” definitions used in papers.

To make evaluation reproducible and audit-ready, the workflow snapshots the exact local dataset copies used in this workspace using:

- Script: `scripts/snapshot_dataset_provenance.py`

Each snapshot records file counts, total bytes, extension counts, and a stable SHA256 fingerprint over the relative file list (without relying on external URLs).

Snapshots generated on 2026-02-09:

- RAF-DB basic (local folder copy): `outputs/provenance/dataset_snapshot__RAFDB-basic__20260209.json`
- FER2013 folder packaging (msambare Kaggle-style folder copy): `outputs/provenance/dataset_snapshot__FER2013__20260209.json`
- ExpW (in-the-wild; local folder copy): `outputs/provenance/dataset_snapshot__Expression in-the-Wild (ExpW) Dataset__20260209.json`

FER2013 official split note (paper-comparison support):

- An official-split evaluation was performed using a local Kaggle/ICML-format `fer2013.csv` (Usage=Training/PublicTest/PrivateTest). Because it is license-restricted, the dataset is not redistributed; instead, the following derived artifacts are stored for reproducibility:
  - Derived official manifests (PublicTest/PrivateTest): `Training_data/FER2013_official_from_csv/manifest__publictest.csv` and `.../manifest__privatetest.csv`
  - Evaluation metrics artifacts: `outputs/evals/students/fer2013_official__*__*test__20260212__{singlecrop,tencrop}/reliabilitymetrics.json`
  - Consolidated summary table (protocol-aware): `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`

## 3.3 Teacher training (Stage A, img224)

Teachers are trained using an ArcFace-style protocol (margin + scale) and saved with full provenance.

Interpretation note (data):

- The Stage-A teacher runs in this report read from `Training_data_cleaned/classification_manifest.csv` (466,284 rows) and apply a source filter.
- Example (RN18): after filtering, the effective dataset is 225,629 rows, with sources `{ferplus: 138,526, affectnet_full_balanced: 71,764, rafdb_basic: 15,339}` and split sizes train=182,960 / val=18,165.
- Although `expw_hq` appears in the run’s `include_sources`, ExpW rows are not present in `source_counts_after_filter` for these Stage-A teacher runs.

Teacher run artifacts:

- `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/`
- `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/`
- `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/`

Each includes:

- `best.pt` and `checkpoint_*.pt`
- `history.json`
- `reliabilitymetrics.json` (accuracy, macro-F1, per-class F1, ECE/NLL; plus temperature-scaled metrics)
- `alignmentreport.json` (data split sizes and filtering provenance)

## 3.4 Ensemble selection and softlabels export

Teacher ensembles are evaluated by weighted logit fusion on a mixed-source benchmark. The selected ensemble is exported as softlabels for student KD/DKD.

Selected benchmark artifact (mixed-source test):

- `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json`

Softlabels directory used by student KD/DKD:

- `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856/`

## 3.5 Student training (CE → KD → DKD)

Student backbone:

- `mobilenetv3_large_100` (via timm)

HQ training manifest (verified size and splits):

- `Training_data_cleaned/classification_manifest_hq_train.csv` has 259,004 rows with split sizes train=213,144 / val=18,020 / test=27,840.

Student artifacts (Dec 2025 main run):

- CE: `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/reliabilitymetrics.json`
- KD: `outputs/students/_archive/2025-12-23/KD/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/reliabilitymetrics.json`
- DKD: `outputs/students/_archive/2025-12-23/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/reliabilitymetrics.json`

## 3.6 NL/NegL screening experiments (offline)

NL/NegL experiments are documented and indexed under:

- `research/nl_negl_plan/`
- Report: `research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md`

The screening results are summarized via:

- `outputs/students/_compare*.md`

with each compared run backed by:

- `outputs/students/*/*/reliabilitymetrics.json`

Terminology note (to avoid confusion in academic reading):

- **NL(proto)** in this repo refers to the *Nested Learning* prototype-style auxiliary mechanism used in the Jan-2026 screening runs (an auxiliary objective with a gating/applied fraction).
- **NegL** in this repo refers to an entropy-gated *complementary-label negative learning* loss (a “not-this-class” auxiliary loss).

These are distinct mechanisms; throughout this report, “NL” means Nested Learning, and “NegL” means complementary-label negative learning.

### 3.6.1 Negative Learning (NegL): objective and gating

Intended objective: improve calibration (ECE/NLL) and reduce overconfident mistakes during KD/DKD by adding a **complementary-label negative learning (NegL)** term that discourages probability mass on likely-wrong classes.

Key design choice in this repo is **gating**:

- NegL is not applied to every sample. A gate decides when NegL is active.
- Gate behavior is logged into the run’s `history.json` so it can be audited (example and numbers in Section 4.5.3).

Practical implication:

- A high entropy threshold can make NegL too selective (low `applied_frac`), reducing its effect.
- A low threshold / high weight can destabilize training and reduce macro-F1 (an instability counterexample is documented in Section 4.5.3).

### 3.6.2 Nested Learning (NL(proto)): intent and configuration transparency

In the Jan-2026 screening runs, “NL(proto)” refers to the prototype-style auxiliary signal used in the compare tables in Section 4.5.

Reproducibility rule used in this report:

- The exact NL(proto) configuration (e.g., `dim`, `m`, `thr`, and/or `top-k`) is always taken from the compare artifact line and run identifier rather than being re-stated from memory.

## 3.7 Domain shift track (webcam + real-time scoring + conservative adaptation)

Domain shift documentation and report:

- `research/domain shift improvement via Self-Learning + Negative Learning plan/`
- Report: `research/domain shift improvement via Self-Learning + Negative Learning plan/domain shift improvement via Self-Learning + Negative Learning report/domain shift improvement via Self-Learning + Negative Learning report.md`

Live scoring artifact per labeled run:

- `demo/outputs/*/score_results.json`

Offline safety gate evaluates adapted checkpoints on:

- `Training_data_cleaned/classification_manifest_eval_only.csv` (verified size: 110,333 rows)

and stores results under:

- `outputs/evals/students/*/reliabilitymetrics.json`

### 3.7.1 Domain shift improvement via Self-Learning + Negative Learning (NegL)

This extension targets **webcam domain shift** by adding a safe adaptation loop. The design principle is: only accept a target-domain update if it passes an offline regression gate.

Implemented components already evidenced in this repo:

- **Webcam measurement protocol** (raw vs smoothed metrics, jitter) stored in `demo/outputs/*/score_results.json` (reported in Section 4.6).
- **Offline safety gate** using `Training_data_cleaned/classification_manifest_eval_only.csv`, with results written to `outputs/evals/students/*/reliabilitymetrics.json` (reported in Section 4.7).

Planned (but not yet claimed as successful results here):

- Add self-learning using high-confidence pseudo-labels plus NegL for medium-confidence samples, with strict defaults and rollback, as documented in `research/domain shift improvement via Self-Learning + Negative Learning plan/`.

Executed evidence update (Feb 2026):

- A first conservative Self-Learning + manifest-driven NegL A/B attempt was executed on 2026-02-21; it passed the offline eval-only gate within rounding when preprocessing and BatchNorm behavior were controlled, but regressed on the deployment-facing same-session webcam replay score (see Section 4.9.2 for artifacts and interpretation boundaries).

## 4. Results & Analysis

All results below are copied directly from the listed artifacts.

## 4.1 Dataset integrity (multi-source)

From `outputs/manifest_validation_all_with_expw.json`:

- Total rows: 466,284
- Missing paths: 0
- Bad labels: 0

Verified distribution summaries (computed from the CSV manifests; see `outputs/manifest_counts_summary.md`):

- `Training_data_cleaned/classification_manifest.csv` split sizes: train=378,965 / val=37,862 / test=49,457.
- `Training_data_cleaned/classification_manifest_hq_train.csv` split sizes: train=213,144 / val=18,020 / test=27,840.
- `Training_data_cleaned/test_all_sources.csv` label totals show a mixed-domain benchmark with 48,928 rows and all 7 classes present.

## 4.2 Teacher performance (Stage A, img224)

Evaluation split sizes are recorded in `outputs/teachers/*/alignmentreport.json`. For example, RN18 uses `val_rows = 18165`.

Interpretation note (scope):

- These teacher metrics are computed on the **Stage-A validation split** after source filtering (typically `affectnet_full_balanced` + `ferplus` + `rafdb_basic`).
- They are **not** the same as performance on the hard/mixed-domain gates (`eval_only`, `expw_full`, `test_all_sources`) or webcam domain shift.
- See: `research/issue__teacher_metrics_interpretation__20260209.md`.
- Hard-gate benchmark (same teacher checkpoints evaluated on `eval_only` / `expw_full` / `test_all_sources`):
  - Summary table: `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`
  - Write-up note: `research/issue__teacher_hard_gates__20260209.md`

### 4.2.1 Stage-A validation (in-distribution; filtered)

Teacher metrics (Stage-A validation split; `val_rows = 18165`; from each `outputs/teachers/*/reliabilitymetrics.json`):

| Model | Eval split (n) | Accuracy | Macro-F1 | Raw NLL | TS NLL | Raw ECE | TS ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RN18 | Stage-A val (18165) | 0.786182 | 0.780828 | 4.025883 | 0.880346 | 0.205298 | 0.148851 |
| B3 | Stage-A val (18165) | 0.796091 | 0.790988 | 3.221890 | 0.787123 | 0.198786 | 0.083927 |
| CNXT | Stage-A val (18165) | 0.794055 | 0.788959 | 3.101407 | 0.769976 | 0.200896 | 0.081701 |

Per-class F1 (raw; rounded to 4 d.p.; values are copied from each `reliabilitymetrics.json`):

| Model | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RN18 | 0.7357 | 0.6940 | 0.7635 | 0.8970 | 0.7415 | 0.8186 | 0.8155 |
| B3 | 0.7521 | 0.7156 | 0.7576 | 0.9197 | 0.7479 | 0.8042 | 0.8399 |
| CNXT | 0.7687 | 0.7194 | 0.7395 | 0.9135 | 0.7313 | 0.8064 | 0.8439 |

### 4.2.2 Hard gates (domain shift / mixed-domain)

For clarity: the Stage-A validation numbers above are **not** the same evaluation as the hard-gate tests.

Here we evaluate the **same three teacher checkpoints** on the hard/mixed-domain gates (`eval_only`, `expw_full`, `test_all_sources`).

Primary artifacts:

- Summary table: `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`
- Per-eval metrics: `outputs/evals/teachers/overall__*__{eval_only,expw_full,test_all_sources}__test__20260209/reliabilitymetrics.json`
- Write-up note: `research/issue__teacher_hard_gates__20260209.md`
- Generator script: `scripts/run_teacher_overall_summary_table.py` (runs `scripts/train_teacher.py --evaluate-only` under the hood)

Headline results (raw metrics; copied from the above artifacts):

| Gate dataset | n (eval rows) | RN18 acc | RN18 macro-F1 | B3 acc | B3 macro-F1 | CNXT acc | CNXT macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eval_only | 11890 | 0.427250 | 0.372670 | 0.470816 | 0.392831 | 0.440875 | 0.388980 |
| expw_full | 9179 | 0.498747 | 0.374009 | 0.584486 | 0.406649 | 0.511167 | 0.382112 |
| test_all_sources | 48928 | 0.644723 | 0.617067 | 0.674297 | 0.645421 | 0.664936 | 0.638065 |

## 4.3 Ensemble robustness benchmark (mixed-source)

Benchmark manifest size (verified): `Training_data_cleaned/test_all_sources.csv` has 48,928 rows.

Selected ensemble benchmark result (from `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json`):

- Weights: RN18/B3/CNXT = 0.4/0.4/0.2
- Accuracy: 0.687255
- Macro-F1: 0.6596075

Additional metrics (same artifact):

- NLL: 4.077156
- ECE: 0.287694
- Brier: 0.590869

Per-class F1 (rounded to 4 d.p.; copied from `ensemble_metrics.json`):

| Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6304 | 0.6034 | 0.5350 | 0.8389 | 0.5924 | 0.7205 | 0.6967 |

## 4.4 Student performance (HQ-train evaluation)

Student metrics are taken from the Dec 2025 CE/KD/DKD run artifacts listed in Section 3.5.

| Student stage | Accuracy | Macro-F1 | Raw NLL | TS NLL | Raw ECE | TS ECE | Global T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.750174 | 0.741952 | 1.315335 | 0.777757 | 0.131019 | 0.049897 | 3.228 |
| KD | 0.734688 | 0.733351 | 2.093148 | 0.768196 | 0.215289 | 0.027764 | 5.000 |
| DKD | 0.737432 | 0.737511 | 1.511788 | 0.765203 | 0.209450 | 0.026605 | 3.348 |

Per-class F1 (raw; rounded to 4 d.p.; copied from each run’s `reliabilitymetrics.json`):

| Stage | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.7263 | 0.6428 | 0.7640 | 0.8014 | 0.7170 | 0.7871 | 0.7550 |
| KD | 0.7237 | 0.6782 | 0.7447 | 0.7610 | 0.7234 | 0.7801 | 0.7224 |
| DKD | 0.7255 | 0.6828 | 0.7561 | 0.7596 | 0.7286 | 0.7915 | 0.7185 |

Interpretation:

- CE provides the best raw macro-F1 in this run.
- KD/DKD improve temperature-scaled calibration (TS ECE) substantially, but do not improve macro-F1 under this configuration.

## 4.4.1 Overall sanity table (CE vs KD vs DKD across hard gates)

To provide a single consolidated view of offline performance without over-claiming comparability to any single paper protocol, a consolidated table is generated across four stress-test manifests:

- `Training_data_cleaned/classification_manifest_eval_only.csv`
- `Training_data_cleaned/expw_full_manifest.csv`
- `Training_data_cleaned/test_all_sources.csv`
- `Training_data/fer2013_folder_manifest.csv` (folder split; not official FER2013)

Interpretation note (protocol):

- We also ran **official FER2013** PublicTest/PrivateTest evaluation from `fer2013.csv` (Usage=PublicTest/PrivateTest). Those results are reported separately under the Feb-2026 paper-comparison addendum (Section 9.3.6) because they are a different protocol from the folder split.

Artifact (single-page table; derived from on-disk `reliabilitymetrics.json`):

- `outputs/benchmarks/overall_summary__20260208/overall_summary.md`

Summary (bounded by the table):

- On the mixed-source gates (`eval_only`, `expw_full`, `test_all_sources`), CE has the best raw macro-F1 among the three checkpoints in this snapshot.
- On `fer2013_folder`, KD has the best raw macro-F1, while DKD has the best TS ECE.
- The consistent fragility remains minority classes (Fear/Disgust), supporting the domain-shift + label-noise framing rather than a conclusion that the model is fundamentally weak.

## 4.5 NL/NegL screening results (Jan 2026, offline)

The full experimental matrix and analysis are in `research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md`. This section expands the key ideas and provides a larger, fully verified snapshot from the repo’s compare artifacts.

Interpretation note (protocol):

- All tables below are copied from `outputs/students/_compare*.md` outputs (which also embed the exact run directories used).
- These are short-budget screening runs (KD 5 epochs; DKD resume runs) intended to detect regressions and characterize gating behavior, not to claim a final “best” configuration.

### 4.5.1 KD-stage (5 epochs) comparisons

Baseline KD vs NegL (high entropy threshold; selective gate):

Source: `outputs/students/_compare_kd5_vs_negl5.md`

| Run | NegL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep): `KD_20251229_182119` | off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 |
| KD+NegL (5ep): `KD_20251228_233720` | entropy gate (w=0.05, ratio=0.5, ent=0.7) | 0.722364 | 0.719800 | 0.039770 | 0.808534 | 0.682749 |

NegL “bite” example (lower entropy threshold; higher activation but TS calibration worsens here):

Source: `outputs/students/_compare_20260101_153859_kd5_negl_entropy_ent0p3_vs_kd.md`

| Run | NegL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep): `KD_20251229_182119` | off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 |
| KD+NegL (5ep): `KD_20260101_165108` | entropy gate (w=0.05, ratio=0.5, ent=0.3) | 0.728177 | 0.726967 | 0.046010 | 0.827339 | 0.698288 |

NL(proto) example (stable, but no clear improvement at these settings):

Source: `outputs/students/_compare_kd5_nlproto_vs_kd5.md`

| Run | NL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep): `KD_20251229_182119` | off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 |
| KD+NL(proto) (5ep): `KD_20251230_004048` | proto (dim=32, m=0.9, thr=0.2, w=0.1) | 0.729573 | 0.728076 | 0.042676 | 0.796150 | 0.694379 |

### 4.5.2 DKD-stage (resume) comparisons

Baseline DKD vs NegL (entropy gate ent=0.7):

Source: `outputs/students/_compare_dkd5_negl_vs_dkd5.md`

| Run | NegL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DKD baseline: `DKD_20251229_223722` | off | 0.735711 | 0.736796 | 0.034764 | 0.783468 | 0.704458 |
| DKD+NegL: `DKD_20251229_230501` | entropy gate (w=0.05, ratio=0.5, ent=0.7) | 0.735060 | 0.734752 | 0.034830 | 0.792553 | 0.702431 |

NL-only DKD example (regression at the tested weight):

Source: `outputs/students/_compare_20260101_204953_dkd5_nlproto_penultimate_topk0p05_w0p1_vs_dkd.md`

| Run | NL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DKD baseline: `DKD_20251229_223722` | off | 0.735711 | 0.736796 | 0.034764 | 0.783468 | 0.704458 |
| DKD+NL(proto): `DKD_20260101_204953` | proto (top-k=0.05, w=0.1) | 0.719807 | 0.717861 | 0.045183 | 0.844715 | 0.688264 |

Synergy example (DKD + NL + NegL; raw calibration improves but F1 does not):

Source: `outputs/students/_compare_20260101_204953_dkd5_nlproto_topk0p05_plus_negl_entropy_ent0p4_vs_dkd.md`

| Run | NL + NegL | Raw acc | Raw macro-F1 | Raw ECE | Raw NLL | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DKD baseline: `DKD_20251229_223722` | off | 0.735711 | 0.736796 | 0.211901 | 1.475317 | 0.034764 | 0.783468 | 0.704458 |
| DKD+NL+NegL: `DKD_20260101_221602` | NL(proto) + NegL(ent=0.4) | 0.733712 | 0.733798 | 0.202779 | 1.412536 | 0.037443 | 0.786831 | 0.701544 |

### 4.5.3 Mechanism sanity signals and failure modes

NegL gating example (why ent=0.7 can be too selective):

- Source: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251228_233720/history.json`
- NegL `applied_frac` starts at 0.031961 (epoch 0) and drops to below 1% by epoch 1 (0.00970), reaching 0.00583 by epoch 4.

Legacy instability counterexample (learned NegL gate interaction):

- Source: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_20251229_194408/reliabilitymetrics.json`
- Raw accuracy drops to 0.5402 with macro-F1 0.5204 (notably Angry F1 = 0.1961 and Neutral F1 = 0.3563), indicating an unstable configuration under that variant.

Summary conclusion (evidence-backed across the existing compare tables):

- Under the tested short-budget configurations, NL/NegL does not provide a consistent macro-F1 or minority-F1 improvement over KD/DKD baselines; outcomes are sensitive to gating/weighting and can regress TS calibration.

### 4.5.4 Feb 2026 addendum: NL/NegL status (what the offline evidence supports)

Building on the baseline CE → KD → DKD pipeline (ensemble-teacher softlabels), we conducted controlled short-budget screening runs to test whether Negative Learning (NL) and Complementary/Negative Learning variants (NegL) can improve offline accuracy, macro-F1, minority-F1 (lowest-3), and calibration (ECE/NLL, including temperature-scaled variants).

Across KD and DKD stages, the stored comparison tables show:

- **NL(proto)** was generally stable but did not consistently outperform the baseline under the tested configurations.
- **NegL with entropy gating** showed mixed effects: high entropy thresholds can apply too sparsely to strongly influence learning, while lower thresholds increase activation but can worsen temperature-scaled calibration in these runs.
- **Synergy (NL + NegL)** can improve raw loss/calibration signals under DKD in some settings, but did not translate into clear gains in macro-F1 or minority-F1 in the recorded comparisons.

Overall (bounded by the on-disk evidence in `outputs/students/_compare*.md`), NL/NegL as currently tuned are **not ready as drop-in improvements** for the existing KD/DKD pipeline.

Future work should focus on safer weighting and more reliable gating (including reporting/monitoring the NegL activation rate), and adding deployment-facing stability metrics (e.g., flip-rate/jitter) to these experiments.

## 4.6 Domain shift: live webcam scoring results (Jan 2026)

Live scoring artifacts:

- Baseline labeled run: `demo/outputs/20260126_205446/score_results.json`
- Later labeled run: `demo/outputs/20260126_215903/score_results.json`

Key deployment-aligned metrics (raw vs smoothed):

| Run | Raw acc | Raw macro-F1 (present) | Raw minority-F1 (lowest-3) | Smoothed acc | Smoothed macro-F1 (present) | Smoothed minority-F1 (lowest-3) | Jitter flips/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260126_205446 | 0.528406 | 0.472101 | 0.156207 | 0.587908 | 0.524829 | 0.160927 | 14.8644 |
| 20260126_215903 | 0.463812 | 0.493257 | 0.336913 | 0.513931 | 0.555236 | 0.361829 | 17.9104 |

Per-class highlights (smoothed metrics; includes supports so interpretation is distribution-aware):

| Run | Fear F1 | Fear support | Sad F1 | Sad support |
| --- | ---: | ---: | ---: | ---: |
| 20260126_205446 | 0.0000 | 393 | 0.0301 | 542 |
| 20260126_215903 | 0.3432 | 1316 | 0.4372 | 1540 |

Interpretation constraints:

- The two runs have different per-class supports (emotion mix), so they should be treated as behavior evidence, not a strict A/B comparison.

### 4.6.1 Feb 2026 addendum: qualitative real-time checkpoint preference (objective mismatch)

Deployment-facing qualitative observation (not a controlled, artifact-backed comparison):

- During informal interactive webcam use, the **CE** student checkpoint (`outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`) appeared more stable (fewer visibly frequent class switches) than the tested KD+LP and DKD checkpoints under the same demo pipeline settings.

Why this does not necessarily contradict offline ranking:

- Offline evaluation optimizes fixed-split metrics (macro-F1, per-class F1, calibration).
- Real-time user experience optimizes a different objective: “looks correct + doesn’t flicker” under webcam noise and face-crop jitter.
- The demo stabilizer (`demo/realtime_demo.py`) operates on *probabilities* (EMA + hysteresis + optional voting). Even when argmax labels are similar, different **probability margins** can produce different flicker behavior.
- Temperature scaling (`logits / T`) is applied in the demo (often loaded from `calibration.json` next to checkpoints). While it does not change argmax, it changes probability sharpness and therefore interacts with EMA/hysteresis.

Working hypothesis (deployment interpretation): apparent instability under webcam domain shift

- The risk discussed here is **not** classic training-domain overfitting (memorization). Given dataset scale and the observed offline behavior, that is not the primary suspicion.
- Instead, the observed real-time instability is consistent with a **structural** issue in the teacher → student chain under **target shift**:
  - KD/DKD are trained to match teacher distributions.
  - Under webcam-like domain shift (lighting/blur/compression/crop jitter, low expression intensity), teachers can become more uncertain and produce softer targets.
  - The student can **inherit that uncertainty**, leading to smaller top-1 vs top-2 margins and more near-ties.
  - Smaller margins interact poorly with EMA/hysteresis (probability-based switching), making flicker more visible even if offline macro-F1 and calibration are acceptable.

Proposed controlled evaluation protocol (to convert qualitative observations into artifact-backed evidence):

1) Record one labeled webcam session and replay/score it for each checkpoint.
2) Keep demo parameters fixed (detector, `ema_alpha`, `hysteresis_delta`, vote settings, CLAHE).
3) Use a comparable temperature policy across checkpoints (e.g., force `--temperature 1.0` for all, or force one fixed $T$ for all).
4) Store run artifacts under `demo/outputs/<run_id>/` and compare `score_results.json` (raw vs smoothed metrics + jitter flips/min).

Detailed deployment-facing note and hypotheses are tracked in: `research/Real time demo/real time demo report.md`.

## 4.7 Domain shift: conservative adaptation and offline safety gate

Offline eval-only safety gate artifacts:

- Eval-only manifest: `Training_data_cleaned/classification_manifest_eval_only.csv`
- Baseline: `outputs/evals/students/_baseline_CE20251223_eval_only_test/reliabilitymetrics.json`
- Head-only: `outputs/evals/students/FT_webcam_head_20260126_1__classification_manifest_eval_only__test__20260126_215358/reliabilitymetrics.json`
- BN-only: `outputs/evals/students/FT_webcam_bn_20260126_1_eval_only_test/reliabilitymetrics.json`

Table 4.7-1 (Offline safety gate). Dataset: `Training_data_cleaned/classification_manifest_eval_only.csv`. Split: `test` (as evaluated in the listed `reliabilitymetrics.json`). Protocol: single-crop. Evidence: metrics artifacts listed above.

| Model | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline (CE20251223) | 0.567368 | 0.485878 | 0.059181 | 1.228754 | baseline |
| Head-only FT | 0.548024 | 0.450845 | 0.060011 | 1.289419 | fail (macro-F1 drop) |
| BN-only FT | 0.548612 | 0.451277 | 0.060566 | 1.289044 | fail (macro-F1 drop) |

Interpretation:

- Both head-only and BN-only fine-tuning (as configured in these **early Jan-2026 FT runs**) reduce offline macro-F1 on a broader eval-only distribution; these checkpoints should not be promoted beyond experiments.
- Note: Section 4.9.2 documents a later 2026-02-21 adaptation attempt where preprocessing (`use_clahe`) and BatchNorm running-stat updates are controlled; that later candidate can pass the offline gate within rounding but still fails the deployment-facing webcam A/B.

## 4.8 Domain shift: ExpW cross-dataset evaluation (Jan 2026)

This subsection reports a controlled **cross-dataset** evaluation on ExpW (static images). It complements the webcam live-scoring evidence in Section 4.6 by providing a repeatable, manifest-defined “in-the-wild” test.

Source artifact (copied table):

- `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md`

Table 4.8-1 (ExpW cross-dataset gate). Dataset: `Training_data_cleaned/expw_full_manifest.csv`. Split: `test`. Protocol: single-crop. Evidence: `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md` (and per-run `reliabilitymetrics.json` inside each `Run dir`; `n` is recorded in those artifacts).

Results table (values copied from the compare artifact; run directories are written as repo-relative paths for portability):

| Label | Mode | Epochs | NegL | NL | Raw acc | Raw macro-F1 | Raw ECE | Raw NLL | TS ECE | TS NLL | Minority-F1 (lowest-3) | Run dir |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722__expw_full_manifest__test__20260119_170620 | dkd | - | off | off | 0.622944 | 0.459529 | 0.254957 | 1.820043 | 0.036783 | 1.102717 | 0.259545 | `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_DKD_20251229_223722__expw_full_manifest__test__20260119_170620` |
| mobilenetv3_large_100_img224_seed1337_DKD_20260101_204953__expw_full_manifest__test__20260119_170620 | dkd | - | off | off | 0.579039 | 0.431158 | 0.236954 | 1.633517 | 0.025713 | 1.194914 | 0.249118 | `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_DKD_20260101_204953__expw_full_manifest__test__20260119_170620` |
| mobilenetv3_large_100_img224_seed1337_DKD_20260101_212203__expw_full_manifest__test__20260119_170620 | dkd | - | off | off | 0.616843 | 0.451807 | 0.248753 | 1.733742 | 0.036499 | 1.116522 | 0.250905 | `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_DKD_20260101_212203__expw_full_manifest__test__20260119_170620` |
| mobilenetv3_large_100_img224_seed1337_DKD_20260101_214949__expw_full_manifest__test__20260119_170620 | dkd | - | off | off | 0.615644 | 0.450946 | 0.246569 | 1.723473 | 0.032534 | 1.119614 | 0.251363 | `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_DKD_20260101_214949__expw_full_manifest__test__20260119_170620` |
| mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602__expw_full_manifest__test__20260119_170620 | dkd | - | off | off | 0.617714 | 0.452596 | 0.241540 | 1.721193 | 0.041437 | 1.120937 | 0.252719 | `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602__expw_full_manifest__test__20260119_170620` |

Interpretation (evidence-limited to the table above):

- On this ExpW test, `DKD_20251229_223722` is the strongest among the listed checkpoints by raw macro-F1 (0.459529).
- Temperature scaling produces low TS ECE across these ExpW evaluations (≈0.026–0.041), while raw ECE remains high (≈0.237–0.255), reinforcing the “calibration benefit” pattern observed elsewhere in this project.

## 4.9 Detailed extension summary (Jan 2026)

### 4.9.1 NL / NegL / complementary learning (offline screening)

Hypothesis:

- Adding NegL (and/or NL(proto)) on top of KD/DKD can improve calibration and potentially improve minority performance by preventing confident mistakes.

What the artifacts show (Sections 4.5.1–4.5.3):

- Under the tested short-budget configurations, **macro-F1 improvements are not consistent**, and some configurations regress.
- NegL gating has a measurable “strength” via `applied_frac` in `history.json`. When the gate is too selective, NegL likely has limited effect; when too strong, it can destabilize training.
- A configuration can improve **raw calibration metrics** (raw ECE / raw NLL) without improving macro-F1 (the DKD+NL+NegL synergy compare in Section 4.5.2 is the clearest example currently in-repo).

Practical takeaway:

- Treat NL/NegL as a sensitive regularizer: it requires careful tuning and stronger safety checks than plain KD/DKD.

### 4.9.2 Domain shift improvement via Self-Learning + NegL (webcam loop)

Figure 4.9.2 (Domain shift loop; “safety gate” prevents silent regressions):

```mermaid
flowchart TD
  A[Webcam session
  per_frame.csv + manual labels (optional)] --> B[Score baseline
  score_results.json]
  B --> C[Build buffer
  stable/high-confidence frames]
  C --> D[Self-learning update
  (small/limited params)]
  D --> E[Optional NegL
  (gated by entropy/confidence)]
  E --> F[Offline gates
  eval-only + ExpW]
  F -->|pass| G[Accept candidate
  save as new checkpoint]
  F -->|fail| H[Reject candidate
  reduce update size / tighten buffer]
```

Problem:

- Webcam deployment differs from training distributions; the live scoring logs show class-specific failures and run-to-run variability (Section 4.6).

What is already implemented and evidenced:

- A reproducible webcam scoring protocol (raw vs smoothed, plus jitter flips/min) stored in `demo/outputs/*/score_results.json`.
- An offline regression gate on the eval-only manifest; in the current experiments, conservative fine-tuning candidates failed this gate and were rejected (Section 4.7).

2026-02-21 update (this report version): issues found, fixes applied, and A/B outcome

Observed issues:

- **Preprocessing mismatch can cause false “regressions”:** the baseline checkpoint used `use_clahe=True` while early adaptation checkpoints were trained with `use_clahe=False`. Since evaluation defaults to the checkpoint’s stored settings, the gate check was not initially apples-to-apples.
- **BatchNorm running-stat drift under small-buffer tuning:** even when tuning “head-only”, calling `model.train()` updates BatchNorm running mean/variance, which can cause large distribution shifts from a tiny webcam buffer.

Fixes implemented (repo changes):

- Added a conservative training safeguard: when `--tune` is not `all`, BatchNorm layers are forced to `eval()` during training to freeze running stats (file: `scripts/train_student.py`).
- Added a replay inference utility to enable fair webcam A/B scoring on the *same* recorded session while preserving manual labels/time: `scripts/reinfer_webcam_session.py`.

Method detail (what “Self-Learning + NegL” means in this A/B):

- Buffer source: the self-learning buffer was built from the recorded session’s `per_frame.csv` by selecting stable frames and using the model’s predicted label as the pseudo-label (not the manual label). The manifest is saved under `demo/outputs/20260126_205446/buffer_selflearn/manifest.csv`.
- Confidence-banded supervision in the manifest:
  - High-confidence frames ($p_{max} \ge \tau_{high}$) become pseudo-labeled positives with `weight=1`.
  - Medium-confidence frames ($\tau_{mid} \le p_{max} < \tau_{high}$) become **NegL-only** samples with `weight=0` and `neg_label=<predicted_label>` (i.e., discourage probability mass on the model’s own uncertain prediction).
  - Low-confidence frames are excluded.
- Training consumption: the adapted run consumes `weight` as weighted CE and consumes `neg_label` as the explicit NegL target. This is intentionally conservative (no positive CE update on medium-confidence frames) but can still be harmful if the buffer is narrow or the negative target policy is mis-specified.

Gate check result (eval-only manifest):

- Baseline checkpoint: `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`
- Adapted checkpoint (head-only + CLAHE + BN-stats frozen): `outputs/students/DA/mnv3_webcamselflearn_negl_clahe_head_frozebn_20260221_211025/best.pt`

On `Training_data_cleaned/classification_manifest_eval_only.csv` (split `test`, single-crop), the adapted checkpoint matches the baseline within rounding in macro-F1, indicating the offline safety gate can be passed when preprocessing and BN behavior are controlled (artifacts: `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_205322/baseline/` and `outputs/evals/students/evalonly_ab_webcamselflearn_20260221_211119/adapted_clahe_head_frozebn/`).

Webcam A/B result (same recorded session, same manual labels):

- Session directory: `demo/outputs/20260126_205446/`
- Baseline score: `demo/outputs/20260126_205446/score_results.json`
- Adapted score (re-infer + re-score): `demo/outputs/20260126_205446/ab_adapted_frozebn/score_results.json`

Summary (smoothed predictions):

- Baseline: accuracy 0.5879, macro-F1 0.5248, minority-F1(lowest-3) 0.1609, jitter 14.86 flips/min
- Adapted: accuracy 0.5269, macro-F1 0.4667, minority-F1(lowest-3) 0.1384, jitter 14.16 flips/min

Interpretation:

- The gate-passing adaptation is **not yet beneficial** for the labeled webcam session: it slightly reduces jitter but regresses accuracy and macro-F1.
- This supports the “safety-gated adaptation” framing: passing an offline gate is necessary, but not sufficient, to claim an improvement on deployment-facing metrics.
- Scope limitation (important for academic interpretation): this is evidence from a single recorded session and a single conservative adaptation candidate. It demonstrates a failure mode (gate pass does not imply webcam improvement), but it does not prove that self-learning + NegL cannot help under other buffer policies, thresholds, or multi-session data.

How ExpW fits this extension:

- ExpW cross-dataset evaluation (Section 4.8) provides an additional controlled in-the-wild test bed to evaluate whether domain-shift adaptations generalize beyond a single webcam session.

## 5. Demo and Application

## 5.1 Real-time demo pipeline

The demo pipeline:

1. Frame capture (webcam/video)
2. Face detection and crop
3. Preprocessing (including optional CLAHE)
4. FER inference (teacher/student checkpoints)
5. Temporal stabilization (smoothing/hysteresis)
6. Visualization and logging (`demo/outputs/*/per_frame.csv` and derived summaries)

Stabilization implementation note (verified from `demo/realtime_demo.py`):

- EMA smoothing over probabilities (parameter `ema_alpha`)
- Hysteresis on the predicted class index (parameter `hysteresis_delta`)
- Optional vote window (parameters `vote_window` / `vote_min_count`)

January 2026 engineering progress (recorded in `research/process_log/Jan process log/Jan_week4_process_log.md`):

- Device forcing support was added to enable repeatable CPU/GPU benchmarking (`--device {auto,cpu,cuda,dml}` in demo code).
- A backup package was generated: `outputs/realtime_fer_backup.zip`.

## 5.2 Deployment KPIs (current status)

What is already measured and stored:

- Live classification behavior: raw vs smoothed macro-F1/accuracy and jitter flips/min from `demo/outputs/*/score_results.json`.

What remains required for a complete deployment report:

- A timed benchmark run on the target device (CPU-only and GPU if applicable) to report FPS and latency distribution.

Note (Dec-24 mini-report pack): `research/report of project restart/mini report 24-12-2025/mini report md file/06_basic_demo_report.md` intentionally does not claim FPS/latency/flip-rate numbers because those require a dedicated timed demo run with an attached per-frame CSV log.

## 6. Discussion and Limitations

This section refines the interpretation boundaries and clarifies limitations to avoid confusing (or overstating) results.

## 6.0 Discussion refinement (what results mean, and what they do not)

- **Different evaluation regimes answer different questions:**
  - Training-time validation (teacher Stage-A val; student HQ-train val) measures *in-distribution* model selection.
  - Offline gates (eval-only / ExpW / mixed-source) measure *deployment-aligned stress* under domain shift and label noise.
  - Official-split tests (e.g., FER2013 PublicTest/PrivateTest from `fer2013.csv`) support *paper-style* comparison.
- **Therefore, numbers are not interchangeable:** a high Stage-A teacher val macro-F1 does not imply high mixed-source macro-F1, and vice versa.

## 6.0.1 Clear limitations (protocol + data + deployment)

- **Protocol mismatch is the #1 paper-compare risk:** single-crop vs ten-crop, preprocessing, and label mapping can move accuracy materially. This is why we label comparisons as `Comparable: Yes/Partial/No`.
- **Dataset packaging vs official splits:** folder-packaged datasets can differ from papers’ official split definitions. This report treats official-split evaluation (when available) as the anchor, and uses folder/uniform-7 as stress tests.
- **Domain shift dominates deployment:** webcam lighting, motion blur, and crop jitter can cause failures that are invisible on curated offline tests.
- **Temporal stabilization changes the measured objective:** raw vs smoothed metrics are both needed; smoothing can improve perceived stability while hiding instantaneous errors.

## 6.0.2 Adaptation risks (why safety gates exist)

- **Small-buffer fine-tuning is risky:** it can improve the target session while harming broad generalization.
- **NL/NegL is sensitive:** gating/weighting choices can produce calibration changes without macro-F1 gains.
- **KD/DKD can reduce probability margins under shift:** increased near-ties can amplify flicker under EMA/hysteresis.

Practical safeguard used in this repo:

- Adaptation candidates must pass offline regression gates (eval-only / ExpW) before any “improved” claim.

## 6.1 Analytical comparison vs papers (trade-off analysis, not “winning”)

Supervisor clarification: in this FYP, “comparison” is primarily an **analytical comparison**.

Goal:

- Not to compete with SOTA papers on a single benchmark number.
- Instead, to explain **where and why** performance differs, whether the gap is reasonable under our constraints, and what engineering trade-offs were chosen to meet real-time deployment goals.

### 6.1.1 Our system objective vs typical paper objective

This project’s target is a deployable **real-time** FER pipeline:

- Student model: MobileNetV3-Large (lightweight)
- Primary target: CPU real-time inference (typical goal: 25–30 FPS)
- Engineering: multi-threaded pipeline + temporal smoothing and hysteresis
- Data: multi-source, large-scale, noisy labels; explicit domain shift focus
- Reproducibility: artifact-grounded pipeline (manifests + checkpoints + metrics JSON)

In contrast, many FER papers primarily optimize an **offline benchmark objective**:

- Heavy backbones (e.g., ResNet-50 / ViT / Swin variants)
- GPU inference assumed
- Cleaner, single-dataset protocols
- No latency, stability, or deployment constraints

Because the objective and constraints differ, a lower offline score in our setting can be both **expected** and **acceptable**, as long as the trade-off is made explicit and measured.

### 6.1.2 Why our numbers should not be compared 1:1 to paper SOTA

Direct numeric comparison is only fair when the following are matched:

- Dataset split/protocol (official train/test, crop policy, etc.)
- Label mapping (7-class vs 8-class vs compound)
- Metric definition (accuracy vs macro-F1, averaging, test-time augmentation)

FER2013 example (common mismatch): many papers report **ten-crop** accuracy, while many baselines (including earlier versions of this project) report **single-crop** evaluation. In this report, we explicitly separate and report **both** single-crop and ten-crop results for the official FER2013 PublicTest/PrivateTest splits, and still avoid claiming strict numeric equivalence unless preprocessing and training protocol are also matched.

In this repo, we keep comparisons evidence-backed and protocol-aware:

- One-page comparability table: `research/paper_vs_us__20260208.md`
- Protocol extraction notes: `research/paper_metrics_extraction__20260208.md`
- Fair comparison rules (living doc): `research/fair_compare_protocol.md`

### 6.1.3 Where performance differs and why that is expected

The most consistent failure modes on hard, mixed-domain gates are minority classes (Fear/Disgust) and cross-domain generalization.

Main reasons the system can underperform paper-reported SOTA numbers:

- **Model capacity trade-off**
  - Heavy backbones generally win raw accuracy/macro-F1.
  - MobileNetV3 is chosen to satisfy CPU real-time constraints; some accuracy loss is a reasonable trade.

- **Dataset difference (scale + noise + mixture)**
  - Many papers use a single curated dataset and report on its official test split.
  - This project trains/evaluates on a large multi-source mixture (466k rows validated), where label noise and domain mismatch can dominate macro-F1.

- **Deployment constraint changes the optimization target**
  - Real-time user experience is shaped by stability (flicker), confidence calibration, and robustness under webcam lighting/pose.
  - These factors are rarely optimized or even reported in offline-only paper settings.

- **Domain shift is explicitly in-scope here**
  - Webcam domain shift (sensor + lighting + motion blur + subject variation) causes predictable drops.
  - This repo treats domain shift as a first-class problem and adds a safety-gated adaptation track rather than optimizing only for one static benchmark.

### 6.1.4 What we optimize that papers often do not

This repo measures and engineers for deployment-facing behaviors:

- **Calibration** via temperature scaling (ECE/NLL) and artifact logging per run.
- **Stability** via smoothing + hysteresis (reported separately from raw classifier metrics).
- **Reproducibility** via manifests, validation, and stored metrics artifacts.

Concrete “overall sanity” snapshot (CE vs KD vs DKD on four hard gates) is provided as a single artifact-backed table:

- `outputs/benchmarks/overall_summary__20260208/overall_summary.md`

Interpretation rule:

- Use SOTA papers as a **reference point** to explain gaps and constraints.
- Only claim a “paper-comparable” number when the protocol is truly matched.

## 6.2 FYP requirements checklist (evidence audit)

The following checklist maps common FYP requirements (as discussed with the supervisor) to concrete evidence already stored in this repo.

| Requirement | Status | Evidence in repo |
| --- | --- | --- |
| 1) Study deep learning methods for FER | Met | Paper study notes: `research/FYP Paper/Paper study report.md`; background + method discussion in this report and interim reports under `research/Interim Report/`. |
| 2) Study attention mechanisms + apply attention / knowledge distillation | Met (KD/DKD) / Met (attention as used in backbones) | KD/DKD implemented and evaluated: `scripts/train_student.py`, `outputs/evals/students/**/reliabilitymetrics.json`. Attention mechanisms are included in used architectures (e.g., SE-style attention in MobileNetV3 / EfficientNet) and discussed in interim reports (e.g., attention modules references in `research/Interim Report/`). |
| 3) Identify at least two datasets for experimentation | Met | Multi-source manifests: `Training_data_cleaned/classification_manifest.csv`, `Training_data_cleaned/classification_manifest_hq_train.csv`, `Training_data_cleaned/classification_manifest_eval_only.csv`, `Training_data_cleaned/expw_full_manifest.csv`, and official FER2013 split manifests derived from local `fer2013.csv`: `Training_data/FER2013_official_from_csv/manifest__publictest.csv`, `Training_data/FER2013_official_from_csv/manifest__privatetest.csv`. |
| 4) Investigate and evaluate at least three methods | Met | Student: CE vs KD vs DKD comparisons with artifact-backed summaries (e.g., `outputs/benchmarks/overall_summary__20260208/overall_summary.md`, `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`). Teacher backbones also provide additional method variety (RN18/B3/CNXT). |
| 5) Explore techniques to enhance FER performance (optional) | Met | Domain-shift loop + NL/NegL screening documented: `research/nl_negl_plan/`, plus Feb-2026 addendum sections in this report. |
| 6) Present at least one paper (seminar / paper study presentation) | Met | Presentation artifacts in repo: `research/presentation/interim_presentation_2025-12-31.md` and `research/presentation/presentation_2026-1-5.md`. |

## 7. Project Timeline (updated)

Jan 2026 (completed/ongoing):

- Domain shift measurement loop (record → score → buffer → fine-tune → offline gate).
- NL/NegL offline screening runs and documentation under `research/nl_negl_plan/`.
- Demo engineering for repeatable device benchmarking and backup packaging.

Feb 2026 (planned):

- Implement a safe self-learning + NegL ablation in the domain-shift pipeline with strict default-off gating (first A/B executed on 2026-02-21; did not yet improve the labeled webcam replay score).
- Tighten buffer construction (stable-only sampling, per-class caps, and replay anchors) and rerun conservative adaptation.
- Paper-study-driven extensions (Week 1): implement Paper #5 Track A as an **optional** supervised auxiliary loss (LP-loss) in student training, with artifact logging and an optional post-training gate evaluation hook (implementation completed; no new numeric results claimed here).
- Experiment order update: establish a fresh **KD baseline first**, then add LP-loss as KD+LP, then consider DKD and other research-y ablations only after the eval-only gate is stable.

Mar 2026 (planned):

- Timed demo KPI reporting (FPS, latency, jitter/flip-rate) on the target CPU and/or GPU.

Apr 2026 (planned):

- Consolidated final evaluation on a controlled benchmark manifest, final write-up, and packaging.

## 8. Lessons Learned from Development

- **Artifact-grounded workflow prevents silent drift:** manifests, JSON metrics, and checkpoints make results auditable.
- **Validation gates save time:** manifest path/label validation prevents wasted training.
- **Resume semantics matter:** DKD resume must guarantee `total_epochs > start_epoch` to avoid no-op runs.
- **Deployment metrics must be explicit:** real-time stability (flip-rate/jitter) is not implied by offline macro-F1.
- **Domain adaptation needs safety rails:** without an offline gate, small-buffer adaptation can silently harm generalization.

## 9. Conclusion and Next Steps

## 9.1 Conclusion

This project successfully rebuilt an end-to-end, reproducible FER pipeline with teacher training, ensemble selection, student distillation, and real-time demo logging. Calibration results show that KD/DKD can improve confidence reliability after temperature scaling, while macro-F1 remains strongest under CE in the main Dec 2025 student evaluation.

January 2026 extensions advanced two practical directions:

- NL/NegL offline screening (not yet showing consistent macro-F1 gains under tested settings).
- A deployment-facing domain shift loop with explicit real-time metrics and an offline regression gate; a first conservative 2026-02-21 Self-Learning + manifest-driven NegL candidate passed the offline gate but regressed on the same-session webcam A/B, reinforcing the need for both gates.

## 9.2 Immediate next steps (evidence-driven)

1. Run a timed demo session to report FPS/latency and attach the output folder under `demo/outputs/`.
2. Establish a fresh **KD baseline** on the current HQ-train pipeline, then evaluate on:
   - eval-only manifest: `Training_data_cleaned/classification_manifest_eval_only.csv`
   - ExpW manifest: `Training_data_cleaned/expw_full_manifest.csv`
   Evidence required: `outputs/evals/students/*/reliabilitymetrics.json` plus a per-run summary file (if enabled).
3. After KD baseline is confirmed stable, run **KD + LP-loss** (small weight) and re-check the same gate artifacts.
4. Complete the pending BN-only webcam gate by recording and scoring a labeled run using `outputs/students/FT_webcam_bn_20260126_1/best.pt`.
5. If adaptation still fails the offline gate, reduce update size and tighten buffer sampling before enabling any NegL behavior.

### 9.2.1 Roadmap (Feb–Apr 2026; presentation-ready)

Phase FC (fair comparison, supervisor-facing):

- Completed: FER2013 **ten-crop** evaluation (keep single-crop too) and report both blocks side-by-side.
- For each paper target, complete a gap checklist (split / crop / preprocessing / resolution / label mapping / backbone capacity / training settings).

Phase DS (domain shift improvement, deployment-facing):

- Run Self-Learning + NegL as a strict, safety-gated ablation (log pseudo-label acceptance rate + buffer class distribution + before/after gate metrics).
- Only promote an adapted checkpoint if it improves target-domain evidence (webcam replay on the same labeled session) and does not regress eval-only/ExpW.

## 9.3 Feb 2026 addendum (Week 1): Paper-study-driven implementation update

The main body of this report covers Aug 2025 – Jan 2026. Section 9.3 documents Feb-2026 updates that extend the codebase and experimental plan; evaluation results are only stated when backed by the listed artifacts.

### 9.3.1 Paper #5 Track A: Deep Locality-Preserving loss (LP-loss) implementation

Implementation status:

- Implemented in: `scripts/train_student.py`
- Safety posture: default-off (`--lp-weight 0.0`), enabling requires explicit CLI flags.
- Logging: when enabled, the run’s `history.json` includes an `lp` block with:
  - `weight`, `k`, `embed`
  - `train_lp_loss`
  - `included_frac` (fraction of batch samples eligible for within-class neighbors)

Backup safeguard:

- A backup snapshot of key scripts was created before editing: `backups/before_lp_loss_20260205_144641/`.

### 9.3.2 Optional post-training evaluation (gate artifact generation)

To keep the workflow artifact-grounded and reduce manual steps, the student training entrypoint now supports an optional post-training evaluation hook:

- Flag: `--post-eval`
- Behavior: runs `scripts/eval_student_checkpoint.py` on eval-only and ExpW manifests after training finishes.
- Outputs:
  - `outputs/evals/students/*/reliabilitymetrics.json` (per evaluation)
  - `post_eval.json` inside the training run folder (summary of return codes and parsed outputs)

### 9.3.3 Updated experiment order (risk-managed)

Recommended order for Feb-2026 experiments:

1. KD baseline (no LP) → confirm eval-only gate stability.
2. KD + LP-loss (small weight) → check ExpW improvement without eval-only regression.
3. DKD and other research-heavy ideas only after the above are stable.

### 9.3.4 Feb 2026 addendum (Week 1): KD baseline vs KD + LP-loss (evidence-backed)

This subsection records the first short-budget screening results produced after the LP-loss implementation, using the repo’s standard artifacts.

Runs (training outputs):

- KD baseline: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/`
- KD + LP-loss: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/`

#### A) HQ-train validation split (from each run’s `reliabilitymetrics.json`)

Table 9.3.4-A (HQ-train validation). Dataset: `Training_data_cleaned/classification_manifest_hq_train.csv`. Split: `val`. Protocol: single-crop. Evidence: each run’s `reliabilitymetrics.json` under the listed training output folders.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Global T |
| --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep) | 0.7297586 | 0.7281613 | 0.0373908 | 0.7926007 | 4.4717526 |
| KD + LP (w=0.01, k=20, penultimate; 5ep) | 0.7296656 | 0.7276670 | 0.0252364 | 0.7612492 | 3.4970691 |

Interpretation (limited to the evidence above):

- On HQ-train val, KD+LP did not improve raw macro-F1 vs KD baseline in this 5-epoch screening.
- Calibration signals improved on this split (TS ECE and TS NLL both decrease), but this does not imply better cross-domain performance.

#### B) Offline gates (post-eval outputs under `outputs/evals/students/`)

Eval-only (safety gate):

Table 9.3.4-B1 (Offline safety gate: eval-only). Dataset: `Training_data_cleaned/classification_manifest_eval_only.csv`. Split: `test`. Protocol: single-crop. Evidence: post-eval `reliabilitymetrics.json` under `outputs/evals/students/`.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.5162321 | 0.4385411 | 0.0217606 | 1.2961859 |
| KD + LP | 0.5207738 | 0.4411229 | 0.0374865 | 1.2773255 |

ExpW (target-domain proxy):

Table 9.3.4-B2 (Cross-dataset gate: ExpW). Dataset: `Training_data_cleaned/expw_full_manifest.csv`. Split: `test`. Protocol: single-crop. Evidence: post-eval `reliabilitymetrics.json` under `outputs/evals/students/`.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.6311145 | 0.4595847 | 0.0276567 | 1.0635237 |
| KD + LP | 0.6356902 | 0.4583109 | 0.0197645 | 1.0421315 |

Interpretation (evidence-first):

- In this screening, KD+LP slightly increases eval-only macro-F1 but slightly decreases ExpW macro-F1.
- ExpW calibration improves (TS ECE and TS NLL decrease), but raw macro-F1 does not improve here.

Recommended next steps (low-risk):

1) If the goal is ExpW macro-F1: try smaller `--lp-weight` (e.g., 0.001) and/or `--lp-embed logits`, then rerun the same gates.
2) If the goal is deployment stability: treat these as candidate base checkpoints for the webcam loop, but do not promote without passing the eval-only gate and re-scoring the same labeled session.

### 9.3.5 Feb 2026 addendum (Week 2): Offline benchmark suite + challenging-benchmark diagnostics (evidence-backed)

This subsection documents Week-2 Feb-2026 diagnostic work aimed at explaining unexpectedly low offline results on (1) eval-only, (2) ExpW, and (3) FER2013 (uniform-7 stress-test). The intent is not to “fix” the numbers in reporting, but to localize failure modes and confirm whether obvious preprocessing mismatches (e.g., CLAHE) are responsible.

Primary evidence artifacts (full path inventory moved to Appendix A.4):

- Offline suite export (canonical “what was scored” snapshot): `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
- Investigation write-up (manifest integrity + per-class comparisons): `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`

#### A) Per-source breakdown on eval-only (CE checkpoint)

To separate “model weakness” from “mixture composition”, eval-only was re-evaluated with a per-source grouped report enabled.

Key observation (from the per-source metrics artifact):

- `expw_hq` is substantially weaker than `expw_full` on eval-only macro-F1, with pronounced failures in Fear/Disgust.

Per-source raw macro-F1 (CE checkpoint; eval-only test split):

| Source | n | Raw acc | Raw macro-F1 |
| --- | ---: | ---: | ---: |
| expw_full | 6780 | 0.6361357 | 0.4895467 |
| expw_hq | 3336 | 0.4679257 | 0.2788844 |
| rafml_argmax | 982 | 0.5539715 | 0.4848964 |
| rafdb_compound_mapped | 792 | 0.4141414 | 0.3296730 |

Interpretation (bounded by this evidence):

- The low aggregate eval-only result is not uniform across all sources; it is driven primarily by source composition (notably `expw_hq`) and minority-class fragility.

#### B) CLAHE ablation (CE checkpoint): ExpW + FER2013

To test whether preprocessing mismatch is a root cause, ExpW and FER2013 uniform-7 were re-evaluated with CLAHE disabled.

ExpW full manifest (`Training_data_cleaned/expw_full_manifest.csv`):

| Setting | Raw acc | Raw macro-F1 | Fear F1 | Disgust F1 |
| --- | ---: | ---: | ---: | ---: |
| CLAHE on | 0.6576969 | 0.4821205 | 0.2152466 | 0.1631505 |
| CLAHE off | 0.6506155 | 0.4689109 | 0.2040816 | 0.1418021 |

FER2013 uniform-7 (`Training_data_cleaned/test_fer2013_uniform_7.csv`):

| Setting | Raw acc | Raw macro-F1 | Fear F1 |
| --- | ---: | ---: | ---: |
| CLAHE on | 0.5241429 | 0.4973563 | 0.1881356 |
| CLAHE off | 0.4882857 | 0.4565744 | 0.1543860 |

Interpretation (evidence-first):

- Disabling CLAHE makes both ExpW and FER2013 worse on macro-F1, so CLAHE is not the main cause of the low benchmark results.
- FER2013 Fear remains a dominant failure mode even in the balanced setting, pointing to domain/preprocessing mismatch beyond calibration alone.

#### C) Interpreting “mixed-source” offline tests (why results can appear low)

This project uses several offline tests that intentionally mix multiple sources/domains (e.g., `classification_manifest_eval_only.csv`, and historically `test_all_sources.csv`). These are valuable, but they are also where aggregate scores can appear low even when the model is strong on cleaner single-dataset tests.

Key reasons (bounded by our stored evidence above):

- **Domain shift dominates**: sources like ExpW (especially `expw_hq`) are much harder than curated lab-style datasets, and can pull down macro-F1 strongly.
- **Minority-class fragility**: Fear/Disgust repeatedly show the weakest F1 under domain shift; macro-F1 is designed to expose this.
- **Label noise / ambiguity**: in-the-wild datasets contain more ambiguous or noisy labels; accuracy can hide these effects when majority classes dominate.

How to use these mixed-source tests correctly:

- Use them as **regression gates** (deployment-realistic stress tests), not as the only “is the model good?” judgment.
- Always report at least one **fixed single-dataset test split** alongside them (e.g., RAF-DB basic test, FERPlus test, AffectNet balanced test).

Paper-style comparison rule:

- For FER2013, many papers report accuracy on the **official public test split** (often via `fer2013.csv` Usage=PublicTest). Our in-repo `fer2013_uniform_7` is a different split/protocol, so strict comparison requires running the official split evaluation.
- Interpretation note (licensing): `fer2013.csv` is license-restricted, so the dataset is not redistributed; only derived manifests and evaluation artifacts are stored.

Key takeaways (Week-2 diagnostics):

- The weakest offline results concentrate in cross-domain/mixed-source settings and minority classes (especially Fear/Disgust).
- CLAHE is not the root cause; turning it off degrades results.
- Per-source breakdown indicates mixture composition (notably `expw_hq`) can dominate aggregate eval-only performance.

### 9.3.6 Feb 2026 addendum (Week 2): Evidence-backed comparison with provided papers (PDFs)

This subsection records a bounded comparison against the papers placed in `research/paper compared/`. Because papers often differ in label space, split definition, preprocessing/alignment, and metrics (accuracy vs macro-F1; balanced vs imbalanced evaluation), this comparison is used to contextualize our results rather than claim strict SOTA equivalence.

Primary comparison artifacts (full evidence index moved to Appendix A.5):

- One-page “paper vs us” table (comparability flags + links to stored artifacts): `research/paper_vs_us__20260208.md`
- Paper protocol/metric extraction notes (quotable lines + limitations): `research/paper_metrics_extraction__20260208.md`
- FER2013 official split summary (protocol-aware): `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`

#### A) RAF-DB (accuracy)

- Paper (face-regions analysis): reports RAF-DB **whole-face** testing accuracy **82.69%** *with padding* (Table 5).
- Ours (student CE, `test_rafdb_basic`): raw accuracy **86.28%**, raw macro-F1 **0.792**.

Interpretation (careful): this looks competitive, but it is not guaranteed to be a strict apples-to-apples comparison (exact split/protocol may differ).

#### B) FER2013 (accuracy; split mismatch warning)

- Paper (“State of the Art Performance on FER2013”): reports test accuracy **73.28%** on the **FER2013 public test set**.

Two relevant evaluation regimes exist in this repo:

1) **Non-official** stress-test split (FER2013 uniform-7 / folder datasets): useful as a hard gate, but not protocol-matched to the paper.

2) **Official** FER2013 split from `fer2013.csv` (Usage=PublicTest/PrivateTest): protocol-matched on split definition, but still a protocol mismatch if the paper uses ten-crop.

Table 9.3.6-B (FER2013 official split). Dataset: official `fer2013.csv` Usage=PublicTest/PrivateTest. Split: PublicTest and PrivateTest (n=3589 each). Protocol: single-crop and ten-crop are reported separately. Evidence: `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`.

| Split | Protocol | n | Accuracy | Macro-F1 | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| PublicTest | single-crop | 3589 | **0.614099** | **0.553776** | `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md` |
| PublicTest | ten-crop | 3589 | **0.609083** | **0.557332** | `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md` |
| PrivateTest | single-crop | 3589 | **0.608247** | **0.539025** | `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md` |
| PrivateTest | ten-crop | 3589 | **0.612148** | **0.547634** | `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md` |

Interpretation (most important limitation): even with the official split and protocol-aware reporting, a strict 1:1 numeric comparison across papers still depends on details that can vary by author (and are not always fully specified): exact preprocessing/alignment, image resolution, training schedule, augmentation, and whether extra data or pretraining was used. Therefore, we treat the official-split table as the strongest **anchor** for gap analysis, but we avoid claiming strict SOTA equivalence without matching those additional variables.

Additional evidence (partial comparable; different split): we also evaluated on a Kaggle FER2013 folder dataset (msambare). This uses a different train/test split and is **not** a strict match to the “public test” protocol, but is useful as a second external sanity check:

- Manifest: `Training_data/fer2013_folder_manifest.csv`
- Metrics artifact (student DKD checkpoint on `test` split):
  - `outputs/evals/students/DKD_20251229_223722__fer2013_folder__test__20260208/reliabilitymetrics.json`

#### C) AffectNet (macro-F1; balanced-subset warning)

- Paper (AffectNet database): Table 7 provides per-class F1 for multiple training approaches. From the weighted-loss approach (Top-1), the macro-average across the eight classes is:
  - Top-1 macro-F1 (Orig): **0.555**
  - Top-1 macro-F1 (skew-normalized): **0.625**
- Repro note (derived from Table 7 values): `outputs/paper_extract/affectnet__table7_weightedloss_macro_f1.md`
- Ours: on `test_affectnet_full_balanced`, student CE achieves raw macro-F1 **0.823** (raw acc **0.822**).

Interpretation (limitation): our test is explicitly a balanced subset/manifest, while the paper discusses original vs skew-normalized evaluation on the AffectNet test set; these are not directly comparable.

#### D) Summary interpretation (bounded by evidence)

- In-domain datasets (e.g., RAF-DB basic, FERPlus) show strong performance for a real-time student.
- The weakest results concentrate in cross-domain/mixed-source scenarios (ExpW/eval-only/FER2013 stress tests) and minority classes, consistent with domain shift + label ambiguity + class fragility rather than uniformly weak modeling.
- The official-split FER2013 table is the strongest anchor for gap analysis, but strict cross-paper comparisons still depend on preprocessing/alignment, training recipe, and protocol details.

#### E) Next work (to improve paper comparability and domain-shift robustness)

- Expand protocol-matched evaluations on official splits and maintain derived manifests + metrics artifacts; document acquisition constraints for licensed datasets and plan for compute/storage (may require additional funding).
- After protocol match, run structured gap analysis across protocol variables (crop/TTA, preprocessing/alignment, resolution, label mapping), model capacity/teachers, and training settings (schedule, augmentation, loss weights, KD/DKD temperatures).
- Treat author-to-author differences as first-class and keep strict protocol separation in reporting (do not mix stress-test splits with official-split claims).
- For deployment robustness, continue conservative safety-gated Self-Learning + NegL, and standardize deployment-facing metrics (flip-rate/jitter + replay-scored accuracy/macro-F1 on the same labeled session).

Appendix pointers (to keep the main report concise):

- Week-2 diagnostics artifact paths + error sampling lists: Appendix A.4
- Paper-comparison evidence index (manifests/checkpoints/metrics artifacts): Appendix A.5

## 10. References

Selected references relevant to the methods used in this project.

[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.

[2] B. Zhao, Q. Cui, R. Song, Y. Qiu, and J. Liang, "Decoupled Knowledge Distillation," in Proc. IEEE/CVF CVPR, 2022.

[3] C. Deng, D. Huang, X. Wang, and M. Tan, "Nested Learning: A New Paradigm for Machine Learning," arXiv:2303.10576, 2023.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in Proc. IEEE/CVF CVPR, 2019.

[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in Proc. ICML, 2019.

[6] A. Howard, M. Sandler, G. Chu, et al., "Searching for MobileNetV3," in Proc. IEEE/CVF ICCV, 2019.

[7] Z. Liu, H. Mao, C.-Y. Wu, et al., "A ConvNet for the 2020s," in Proc. IEEE/CVF CVPR, 2022.

[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. IEEE CVPR, 2016.

[9] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks," in Proc. ECCV, 2016.

[10] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," in Proc. ECCV, 2018.

[11] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in Proc. ICCV, 2017.

[12] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples," in Proc. IEEE/CVF CVPR, 2019.

[13] A. Menon, S. Jayasumana, A. S. Rawat, et al., "Long-Tail Learning via Logit Adjustment," in Proc. ICLR, 2021.

[14] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proc. ICML, 2017.

[15] Y. Zhang, T. Liu, M. Long, and M. I. Jordan, "Learning with Negative Learning," in Proc. ICML, 2019.

[16] T. Ishida, G. Niu, W. Hu, and M. Sugiyama, "Learning from Complementary Labels," in Proc. NeurIPS, 2017.

[17] A. Mollahosseini, D. Chan, and M. H. Mahoor, "AffectNet: A Database for Facial Expression, Valence, and Arousal in the Wild," IEEE Trans. Affective Comput., 2019.

[18] W. Wu, Y. He, S. Wang, et al., "YuNet: A Fast and Accurate Face Detector," arXiv:2111.04088, 2021.

[19] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019.

## 11. Appendix

## A.0 Interim Report v4 figure-to-artifact mapping

The interim report v4 contains several ASCII “figures” that visualize the same artifact-backed tables reported here. This subsection maps those figures to the corresponding sections in this final report.

Source document:

- `research/Interim Report/version 4 Real-time-Facial-Expression-Recognition-System Interim Report (25-12-2025).md`

Mapping:

- Figure 1.5.1 (Teacher macro-F1 by backbone) → Section 4.2 (Teacher performance)
- Figure 1.5.2 (Per-class F1 across teachers) → Section 4.2 (Per-class F1 table)
- Figure 1.5.4 (Student CE/KD/DKD comparison) → Section 4.4 (Student performance)
- Figure 1.5.5 (Ensemble robustness benchmark) → Section 4.3 (Ensemble robustness benchmark)

Notes:

- Figures in the interim report that include class-count breakdowns or other derived quantities are not duplicated here unless the underlying counts are explicitly traced to a stored manifest-count artifact.

## A.1 Evidence inventory (key artifacts)

Dataset and integrity:

- `outputs/manifest_validation_all_with_expw.json`

Teachers:

- `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/reliabilitymetrics.json`
- `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/reliabilitymetrics.json`
- `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/reliabilitymetrics.json`
- `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/alignmentreport.json` (split sizes)

Ensemble benchmark:

- `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json`

Student (Dec 2025 main run):

- `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/reliabilitymetrics.json`
- `outputs/students/_archive/2025-12-23/KD/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/reliabilitymetrics.json`
- `outputs/students/_archive/2025-12-23/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/reliabilitymetrics.json`

NL/NegL report pack:

- `research/nl_negl_plan/NL_NegL_report/NL_NegL_report.md`
- Example compare: `outputs/students/_compare_kd5_vs_negl5.md`

Domain shift (webcam):

- `demo/outputs/20260126_205446/score_results.json`
- `demo/outputs/20260126_215903/score_results.json`
- Offline gate: `outputs/evals/students/*/reliabilitymetrics.json`

Supporting documentation (narrative/design notes; metrics are still taken from the artifacts above):

- Interim report v4: `research/Interim Report/version 4 Real-time-Facial-Expression-Recognition-System Interim Report (25-12-2025).md`
- Dec-24 mini-report pack index: `research/report of project restart/mini report 24-12-2025/mini report md file/00_index.md`

## A.2 Metric definitions

- **Macro-F1**: unweighted mean of per-class F1.
- **ECE**: Expected Calibration Error (binned absolute gap between accuracy and confidence).
- **NLL**: Negative Log-Likelihood.
- **Temperature scaling**: global temperature $T$ applied to logits; accuracy/macro-F1 remain unchanged, while NLL/ECE can change.

## A.3 Manifest distribution tables

The following distribution summaries are generated directly from the CSV manifests under `Training_data_cleaned/`:

- Generator: `scripts/summarize_manifest_counts.py`
- Outputs: `outputs/manifest_counts_summary.md` and `outputs/manifest_counts_summary.json`

## A.4 Feb 2026 Week-2 diagnostics: artifact inventory

This appendix subsection contains the long-form artifact-path inventory for Section 9.3.5.

Primary suite artifacts:

- Offline suite index + CSV export:
  - `outputs/benchmarks/offline_suite__20260208_192604/benchmark_index.json`
  - `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
- Investigation report (manifest integrity + per-class F1 comparisons):
  - `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`

Per-source breakdown artifacts:

- Run artifact folder (eval-only, grouped by manifest source):
  - `outputs/diagnostics/bad_datasets/source_breakdown__CE__eval_only__clahe_on/`
  - Key file: `reliabilitymetrics_by_source.json`

Error sampling lists (for manual inspection):

- ExpW full (Fear/Disgust):
  - `outputs/diagnostics/bad_datasets/error_samples__CE__expw_full__fear_disgust__confident_wrong.csv`
  - `outputs/diagnostics/bad_datasets/error_samples__CE__expw_full__fear_disgust__ambiguous_wrong.csv`
- FER2013 uniform-7 (Fear):
  - `outputs/diagnostics/bad_datasets/error_samples__CE__fer2013_uniform7__fear__confident_wrong.csv`
- Eval-only, restricted to `expw_hq` (Fear/Disgust):
  - `outputs/diagnostics/bad_datasets/error_samples__CE__eval_only__expw_hq__fear_disgust__confident_wrong.csv`

## A.5 Paper comparison: evidence index and source notes

This appendix subsection contains the long-form evidence index for Section 9.3.6.

Key paper evidence sources (local PDF text extraction outputs):

- AffectNet paper text: `outputs/paper_extract/AffectNet A Database for Facial Expression, Valence, and Arousal Computing in the Wild.txt`
- FER2013 SOTA paper text: `outputs/paper_extract/Facial Emotion Recognition State of the Art Performance on FER2013.txt`
- Face-regions paper text: `outputs/paper_extract/Expression Analysis Based on Face Regions in Read-world Conditions.txt`

Evidence index maintenance rule:

- Add one new row per evaluation target.
- Always link the manifest (source of truth) and at least one metrics JSON artifact.
- Mark Comparable as `Yes` only when split + protocol match the paper (otherwise `Partial`/`No`).

Table A.5-1 (Evidence index). Purpose: map each evaluation claim to its manifest + checkpoint + metrics artifact.

| Evaluation target | Dataset / split | Manifest (source of truth) | Checkpoint(s) evaluated | Metrics artifact(s) | Comparable? |
| --- | --- | --- | --- | --- | --- |
| FER2013 official (PublicTest / PrivateTest) | Official split from `fer2013.csv` (Usage=PublicTest/PrivateTest); protocol-aware (single-crop + ten-crop) | `Training_data/FER2013_official_from_csv/manifest__publictest.csv` and `.../manifest__privatetest.csv` | Student CE/KD/DKD (best DKD: `DKD_20251229_223722`) | Summary: `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`; raw JSONs: `outputs/evals/students/fer2013_official__*__*test__20260212__{singlecrop,tencrop}/reliabilitymetrics.json` | Partial (closest match; remaining differences may include preprocessing/alignment/model/training) |
| FER2013 (folder dataset; msambare) | Non-official folder packaging (stress-test) | `Training_data/fer2013_folder_manifest.csv` | Student DKD: `DKD_20251229_223722` | `outputs/evals/students/DKD_20251229_223722__fer2013_folder__test__20260208/reliabilitymetrics.json` | No |
| RAF-DB basic (student) | Single-dataset test (protocol may differ vs paper) | (from offline suite index) | Student CE: `CE_20251223_225031` | Offline suite CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | Partial |
| FERPlus (student) | Single-dataset test (protocol may differ vs paper) | (from offline suite index) | Student CE/KD/DKD | Offline suite CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | Partial |
| AffectNet balanced (student) | Balanced subset test (not paper’s original skew) | (from offline suite index) | Student CE: `CE_20251223_225031` | Offline suite CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | No (balanced-subset mismatch) |
| ExpW (cross-dataset gate) | In-the-wild proxy test | `Training_data_cleaned/expw_full_manifest.csv` | Student DKD variants (incl. `DKD_20251229_223722`) | `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md` and per-run `outputs/evals/students/*/reliabilitymetrics.json` | No |
| Eval-only (safety gate) | Mixed-source stress test (deployment-aligned) | `Training_data_cleaned/classification_manifest_eval_only.csv` | Student CE baseline and adapted runs | Gate metrics JSONs: `outputs/evals/students/*__eval_only__test__*/reliabilitymetrics.json` | No |
| Mixed-source benchmark (teachers / ensemble) | Mixed-domain benchmark (48,928 rows) | `Training_data_cleaned/test_all_sources.csv` | Teacher ensemble RN18/B3/CNXT (0.4/0.4/0.2) | `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json` | No |
