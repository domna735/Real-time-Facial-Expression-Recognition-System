# Real-time Facial Expression Recognition System: Interim Report (Turnitin Version)

**Project Title:** Real-time Facial Expression Recognition System via Knowledge Distillation and Nested Learning  
**Author:** Donovan Ma  
**Institution:** HKpolyU  
**Supervisor:** Prof. Lam  
**Report Period:** Aug 2025 – Dec 2025  
**Document Date:** Dec 24, 2025  
**Report Version:** 3.0

---

## Abstract

**Background:** Real-time facial expression recognition (FER) must run under latency constraints while remaining robust to class imbalance, dataset noise, and domain shift. In this restart/reconstruction phase, the priority is to rebuild a reproducible training + distillation pipeline with artifact-based provenance (manifests, metrics JSONs, checkpoints).

**Objectives:**

- Build validated multi-source cleaned manifests in a canonical 7-class label space.
- Train strong teacher backbones and select an ensemble for better generalization.
- Train a deployable student (MobileNetV3-Large) using CE → KD → DKD with exported softlabels.
- Track calibration (NLL/ECE) and produce reproducible artifacts for every run.

**Methods:**

- Data cleaning and validation produce unified manifests under `Training_data_cleaned/` and validation reports under `outputs/`.
- Teachers (RN18 / B3 / CNXT) are trained with ArcFace-style protocol at img224 and export full provenance (`best.pt`, `history.json`, `reliabilitymetrics.json`, `calibration.json`, `alignmentreport.json`).
- Ensembles are evaluated via weighted logit fusion and exported as softlabels (`softlabels.npz` + `softlabels_index.jsonl`) for student KD/DKD.
- Student training is orchestrated on Windows via PowerShell runner and evaluates accuracy, macro-F1, per-class F1, and calibration metrics.

**Results (artifact-grounded):**

- **Manifest integrity:** `Training_data_cleaned/classification_manifest.csv` has **466,284** rows with **0 missing paths** and **0 bad labels** (validated in `outputs/manifest_validation_all_with_expw.json`).
- **Teachers (Stage A, img224, CLAHE):** macro-F1 is **0.7808–0.7910** on the recorded validation split (18,165 images) from each run’s `reliabilitymetrics.json`.
- **Ensemble selection:** best multi-source robustness on `test_all_sources.csv` (48,928 images) is the 3-teacher ensemble **RN18/B3/CNXT = 0.4/0.4/0.2** with **macro-F1 0.659608** (artifact `ensemble_metrics.json`).
- **Student (HQ-train evaluation):** CE achieves **macro-F1 0.741952**; KD and DKD do not surpass CE in this first run, but **temperature-scaled ECE improves substantially** (KD/DKD ≈ 0.027 vs CE ≈ 0.050).
- **Reproducibility incident:** DKD previously produced an “empty output” due to resume epoch > configured total epochs; the runner was patched so DKD trains **additional epochs after resume**, and DKD outputs were successfully produced.

**Conclusions:** The pipeline is now artifact-grounded end-to-end: validated manifests, teacher training, ensemble softlabel export, and student CE/KD/DKD runs with consistent metrics JSONs. The next milestone is (1) evaluate student models on the mixed-source test benchmark and (2) run a timed real-time demo to report FPS/latency/flip-rate from CSV logs.

**Keywords:** facial expression recognition, knowledge distillation, decoupled knowledge distillation, calibration, expected calibration error, multi-source dataset, reproducibility, real-time inference

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

Real-time FER aims to classify a person’s facial expression from video frames under strict latency and compute constraints. The key tension is that high-capacity models (and especially ensembles) can achieve better accuracy and robustness, but are often too heavy for real-time deployment. This motivates the teacher→student approach: train strong teachers offline, then distill knowledge into a compact student.

### 1.1 Problem context

Key challenges for deployment-oriented FER in this project:

- **Imbalance and hard classes:** minority / subtle classes (e.g., Fear, Disgust) are frequently confused.
- **Calibration:** overconfident errors reduce trust and complicate downstream decision making.
- **Domain shift:** offline static-image performance may not match webcam performance due to lighting, camera noise, and temporal dynamics.
- **Reproducibility:** multi-stage pipelines (cleaning → training → softlabel export → distillation → demo) require explicit provenance to avoid silent drift.

### 1.2 Problem definition

This project targets a practical deployment question:

- **Input:** a live video stream (webcam/video) containing a detectable face.
- **Output:** one of **7 canonical expressions** (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral), updated in real time.
- **Constraints:** low latency, stable (non-flickering) predictions, and reliability of confidence scores (calibration).

Formally, given a face crop sequence $\{x_t\}$, we aim to learn a function $f(x_t)$ that outputs class probabilities $p_t \in [0,1]^7$ with:

- strong classification performance (accuracy, macro-F1, per-class F1), especially for hard/minority classes
- well-calibrated confidence (low NLL/ECE) to support thresholding and safe downstream use
- real-time feasibility on the target machine

### 1.3 Canonical baselines (this report’s scope)

Important note: the results in this report are measured on **different evaluation manifests** depending on the component. Every table below explicitly states the evaluation source.

Teacher (single-model) summary (Stage A val split from `Training_data_cleaned/classification_manifest.csv` after filtering; val=18,165):

| Role | Model | Accuracy | Macro-F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Teacher | RN18 | 0.7862 | 0.7808 | Stage A img224; CLAHE |
| Teacher | B3 | 0.7961 | 0.7910 | Stage A img224; CLAHE |
| Teacher | CNXT | 0.7941 | 0.7890 | Stage A img224; CLAHE |

Ensemble selection summary (mixed-source benchmark `Training_data_cleaned/test_all_sources.csv`, n=48,928):

| Role | Model | Accuracy | Macro-F1 | Weights |
| --- | --- | ---: | ---: | --- |
| Teacher ensemble | RN18/B3/CNXT | 0.687255 | 0.659608 | 0.4 / 0.4 / 0.2 |

Student summary (HQ-train evaluation from `Training_data_cleaned/classification_manifest_hq_train.csv` test split; test=27,840):

| Role | Model | Accuracy | Macro-F1 | Notes |
| --- | --- | ---: | ---: | --- |
| Student | CE | 0.750174 | 0.741952 | img224; CLAHE+AMP |
| Student | KD | 0.734688 | 0.733351 | T=2, α=0.5 |
| Student | DKD | 0.737432 | 0.737511 | T=2, α=0.5, β=4 |

### 1.4 Objectives (Dec 2025 checkpoint)

- Keep all experiments reproducible (saved manifests, run folders, checkpoints, metrics JSONs).
- Use a single canonical 7-class mapping across all datasets.
- Select teacher ensemble weights based on a mixed-source benchmark rather than a single dataset.
- Produce a student that is deployable and well-calibrated (report raw + temperature-scaled NLL/ECE).


---

## 2. Literature Review

This section consolidates prior work and motivates the design choices used in this restart phase: multi-source dataset handling, imbalance-aware evaluation, efficient architectures, metric-learning style teacher training, knowledge distillation, calibration, and real-time deployment considerations.

### 2.1 Datasets and labeling in the wild

- **From controlled to in-the-wild:** Early FER relied on posed, studio-like images with high accuracy but weak generalization. The shift to unconstrained settings introduced pose, occlusion, and illumination variation. RAF-DB and related works established reliable crowdsourcing protocols for in-the-wild FER [21,23]. FERPlus extended FER2013 by using multiple annotations per image and label distributions [24]. AffectNet scaled FER to a large in-the-wild corpus with strong imbalance and noise [20]. ExpW broadened in-the-wild coverage but inherits automated labeling noise [22]. EmotioNet demonstrated large-scale automatic annotation and real-time speed considerations [25].
- **Reproducibility and integrity:** In-the-wild datasets carry non-trivial label and path errors; distribution-aware calibration and validation practices are often underreported [32].
- **Gap:** Cross-dataset label conventions and domain shift (static images vs real-time webcam) can dominate apparent model differences if the pipeline is not controlled.
- **This project’s approach (restart):** Enforce a canonical 7-class mapping and validate a unified manifest using explicit integrity checks (path existence + decode sampling) before any training.

### 2.2 Long-tail learning and imbalance remedies

- **Foundations:** Long-tail methods address imbalance via loss reweighting, sampling, and decision-boundary adjustments. Focal Loss emphasizes hard examples [11]. Class-Balanced Loss weights by the effective number of samples [12]. Logit adjustment corrects label-prior bias [14], with modern adaptive variants [33]. CIFAR baselines contextualize imbalance sensitivity [13].
- **FER relevance:** Minority expressions (e.g., Fear/Disgust) are frequently subtle and under-represented, so accuracy-only evaluation can hide failure modes.
- **Gap:** Many FER reports optimize accuracy while under-reporting macro-F1 and per-class performance.
- **This project’s approach (restart):** Treat macro-F1 and per-class F1 as primary metrics alongside accuracy, and benchmark robustness on mixed-source test data rather than a single dataset split.

### 2.3 Architectures: efficiency, robustness, and calibration

- **CNN backbones:** ResNet’s residual learning underpins stable ConvNet training [8,9]. EfficientNet demonstrates compound scaling for accuracy–efficiency trade-offs [5]. ConvNeXt modernizes ConvNets with transformer-inspired design, often improving representation quality but not guaranteeing calibration [7]. Attention modules (e.g., CBAM) are commonly used to refine feature emphasis when needed [10].
- **Mobile deployment:** MobileNetV3 is a strong latency/accuracy option for edge-style deployment [6]. In this project, MobileNetV3-Large is used as the student due to practical real-time constraints.
- **Vision Transformers:** DeiT shows transformers can benefit from distillation [30], and ViT scales with data/resolution [31]. In FER, the data regime and fine-grained facial cues may make CNNs and efficient hybrids more reliable.
- **Gap:** Strong offline accuracy does not imply robust cross-domain performance or calibrated confidence.

### 2.4 Metric learning with ArcFace and its calibration

- **ArcFace objective:** Additive angular margin on a hypersphere encourages inter-class separation [4]:

$$L_{ArcFace} = -\frac{1}{n} \sum_{i=1}^n \log \frac{e^{s \cos(\theta_{y_i,i} + m)}}{e^{s \cos(\theta_{y_i,i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_{j,i}}}$$

- **Calibration interaction:** Temperature scaling improves reliability of predicted confidence without changing the predicted class [16,34]. ArcFace-style training can increase logit magnitudes, which often benefits from post-hoc calibration.
- **This project’s approach (restart):** Train teachers with an ArcFace-style protocol and report both raw and temperature-scaled calibration metrics (NLL/ECE) for transparency.

### 2.5 Knowledge distillation: classical and decoupled

- **Classical KD:** Student minimizes a mixture of hard-label CE and KL divergence to softened teacher targets [1]:

$$L_{KD} = (1-\alpha) \mathcal{L}_{CE}(y, \sigma(z_s)) + \alpha T^2 \cdot \text{KL}(\sigma(z_t/T) \| \sigma(z_s/T))$$

- **Decoupled KD (DKD):** DKD separates target-class and non-target-class knowledge so their influence can be weighted independently [2]:

$$L_{DKD} = (1-\alpha)\mathcal{L}_{CE} + \alpha T^2 \mathcal{L}_{TCKD} + \beta T^2 \mathcal{L}_{NCKD}$$

- **This project’s approach (restart):** Use a staged CE → KD → DKD student training workflow while tracking both classification and calibration metrics after training.

### 2.6 Multi-teacher distillation and ensembles

- **Why ensembles help:** Different backbones capture complementary cues; combining them can improve robustness under domain shift.
- **Typical approaches:** averaging or logit fusion (teacher ensemble), and using ensemble outputs to supervise a student via KD.
- **This project’s approach (restart):** Select teacher ensemble weights using a mixed-source benchmark, then export softlabels (`softlabels.npz` and index) so student KD/DKD does not require running teachers during student training.

### 2.7 Meta-optimizers and negative learning (advanced directions)

- **Nested Learning (NL):** Meta-optimizers with associative memory aim to adapt update dynamics [3], but can introduce substantial compute/memory overhead in practical vision training.
- **Negative/complementary learning:** Learning from complementary or negative labels can improve robustness under label noise [18,19].
- **Status in this project:** NL/NegL are treated as planned research directions (scaffolded notes), after stabilizing the core teacher–student pipeline.

### 2.8 Calibration and uncertainty

- **Why calibration matters:** Confidence values are used in thresholding, abstention, and downstream decision making; miscalibration can cause overconfident errors.
- **Foundations:** Temperature scaling is a common post-hoc calibration method [16]; probability quality and calibration have long-standing theory [34]. OOD/misclassification detection motivates reliable scores [15].
- **This project’s approach (restart):** Report raw and temperature-scaled NLL/ECE for teachers and students, and treat calibration as a first-class outcome alongside macro-F1.

### 2.9 Selective prediction and abstention

- **Concept:** Predict only when confidence exceeds a threshold; otherwise abstain. Calibration improves the meaning of confidence thresholds [16].
- **Relevance to FER:** Real-time FER may benefit from conservative decisions (e.g., “unknown/abstain” in ambiguous frames), especially under domain shift.

### 2.10 Real-time deployment: closing the offline–online gap

- **Preprocessing consistency:** Matching training-time preprocessing (e.g., CLAHE [26]) to the real-time pipeline can reduce deployment drift.
- **Face detection:** Practical systems require robust and efficient detectors; YuNet is a representative fast face detector [28].
- **Temporal stabilization:** Simple smoothing/hysteresis/voting can reduce rapid label flips and improve user-perceived stability.

### 2.11 Summary of gaps and motivations

- **Data:** in-the-wild noise and domain shift motivate validation gates and mixed-source benchmarking.
- **Metrics:** macro-F1, per-class F1, and calibration metrics help expose failure modes that accuracy alone hides.
- **Modeling:** teacher ensembles can improve robustness; students enable deployment; DKD provides a principled extension to KD.
- **Deployment:** preprocessing consistency and temporal stabilization are necessary for real-time usability.

---

## 3. Methodology

This section summarizes the pipeline as implemented in this repository, and points to the artifact files used for verification.

### 3.0 Pipeline overview (teacher → ensemble → softlabels → student → demo)

The end-to-end workflow for this restart phase is:

```mermaid
flowchart LR
  A[Cleaned manifests\n(7-class mapping + validation)] --> B[Teacher training\nRN18 / B3 / CNXT]
  B --> C[Ensemble selection\nweighted logit fusion]
  C --> D[Softlabels export\nsoftlabels.npz + index]
  D --> E[Student training\nCE → KD → DKD]
  E --> F[Real-time demo\nwebcam/video inference]
```

### 3.1 Dataset cleaning and validation

- **Canonical label space (7 classes):** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral.
- **Primary unified manifest:** `Training_data_cleaned/classification_manifest.csv`.
- **Validation:** `outputs/manifest_validation_all_with_expw.json` confirms 466,284 rows with 0 missing paths.

Source composition (rows) from `outputs/manifest_validation_all_with_expw.json`:

- fer2013_uniform_7: 140,000
- ferplus: 138,526
- expw_full: 91,793
- affectnet_full_balanced: 71,764
- rafdb_basic: 15,339
- rafml_argmax: 4,908
- rafdb_compound_mapped: 3,954

### 3.2 Teacher training (Stage A img224)

- Backbones: RN18 (`resnet18`), B3 (`tf_efficientnet_b3`), CNXT (`convnext_tiny`).
- Training protocol: ArcFace-style margins with warmup and ramp; CLAHE enabled.
- Run folders store: `best.pt`, `history.json`, `reliabilitymetrics.json`, `calibration.json`, and `alignmentreport.json`.

Evaluation dataset for the teacher metrics in this report (per `alignmentreport.json`):

- Manifest: `Training_data_cleaned/classification_manifest.csv`
- After filter: total 225,629; train 182,960; val 18,165
- Sources present after filter: affectnet_full_balanced, ferplus, rafdb_basic

### 3.3 Ensemble selection and softlabel export

- Fusion method: weighted **logit fusion** + softmax.
- Selected weights (based on `test_all_sources.csv`): RN18/B3/CNXT = 0.4/0.4/0.2.
- Student KD/DKD uses exported teacher logits; teachers are not executed during student training.

Selected softlabel export folder used for student KD/DKD:

- `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856/`

Expected files:

- `softlabels.npz`
- `softlabels_index.jsonl`
- `alignmentreport.json`
- `hash_manifest.json`
- `classorder.json`
- `ensemble_metrics.json`

### 3.4 Student training (CE → KD → DKD)

- Student: MobileNetV3-Large (`mobilenetv3_large_100` via timm).
- Data: `Training_data_cleaned/classification_manifest_hq_train.csv` (total 259,004; train 213,144; val 18,020; test 27,840).
- Orchestration: `scripts/run_student_mnv3_ce_kd_dkd.ps1`.

KD settings (this run):

- T=2, α=0.5, 20 epochs

DKD settings (this run):

- T=2, α=0.5, β=4
- Resume from KD `best.pt` and train 10 additional epochs (runner ensures epochs extend beyond resume)

### 3.5 Evaluation protocol

Metrics are stored as JSONs per run:

- Accuracy, macro-F1, per-class F1
- NLL, ECE, Brier (where available)
- Temperature scaling results (global T and post-calibration metrics)

Key point for interpretation:

- Teachers, ensembles, and student stages may be evaluated on **different manifests** (teacher val split vs mixed-source benchmark vs HQ-train test). This report always states the evaluation source for each table and avoids claiming a single “best overall model” without controlling the evaluation set.

### 3.6 Ethical considerations

This project uses publicly available FER datasets and a real-time webcam demo. Key ethical considerations are:

- **Privacy & consent:** real-time webcam inference should be used only with informed consent and should avoid storing identifiable frames unless strictly necessary.
- **Bias & fairness:** expression datasets can be demographically imbalanced; performance may vary across age/ethnicity/lighting conditions. Mixed-source validation is used to reduce single-dataset bias, but fairness auditing remains future work.
- **Misuse risk:** FER can be used in sensitive contexts; this work is positioned as a technical study and should not be deployed for high-stakes decision making without additional safeguards.
- **Transparency:** calibration metrics (NLL/ECE) are reported to reflect confidence reliability rather than accuracy-only reporting.

---

## 4. Results & Analysis

This section consolidates the key results from the mini-reports (01–05, 07–08).

### 4.1 Dataset integrity

From `outputs/manifest_validation_all_with_expw.json`:

- Total rows: 466,284
- Missing paths: 0
- Bad labels: 0

This validation is treated as a hard gate before training runs.

### 4.2 Teacher performance (Stage A, img224)

Evaluation: teacher run’s validation set (18,165) as recorded in each `reliabilitymetrics.json`.

| Teacher | Accuracy | Macro-F1 | Raw NLL | Temp NLL | Raw ECE | Temp ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RN18 | 0.7862 | 0.7808 | 4.025883 | 0.880346 | 0.205298 | 0.148851 |
| B3 | 0.7961 | 0.7910 | 3.221890 | 0.787123 | 0.198786 | 0.083927 |
| CNXT | 0.7941 | 0.7890 | 3.101407 | 0.769976 | 0.200896 | 0.081701 |

Per-class F1 (raw; same ordering as the canonical 7-class label space):

| Class | RN18 | B3 | CNXT |
| --- | ---: | ---: | ---: |
| Angry | 0.735663 | 0.752069 | 0.768721 |
| Disgust | 0.694022 | 0.715569 | 0.719442 |
| Fear | 0.763499 | 0.757627 | 0.739450 |
| Happy | 0.897026 | 0.919703 | 0.913479 |
| Sad | 0.741481 | 0.747881 | 0.731278 |
| Surprise | 0.818592 | 0.804170 | 0.806429 |
| Neutral | 0.815514 | 0.839894 | 0.843916 |

Observation: temperature scaling meaningfully improves NLL/ECE for all teachers in these runs (each run reports a global temperature of 5.0).

### 4.3 Ensemble selection (robustness benchmark)

Benchmark: `Training_data_cleaned/test_all_sources.csv` (48,928 images).

Selected ensemble:

- RN18/B3/CNXT = 0.4/0.4/0.2
- Accuracy 0.687255; macro-F1 0.659608

Note: this benchmark is intentionally mixed-source and harder than a single-dataset test.

### 4.4 Student performance (HQ-train evaluation)

Evaluation: HQ-train manifest evaluation as recorded in each student run’s `reliabilitymetrics.json`.

| Student stage | Accuracy | Macro-F1 | Raw NLL | Temp NLL | Raw ECE | Temp ECE | Global T |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.750174 | 0.741952 | 1.315335 | 0.777757 | 0.131019 | 0.049897 | 3.228 |
| KD | 0.734688 | 0.733351 | 2.093148 | 0.768196 | 0.215289 | 0.027764 | 5.000 |
| DKD | 0.737432 | 0.737511 | 1.511788 | 0.765203 | 0.209450 | 0.026605 | 3.348 |

Per-class F1 (raw):

| Class | CE | KD | DKD |
| --- | ---: | ---: | ---: |
| Angry | 0.726340 | 0.723717 | 0.725522 |
| Disgust | 0.642839 | 0.678227 | 0.682833 |
| Fear | 0.764029 | 0.744691 | 0.756052 |
| Happy | 0.801425 | 0.760978 | 0.759617 |
| Sad | 0.716981 | 0.723361 | 0.728567 |
| Surprise | 0.787086 | 0.780076 | 0.791491 |
| Neutral | 0.754961 | 0.722405 | 0.718493 |

Observation: KD/DKD improve temperature-scaled calibration (NLL/ECE) but do not yet improve macro-F1 over CE in this first run. Raw KD/DKD calibration is worse than CE, suggesting the student is overconfident without post-hoc scaling.

### 4.5 Reproducibility incident: DKD empty output

- Root cause: DKD resume `start_epoch` exceeded configured `--epochs` causing a no-op training loop.
- Fix: compute DKD total epochs as `(resume_epoch + 1 + extra_dkd_epochs)` in the runner.
- Status: validated by a successful DKD rerun producing full artifacts.

---

## 5. Demo and Application

### 5.1 System architecture (real-time pipeline)

The demo pipeline is implemented in `demo/realtime_demo.py` and follows:

1. Frame capture (webcam/video)
2. Face detection
3. Face crop + resize to img224
4. Preprocessing (including CLAHE when enabled)
5. FER inference (logits/probabilities)
6. Temporal stabilization (EMA / hysteresis / voting)
7. Visualization + CSV logging to `demo/outputs/`

### 5.2 Performance evaluation (current status)

Current demo status (Dec 24, 2025):

- **Teacher inference:** confirmed working in the demo pipeline (real-time inference + CSV logging).
- **Student inference:** planned next; needs a dedicated demo run using the student checkpoint and (optionally) the saved temperature scaling parameters.
- **Stability logic:** temporal smoothing + hysteresis/voting are implemented to reduce flicker.

A timed demo run is still required to report real FPS/latency/flip-rate. This interim report does **not** claim those numbers because no timed CSV log from a fixed-length run is attached here.

Next-step measurement plan:

- Run demo 2–3 minutes on target machine.
- Compute FPS, latency (mean/median), label flip-rate/min from `demo/outputs/*.csv`.

---

## 6. Discussion and Limitations

- **Different evaluation distributions:** teachers, ensembles, and student stages currently report metrics on different manifests; direct comparisons must control for evaluation set.
- **Student macro-F1 not yet improved by KD/DKD:** suggests that current hyperparameters and/or training length may be favoring calibration over raw macro-F1.
- **Mixed-source benchmark is harder:** ensemble macro-F1 on `test_all_sources` is substantially lower than teacher val macro-F1; this is expected under domain shift and stricter benchmarking.
- **Real-time KPIs missing:** demo is implemented and logs outputs, but performance numbers require a timed run.
- **NL/NegL status:** currently scaffolded as research notes; no integrated training results are claimed yet.

---

## 7. Project Timeline (updated)

This timeline is the forward plan for the remainder of the project (Jan–Apr 2026), based on the current Dec-24 restart artifacts.

| Month | Focus | Deliverables (evidence) |
| --- | --- | --- |
| Jan 2026 | Student tuning | 2–3 additional student runs (CE/KD/DKD) with controlled evaluation on `test_all_sources.csv`; updated report tables |
| Feb 2026 | NL/NegL experiments | One minimal NL/NegL baseline integrated into training loop; measurable results + ablation note |
| Mar 2026 | Real-time demo KPIs | Timed demo run(s) with FPS/latency/flip-rate summary from `demo/outputs/*.csv`; stability settings documented |
| Apr 2026 | Final report | Consolidated evaluation on a consistent benchmark + final report write-up and packaging |

---

## 8. Lessons Learned from Development

- **Artifact-grounded workflow prevents silent drift:** every training stage should emit manifests, metrics JSONs, and checkpoints.
- **Validation gates are non-negotiable:** path existence and decode sampling avoid wasting training on corrupted inputs.
- **Resume semantics matter for multi-stage training:** DKD must guarantee `total_epochs > start_epoch`, otherwise “successful” runs can produce empty outputs.
- **Windows stability:** use safer DataLoader settings and PowerShell orchestration for reproducible runs.

---

## 9. Conclusion and Next Steps

### 9.1 Key achievements (Dec 24, 2025)

- Unified cleaned manifest validated at 466,284 rows with 0 missing paths.
- Teachers trained and evaluated with full provenance artifacts.
- Ensemble weights selected on a mixed-source benchmark; softlabels exported and used for student KD/DKD.
- Student CE/KD/DKD pipeline runs end-to-end and outputs full metrics and calibration artifacts.
- DKD resume bug diagnosed, fixed, and validated.

### 9.2 Immediate next steps

- Evaluate CE/KD/DKD students on `Training_data_cleaned/test_all_sources.csv` for mixed-source generalization.
- Run a timed real-time demo and report FPS/latency/flip-rate from CSV logs.
- Tune KD/DKD (`T`, `α`, `β`) and/or train longer, aiming for macro-F1 gains without losing calibration.
- Define a minimal NL/NegL experiment matrix and run a first measurable baseline.

---

## 10. References

This v3 report is focused on updated, artifact-grounded results. The reference list below is carried forward (and lightly cleaned) from the v2 interim report as background reading for the methods used.

### 10.1 Citation-to-reference consistency check

The following consistency check scans Sections 1–9 for numeric citations in the form `[n]` and `[n,m,...]`, and compares them against the numbered entries in this References section.

| Item | Result |
| --- | --- |
| References listed | 34 items (`[1]`–`[34]`) |
| In-text citations used | 31 unique items: `[1] [2] [3] [4] [5] [6] [7] [8] [9] [10] [11] [12] [13] [14] [15] [16] [18] [19] [20] [21] [22] [23] [24] [25] [26] [28] [30] [31] [32] [33] [34]` |
| Used-but-missing in References | None |
| Listed-but-not-cited | `[17] [27] [29]` |
| Notes | The token `[0,1]^7` in Section 1.2 is a probability range, not a citation. |

### Core Methodologies

[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," in Proc. NIPS Deep Learning and Representation Learning Workshop, Montreal, QC, Canada, 2015. [Online]. Available: arXiv:1503.02531

[2] B. Zhao, Q. Cui, R. Song, Y. Qiu, and J. Liang, "Decoupled Knowledge Distillation," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), New Orleans, LA, USA, 2022, pp. 11953–11962.

[3] C. Deng, D. Huang, X. Wang, and M. Tan, "Nested Learning: A New Paradigm for Machine Learning," arXiv preprint arXiv:2303.10576, 2023.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Long Beach, CA, USA, 2019, pp. 4690–4699.

### Architectures

[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in Proc. Int. Conf. Mach. Learn. (ICML), Long Beach, CA, USA, 2019, pp. 6105–6114.

[6] A. Howard, M. Sandler, G. Chu, et al., "Searching for MobileNetV3," in Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), Seoul, South Korea, 2019, pp. 1314–1324.

[7] Z. Liu, H. Mao, C.-Y. Wu, et al., "A ConvNet for the 2020s," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), New Orleans, LA, USA, 2022, pp. 11974–11984.

[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Las Vegas, NV, USA, 2016, pp. 770–778.

[9] K. He, X. Zhang, S. Ren, and J. Sun, "Identity Mappings in Deep Residual Networks," in Proc. Eur. Conf. Comput. Vis. (ECCV), Amsterdam, The Netherlands, 2016, pp. 630–645.

[10] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional Block Attention Module," in Proc. Eur. Conf. Comput. Vis. (ECCV), Munich, Germany, 2018, pp. 3–19.

### Long-Tail and Imbalance

[11] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), Venice, Italy, 2017, pp. 2980–2988.

[12] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-Balanced Loss Based on Effective Number of Samples," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), Long Beach, CA, USA, 2019, pp. 9268–9277.

[13] A. Krizhevsky and G. Hinton, "Learning Multiple Layers of Features from Tiny Images," Univ. Toronto, Tech. Rep., 2009.

[14] A. Menon, S. Jayasumana, A. S. Rawat, et al., "Long-Tail Learning via Logit Adjustment," in Proc. Int. Conf. Learn. Represent. (ICLR), Virtual, 2021.

### Calibration and Uncertainty

[15] D. Hendrycks and K. Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks," in Proc. Int. Conf. Learn. Represent. (ICLR), Toulon, France, 2017.

[16] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On Calibration of Modern Neural Networks," in Proc. Int. Conf. Mach. Learn. (ICML), Sydney, NSW, Australia, 2017, pp. 1321–1330.

[17] G. Pleiss, C. Guo, Y. Sun, Z. C. Lipton, A. Kumar, and K. Q. Weinberger, "On Fairness and Calibration," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Long Beach, CA, USA, 2017.

### Complementary/Negative Learning

[18] Y. Zhang, T. Liu, M. Long, and M. I. Jordan, "Learning with Negative Learning," in Proc. Int. Conf. Mach. Learn. (ICML), Long Beach, CA, USA, 2019, pp. 7329–7338.

[19] T. Ishida, G. Niu, W. Hu, and M. Sugiyama, "Learning from Complementary Labels," in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), Long Beach, CA, USA, 2017, pp. 5639–5649.

### FER Datasets and Benchmarks

[20] A. Mollahosseini, D. Chan, and M. H. Mahoor, "AffectNet: A Database for Facial Expression, Valence, and Arousal in the Wild," IEEE Trans. Affective Comput., vol. 10, no. 1, pp. 18–31, 2019.

[21] S. Li and W. Deng, "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Unconstrained Facial Expression Recognition," IEEE Trans. Image Process., vol. 28, no. 1, pp. 375–388, 2019.

[22] Z. Zhang, P. Luo, C.-C. Loy, and X. Tang, "From Facial Expression Recognition to Interpersonal Relation Prediction," Int. J. Comput. Vis. (IJCV), vol. 126, pp. 550–569, 2018.

[23] S. Li, W. Deng, and J. Du, "Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Honolulu, HI, USA, 2017.

[24] E. Barsoum, C. Zhang, C. C. Ferrer, and Z. Zhang, "Training Deep Networks for Facial Expression Recognition with Crowd-Sourced Label Distribution," in Proc. ACM Int. Conf. Multimodal Interaction (ICMI), Tokyo, Japan, 2016.

[25] C. F. Benitez-Quiroz, R. Srinivasan, and A. M. Martinez, "EmotioNet: An Accurate, Real-Time Algorithm for the Automatic Annotation of a Million Facial Expressions," in Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR), Las Vegas, NV, USA, 2016.

### Real-Time Processing and Deployment

[26] S. M. Pizer, E. P. Amburn, J. D. Austin, et al., "Adaptive Histogram Equalization and Its Variations," Comput. Vis., Graph., Image Process., vol. 39, no. 3, pp. 355–368, 1987.

[27] M. Liu, S. Li, S. Shan, and X. Chen, "Facial Expression Recognition via Deep Learning," IEEE Trans. Syst., Man, Cybern., Syst., vol. 47, no. 6, pp. 1011–1024, 2017.

[28] W. Wu, Y. He, S. Wang, et al., "YuNet: A Fast and Accurate Face Detector," arXiv:2111.04088, 2021.

[29] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019. [Online]. Available: <https://github.com/rwightman/pytorch-image-models>

### Additional Implementation and Evaluation References

[30] H. Touvron, M. Cord, M. Douze, F. Massa, A. Sablayrolles, and H. Jégou, "Training Data-Efficient Image Transformers & Distillation Through Attention," in Proc. Int. Conf. Mach. Learn. (ICML), 2021, pp. 10347–10357.

[31] A. Dosovitskiy, J. Beyer, A. Kolesnikov, et al., "An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale," in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.

[32] Y. Cui, L. Zhang, J. Wang, L. Lin, and S. Z. Li, "Distribution-Aware Calibration for In-the-Wild Recognition," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR) Workshops, 2021.

[33] S. Liu, Y. Wang, J. Long, et al., "Adaptive Logit Adjustment Loss for Long-Tailed Visual Recognition," in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2023, pp. 14668–14677.

[34] A. Niculescu-Mizil and R. Caruana, "Predicting Good Probabilities with Supervised Learning," in Proc. Int. Conf. Mach. Learn. (ICML), Bonn, Germany, 2005, pp. 625–632.

---

## 11. Appendix

### A.1 Mathematical formulations (used in this repo)

**Knowledge Distillation (KD):**

$$L_{KD} = (1-\alpha)L_{CE} + \alpha T^2 \cdot \frac{1}{n}\sum_{i=1}^n \text{KL}(p_{t,i}^T \Vert p_{s,i}^T)$$

where $p_{t,i}^T = \text{softmax}_T(\mathbf{t}_i)$, $p_{s,i}^T = \text{softmax}_T(\mathbf{z}_i)$. In the student run documented in this report: $\alpha=0.5$, $T=2$.

**Decoupled KD (DKD):**

$$L_{DKD} = (1-\alpha) L_{CE} + \alpha T^2 L_{TCKD} + \beta T^2 L_{NCKD}$$

In the DKD run documented in this report: $\alpha=0.5$, $\beta=4$, $T=2$.

**ArcFace Loss (teacher training):**

$$L_{ArcFace} = -\frac{1}{n} \sum_{i=1}^n \log \frac{e^{s \cos(\theta_{y_i,i} + m)}}{e^{s \cos(\theta_{y_i,i} + m)} + \sum_{j \neq y_i} e^{s \cos \theta_{j,i}}}$$

For the teacher runs summarized here: $m=0.35$, $s=30.0$.

**Calibration metrics:**

Expected Calibration Error (ECE):

$$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{n} \left| \text{acc}(B_b) - \text{conf}(B_b) \right|$$

Negative Log-Likelihood (NLL):

$$\text{NLL} = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{K} y_{i,c} \log p_{i,c}$$

### A.2 Dataset specifications (current artifacts)

All counts below are taken from the current project artifacts (24-12-2025 report pack).

**Unified cleaned manifest (includes ExpW):**

- Manifest: `Training_data_cleaned/classification_manifest.csv`
- Validation: `outputs/manifest_validation_all_with_expw.json`
- Total rows: **466,284**
- Missing paths: **0**
- Bad labels: **0**

**Source composition (rows) from `outputs/manifest_validation_all_with_expw.json`:**

- fer2013_uniform_7: **140,000**
- ferplus: **138,526**
- expw_full: **91,793**
- affectnet_full_balanced: **71,764**
- rafdb_basic: **15,339**
- rafml_argmax: **4,908**
- rafdb_compound_mapped: **3,954**

**HQ training manifest (used for the student CE/KD/DKD run in this report):**

- Manifest: `Training_data_cleaned/classification_manifest_hq_train.csv`
- Total rows: **259,004**
- Split sizes (from the student report pack): train **213,144**, val **18,020**, test **27,840**

**Mixed-source robustness test (used for ensemble selection):**

- Manifest: `Training_data_cleaned/test_all_sources.csv`
- Total rows: **48,928**

### A.3 Appendix: run and artifact map (current)

This table is meant to make every claim in the report traceable to a file.

| Component | Folder / file | What it contains |
| --- | --- | --- |
| Manifest validation | `outputs/manifest_validation_all_with_expw.json` | Total rows and integrity checks for `classification_manifest.csv` |
| Cleaning provenance | `Training_data_cleaned/clean_report.json` | Cleaning mode and dataset adapter provenance |
| Teacher RN18 run | `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/` | `best.pt`, `history.json`, `reliabilitymetrics.json`, `calibration.json`, `alignmentreport.json` |
| Teacher B3 run | `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/` | Same run artifacts as above |
| Teacher CNXT run | `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/` | Same run artifacts as above |
| Selected softlabels | `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856/` | `softlabels.npz`, `softlabels_index.jsonl`, `ensemble_metrics.json`, `hash_manifest.json`, `classorder.json`, `alignmentreport.json` |
| Ensemble benchmark metric (archived) | `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json` | Selected mixed-source benchmark score for 0.4/0.4/0.2 |
| Student CE run | `outputs/students/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/` | `best.pt`, `history.json`, `reliabilitymetrics.json`, `calibration.json` |
| Student KD run | `outputs/students/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/` | Same run artifacts as above |
| Student DKD run | `outputs/students/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/` | Same run artifacts as above |
| Demo logging outputs | `demo/outputs/` | Per-frame CSV logs produced by `demo/realtime_demo.py` |

### A.4 Reproducibility checklist (current)

- For any training run folder, verify the presence of: `best.pt`, `history.json`, `reliabilitymetrics.json`, `calibration.json` (and `alignmentreport.json` where applicable).
- For any distillation run, verify the exact softlabels directory used (should contain `softlabels.npz` + `softlabels_index.jsonl`).
- For Windows stability, keep DataLoader worker counts conservative (as used in the documented student run).
- DKD resume rule: always ensure `total_epochs > start_epoch`; the PowerShell runner was updated to compute DKD total epochs as `(resume_epoch + 1 + extra_dkd_epochs)`.

