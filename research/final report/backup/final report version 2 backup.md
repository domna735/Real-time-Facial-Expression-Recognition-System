# Real-time Facial Expression Recognition System: Final Report

Project Title: Real-time Facial Expression Recognition System via Knowledge Distillation and Self-Learning + Negative Learning (NegL)

Author: Donovan Ma  
Institution: HKpolyU  
Supervisor: Prof. Lam  
Report Period: Aug 2025 – Mar 2026  
Document Date: Mar 7, 2026  
Report Version: 2

---

## Abstract

Real-time facial expression recognition (FER) demands not only classification accuracy but also prediction stability, confidence calibration, and robustness to domain shift in deployment. This project implements a reproducible real-time FER system for the canonical 7-class emotion space using a teacher–student knowledge distillation pipeline. Three teacher backbones (ResNet-18, EfficientNet-B3, ConvNeXt-Tiny) are trained with ArcFace-style margins on 466,284 validated multi-source samples, achieving 0.781–0.791 macro-F1. A 3-teacher ensemble is distilled into a MobileNetV3-Large student via cross-entropy (CE), knowledge distillation (KD), and decoupled KD (DKD). KD/DKD consistently improve temperature-scaled calibration (ECE: 0.050→0.027) but do not surpass CE macro-F1 (0.742). Domain shift evaluation reveals persistent minority-class fragility (Fear/Disgust F1 near zero under webcam conditions). A safety-gated adaptation loop is implemented: adapted checkpoints pass offline regression gates but regress on webcam replay, establishing that offline non-regression is necessary but insufficient for deployment claims. Seven negative results are formally documented with artifact-backed evidence throughout. The key methodological contribution is a dual-gate evaluation framework requiring both offline non-regression and deployment-facing replay improvement before promoting any checkpoint.

## Executive Summary

**One-sentence summary:** A reproducible real-time FER pipeline is implemented using a teacher-student knowledge distillation design, evaluated with protocol-aware offline benchmarks and deployment-facing domain shift scoring.

**Deliverables:**

- Cleaned multi-source training and evaluation pipeline (466,284 validated samples across 7 emotion classes).
- Teacher training with three backbone architectures (ResNet-18, EfficientNet-B3, ConvNeXt-Tiny) achieving 0.781-0.791 macro-F1.
- Student distillation pipeline (CE, KD, DKD) targeting real-time CPU inference with MobileNetV3-Large.
- Real-time demo system with temporal stabilisation (EMA, hysteresis, vote window) and deployment-facing metrics (jitter flips/min).
- Protocol-aware paper comparison framework with explicit comparability flags.

**Key findings:**

- KD/DKD improve temperature-scaled calibration (TS ECE: 0.050 to 0.027) but do not surpass CE macro-F1 (0.742), demonstrating that calibration and decision-boundary quality are distinct optimisation targets.
- Domain shift causes severe minority-class degradation: teacher macro-F1 drops from 0.791 (in-distribution) to 0.393 on mixed-domain gates; Fear/Disgust reach near-zero F1 under webcam conditions.
- A safety-gated adaptation candidate passed offline regression but regressed on webcam replay, establishing that offline non-regression is necessary but insufficient for deployment claims.
- Fair comparison depends on protocol details (split definition, crop policy, preprocessing); RAF-DB accuracy (86.3%) is competitive while FER2013 official-split accuracy (61.4%) reflects the MobileNetV3 capacity-latency trade-off.

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

### 1.1 Contributions

This project makes the following contributions:

1. **Reproducible teacher–student FER pipeline.** An end-to-end, artifact-grounded pipeline covering multi-source data cleaning, teacher training (3 backbones), ensemble softlabel export, and student distillation (CE/KD/DKD) with stored JSON metrics at every stage.

2. **Dual-gate evaluation framework.** A deployment-aware evaluation protocol requiring both (a) offline non-regression on broad-distribution gates (eval-only, ExpW) and (b) improvement on fixed-protocol webcam replay, before promoting any checkpoint. This is motivated by the empirical finding that offline gate pass does not imply webcam improvement (NR-1).

3. **Systematic negative result documentation.** Seven formally catalogued negative results (NR-1–NR-7) with evidence-backed vs hypothesis classifications, covering adaptation failures, auxiliary loss instability, and calibration–accuracy decoupling.

4. **Domain shift characterisation.** Quantitative analysis of the teacher→student→deployment transfer gap, including per-class fragility analysis (Fear F1 = 0.00 under webcam shift) and root-cause diagnosis (CLAHE mismatch, BN running-stat drift).

5. **Real-time stabilisation analysis.** Deployment-facing metrics (EMA-smoothed accuracy/F1, jitter flips/min) reported alongside offline metrics, with analysis of how probability margin dynamics interact with temporal smoothing.

### 1.2 Relationship to Interim Report v4 (Dec 25, 2025)

This final report is a continuation of the project’s Dec-2025 interim deliverable:

- The interim report (v4, Dec 2025) is available as a separate submission

How it is integrated here:

- The interim report’s framing (deployment constraints, domain shift risk, and the need for artifact-grounded provenance) is preserved as the motivation for the pipeline design in Sections 1–3.
- All **numeric results** in this final report are still taken directly from run artifacts (JSON/compare tables) as listed in Sections 3–4.
- The Jan-2026 extensions (NL/NegL screening, webcam scoring protocol, and offline safety gate) extend beyond the interim report’s Dec-2025 scope.

## 2. Literature Review

This literature review expands the key research themes that motivate the system design choices in this project. The goal is not to reproduce any single paper’s method in full, but to explain the technical background needed to justify: (1) why the teacher→student pipeline is used, (2) why evaluation must be protocol-aware for “fair comparison”, and (3) why domain shift and deployment stability are first-class concerns in real-time FER.

### 2.1 Problem setting: FER is noisy, imbalanced, and ambiguous

Facial expression recognition (FER) typically maps a face crop to a discrete set of emotion categories (commonly 7 basic emotions). In practice, this mapping is inherently ambiguous:

- **Subtlety and intensity variation:** low-intensity expressions can be visually close to Neutral.
- **Class overlap:** Fear vs Surprise, Disgust vs Angry, and Sad vs Neutral are common confusions in-the-wild.
- **Annotation noise:** crowd-labeled in-the-wild datasets contain label noise and context ambiguity.

Because of these factors, accuracy alone can hide important failure modes when the dataset is class-imbalanced. Techniques such as focal loss [11], class-balanced loss [12], and logit adjustment [13] have been proposed to address long-tailed distributions. For this reason, macro-F1 and per-class F1 are commonly used to reveal minority-class brittleness.

### 2.2 Datasets and protocols: why “fair comparison” is hard

FER papers often report results on one curated dataset and a specific official split (or a defined cross-validation protocol). Real deployments (and this project) face a different reality:

- **Training distribution is a mixture:** multi-source data can be larger and more diverse, but also noisier.
- **Test distribution may shift:** webcam usage differs in sensor, lighting, pose, motion blur, and cropping jitter.

This makes protocol details essential for fair comparison:

- **Split definition:** official train/test splits vs folder-packaged splits.
- **Label mapping:** 7-class vs 8-class (or compound classes).
- **Test-time protocol:** single-crop vs multi-crop / test-time augmentation (TTA).
- **Preprocessing:** face alignment, cropping policy, image resolution, and histogram normalization can all change results.

In this project, the official FER2013 [20] PublicTest/PrivateTest evaluation is treated as the strongest anchor for “paper-like” comparison on FER2013, while mixed-source gates (eval-only / ExpW / test_all_sources) are treated as deployment-aligned stress tests.

### 2.3 Backbone architectures and attention: capacity vs efficiency vs calibration

Modern FER systems commonly use CNN backbones pretrained on large-scale image datasets. This project uses multiple backbone families for different roles:

- **Teachers (capacity-first):** ResNet-18 [8] / EfficientNet-B3 [5] / ConvNeXt-Tiny [7] are used as higher-capacity feature extractors.
- **Student (efficiency-first):** MobileNetV3-Large [6] is used for real-time constraints, loaded via the timm library [19].

Attention mechanisms are frequently used inside modern backbones to improve representational power:

- **Channel attention (SE-style):** reweights feature channels based on global context.
- **Spatial+channel attention (e.g., CBAM [10]):** can further focus on salient regions.

These modules can improve accuracy, but they also interact with calibration: models that become “sharper” can become overconfident. This is one reason calibration metrics (ECE/NLL) and temperature scaling are important when the system uses confidence thresholds (e.g., in self-learning or in UI decisions).

### 2.4 Knowledge Distillation (KD) and Decoupled KD (DKD): teacher→student transfer

Knowledge distillation trains a compact student model by combining:

- **Hard labels (cross-entropy):** encourage correct classification on ground-truth labels.
- **Soft targets (teacher probabilities/logits):** transfer teacher knowledge, including class similarity structure.

In typical KD [1], the teacher logits are softened with a temperature $T$ during training to provide a smoother target distribution. The standard KD loss combines a hard-label cross-entropy term with a soft-label KL-divergence term:

$$
\mathcal{L}_{\text{KD}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \alpha \cdot T^2 \cdot D_{\text{KL}}\!\left(\sigma\!\left(\frac{z_t}{T}\right) \;\middle\|\; \sigma\!\left(\frac{z_s}{T}\right)\right)
$$

where $z_s$ and $z_t$ are student and teacher logits respectively, $\sigma(\cdot)$ denotes the softmax function, $y$ is the ground-truth label, $\alpha$ is the interpolation weight between hard and soft targets, and $T$ is the distillation temperature. The $T^2$ factor compensates for the gradient magnitude reduction caused by the softened distributions.

However, vanilla KD can have trade-offs:

- The student may inherit teacher uncertainty patterns.
- Improvements in probability quality may not translate to macro-F1 gains on hard, shifted test sets.

Decoupled KD (DKD) [2] modifies the distillation objective so that target-class and non-target-class contributions can be weighted differently. DKD decomposes the KL-divergence into two parts:

$$
\mathcal{L}_{\text{DKD}} = \alpha_{\text{tckd}} \cdot \mathcal{L}_{\text{TCKD}} + \beta_{\text{nckd}} \cdot \mathcal{L}_{\text{NCKD}}
$$

where $\mathcal{L}_{\text{TCKD}}$ (Target-Class KD) handles the binary probability of the target class, and $\mathcal{L}_{\text{NCKD}}$ (Non-target-Class KD) handles the distribution over non-target classes. The motivation is that a student may need strong target-class learning ($\alpha_{\text{tckd}}$) while independently controlling how much of the teacher's non-target similarity structure to absorb ($\beta_{\text{nckd}}$).

In this project, KD/DKD are treated as engineering tools to improve efficiency and reliability, and are evaluated on multiple manifests to understand where the trade-offs appear.

### 2.5 Calibration and temperature scaling: why probability quality matters

Calibration measures whether predicted confidence aligns with empirical correctness. Two common metrics are:

- **ECE (Expected Calibration Error):** a binned estimate of confidence vs accuracy mismatch. Formally:

$$
\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|
$$

where $B_m$ is the set of samples in the $m$-th confidence bin, $\text{acc}(B_m)$ and $\text{conf}(B_m)$ are the average accuracy and average confidence within that bin, and $n$ is the total number of samples.

- **NLL (Negative Log-Likelihood):** penalizes overconfident wrong predictions more strongly.

Temperature scaling [14] is a simple post-hoc calibration method that rescales logits by a single scalar $T$ (without changing argmax labels):

- The system uses confidence thresholds (e.g., pseudo-label acceptance for self-learning).
- The UI or downstream logic depends on probability margins.
- Real-time smoothing/hysteresis reacts differently to “peaky” vs “flat” distributions.

In this project, calibration is reported alongside classification metrics because “real-time usable” behavior often depends on the confidence profile, not only top-1 accuracy.

### 2.6 Real-time FER: temporal stabilization and deployment KPIs

Real-time FER differs from offline image classification because predictions are made on a stream. A model that is accurate frame-by-frame can still be frustrating if it flickers rapidly between classes due to noise.

Common stabilization techniques include:

- **Exponential moving average (EMA) smoothing** over probability vectors.
- **Hysteresis** to resist switching classes unless the new class is sufficiently stronger.
- **Voting windows** over recent predictions.

These techniques change what is being optimized and what should be measured. Therefore, reporting both **raw** and **smoothed** behavior (plus stability metrics like flip-rate/jitter) is important for deployment-aligned evaluation.

### 2.7 Domain shift and adaptation: self-learning and negative learning need safety rails

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

### 2.8 Reference Linkage and Notation Discipline

To improve academic traceability, the key methodological claims in Sections 2.1-2.7 map to canonical references as follows:

- Distillation foundations and objective variants: KD [1] and DKD [2].
- Margin-based teacher training and backbone context: ArcFace [4], ResNet [8], [9], EfficientNet [5], ConvNeXt [7], MobileNetV3 [6].
- Calibration and confidence reliability: temperature scaling and calibration diagnostics [14].
- Negative/complementary learning context for auxiliary losses: [15], [16].
- Dataset and deployment context: FER2013 [20], AffectNet [17], RAF-DB and LP-loss [21], fast face detection context (YuNet) [18].
- Domain adaptation context: test-time adaptation [22].

### 2.9 Synthesis: why this project's design choices follow from the literature

The literature above motivates several non-obvious design decisions in this project. First, the persistent failure of minority classes (Fear/Disgust) under domain shift is a known structural challenge in FER [20, 21], not merely a tuning failure — this justifies treating per-class F1 and macro-F1 as primary metrics rather than overall accuracy. Second, while KD/DKD [1, 2] are well-established for model compression, their effect on calibration vs decision-boundary quality is not guaranteed to be aligned, which motivates our dual reporting of macro-F1 alongside ECE/NLL. Third, the test-time adaptation literature [22] warns that naïve entropy-based updates can fail under distribution shift; concrete failure mode analysis (e.g., SAR's entropy threshold $E_0 = 0.4 \times \ln(C)$ for $C$-class problems) informed our conservative gating design. Fourth, LP-loss [21] was selected as the most promising auxiliary loss for FER because it explicitly preserves local structure in the embedding space, addressing the class-similarity confusions (Fear↔Surprise, Disgust↔Angry) that dominate our error analysis. Finally, the gap between in-distribution validation and deployment-facing evaluation is not unique to this project; the domain adaptation literature consistently emphasises that proxy metrics must be complemented by target-domain evaluation, which motivates our dual-gate framework.

Notation rule used in this final report (for HTML/MathJax and docx conversion):

- Inline math uses `$...$` (for example, $T$, $p_{max}$, $\tau_{high}$).
- Display math uses `$$...$$` when equations are presented in block form.
- This notation is compatible with both MathJax rendering and pandoc markdown-to-docx conversion.

## 3. Methodology

This section summarizes the implemented pipeline and the artifacts that make every claim traceable.

### 3.0 Pipeline overview (figure)

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

### 3.1 Canonical label space

All training and evaluation in this report uses a 7-class mapping:

- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### 3.2 Data cleaning and manifest validation

All training and evaluation data are managed through CSV manifests with automated integrity validation (466,284 rows, 0 missing paths, 0 bad labels). Split sizes and label distributions are computed directly from the manifests and stored as reproducibility artifacts.

#### 3.2.1 Dataset provenance snapshots (Kaggle / Drive packaging)

Some datasets used in this project are downloaded via Kaggle or shared drives, and their folder packaging may not exactly match the “official split” definitions used in papers.

To make evaluation reproducible and audit-ready, the workflow snapshots the exact local dataset copies used in this workspace using:

Each snapshot records file counts, total bytes, extension counts, and a stable SHA256 fingerprint over the relative file list (without relying on external URLs).

Snapshots generated on 2026-02-09:

- RAF-DB basic, FER2013 folder packaging (msambare), and ExpW (in-the-wild) are each snapshotted with file counts, byte totals, and SHA256 fingerprints.

FER2013 official split note (paper-comparison support):

- An official-split evaluation was performed using a local Kaggle/ICML-format `fer2013.csv` (Usage=Training/PublicTest/PrivateTest). Because it is license-restricted, the dataset is not redistributed; instead, the following derived artifacts are stored for reproducibility:
  - Derived official manifests (PublicTest/PrivateTest) are stored for reproducibility
  - Per-split evaluation metrics (single-crop and ten-crop) are stored as JSON artifacts
  - A consolidated protocol-aware summary table is generated per Section 9.3.4

### 3.3 Teacher training (Stage A, img224)

Teachers are trained using an ArcFace-style protocol (margin + scale) and saved with full provenance.

Interpretation note (data):

- The Stage-A teacher runs read from the primary classification manifest (466,284 rows) and apply a source filter.
- Example (RN18): after filtering, the effective dataset is 225,629 rows, with sources `{ferplus: 138,526, affectnet_full_balanced: 71,764, rafdb_basic: 15,339}` and split sizes train=182,960 / val=18,165.
- Although `expw_hq` appears in the run’s `include_sources`, ExpW rows are not present in `source_counts_after_filter` for these Stage-A teacher runs.

Each teacher run produces a standardised output folder containing the best checkpoint, training history, reliability metrics (accuracy, macro-F1, per-class F1, ECE/NLL with temperature-scaled variants), and data alignment provenance.

### 3.4 Ensemble selection and softlabels export

Teacher ensembles are evaluated by weighted logit fusion on a mixed-source benchmark. The selected ensemble is exported as softlabels for student KD/DKD.

The selected ensemble (RN18/B3/CNXT = 0.4/0.4/0.2) is evaluated on the mixed-source test and exported as softlabels for student KD/DKD training.

### 3.5 Student training (CE → KD → DKD)

Student backbone:

- `mobilenetv3_large_100` (via timm)

HQ training manifest (verified size and splits):

- The HQ-train manifest has 259,004 rows with split sizes train=213,144 / val=18,020 / test=27,840.

Each student run (CE, KD, DKD) produces the same standardised output structure as teachers, with reliability metrics stored as JSON artifacts.

### 3.6 NL/NegL screening experiments (offline)

NL/NegL screening experiments are documented in dedicated planning and report files, with results summarised via standardised comparison tables. Each compared run is backed by stored reliability metrics.

Terminology note (to avoid confusion in academic reading):

- **NL(proto)** in this project refers to the *Nested Learning* [3] prototype-style auxiliary mechanism used in the Jan-2026 screening runs (an auxiliary objective with a gating/applied fraction).
- **NegL** in this project refers to an entropy-gated *complementary-label negative learning* loss (a “not-this-class” auxiliary loss).

These are distinct mechanisms; throughout this report, “NL” means Nested Learning, and “NegL” means complementary-label negative learning.

#### 3.6.1 Negative Learning (NegL): objective and gating

Intended objective: improve calibration (ECE/NLL) and reduce overconfident mistakes during KD/DKD by adding a **complementary-label negative learning (NegL)** term [16] that discourages probability mass on likely-wrong classes. Formally, given a complementary label $\bar{y}$ (a class that the sample is believed *not* to belong to), the NegL loss is:

$$
\mathcal{L}_{\text{NegL}} = -\log\!\left(1 - p(\bar{y} \mid x)\right)
$$

where $p(\bar{y} \mid x)$ is the model's predicted probability for the complementary label $\bar{y}$. This encourages the model to assign low probability to the specified wrong class.

Key design choice in this project is **gating**:

- NegL is not applied to every sample. A gate decides when NegL is active.
- Gate behavior is logged into the run’s `history.json` so it can be audited (example and numbers in Section 4.5.3).

Practical implication:

- A high entropy threshold can make NegL too selective (low `applied_frac`), reducing its effect.
- A low threshold / high weight can destabilize training and reduce macro-F1 (an instability counterexample is documented in Section 4.5.3).

#### 3.6.2 Nested Learning (NL(proto)): intent and configuration transparency

In the Jan-2026 screening runs, “NL(proto)” refers to the prototype-style auxiliary signal used in the compare tables in Section 4.5.

Reproducibility rule used in this report:

- The exact NL(proto) configuration (e.g., `dim`, `m`, `thr`, and/or `top-k`) is always taken from the compare artifact line and run identifier rather than being re-stated from memory.

### 3.7 Domain shift track (webcam + real-time scoring + conservative adaptation)

The domain shift track is documented in dedicated planning and report files. Live scoring artifacts (per labeled webcam run) and offline safety gate evaluations (on the eval-only manifest, 110,333 rows) are stored as JSON artifacts.

#### 3.7.1 Domain shift improvement via Self-Learning + Negative Learning (NegL)

This extension targets **webcam domain shift** by adding a safe adaptation loop. The design principle is: only accept a target-domain update if it passes an offline regression gate.

Implemented components already evidenced in this project:

- **Webcam measurement protocol** (raw vs smoothed metrics, jitter) as reported in Section 4.6.
- **Offline safety gate** on the eval-only manifest, as reported in Section 4.7.

Planned (but not yet claimed as successful results here):

- Add self-learning using high-confidence pseudo-labels plus NegL for medium-confidence samples, with strict defaults and rollback.

Executed evidence update (Feb 2026):

- A first conservative Self-Learning + manifest-driven NegL A/B attempt was executed on 2026-02-21; it passed the offline eval-only gate within rounding when preprocessing and BatchNorm behavior were controlled, but regressed on the deployment-facing same-session webcam replay score (see Section 4.9.2 for artifacts and interpretation boundaries).

## 4. Results & Analysis

All results below are copied directly from the listed artifacts.

### 4.1 Dataset integrity (multi-source)

Dataset integrity summary:

- Total rows: 466,284
- Missing paths: 0
- Bad labels: 0

Verified distribution summaries:

- Full multi-source manifest split sizes: train=378,965 / val=37,862 / test=49,457.
- HQ-train manifest split sizes: train=213,144 / val=18,020 / test=27,840.
- Mixed-domain benchmark (test_all_sources) contains 48,928 rows and all 7 classes present.

### 4.2 Teacher performance (Stage A, img224)

These teacher metrics are computed on the **Stage-A validation split** (n = 18,165) after source filtering (AffectNet balanced + FERPlus + RAF-DB basic). They are **not** the same as performance on the hard/mixed-domain gates reported in Section 4.2.2.

#### 4.2.1 Stage-A validation (in-distribution; filtered)

**Figure 4.2-1: Teacher per-class F1 comparison (Stage-A validation)**

![Teacher Per-class F1: RN18 vs B3 vs CNXT](figures/fig2_teacher_perclass_f1.png)

*Figure 4.2-1.* Per-class F1 scores for the three teacher backbones on the Stage-A validation split (n = 18,165). Disgust is the weakest class across all teachers, while Happy consistently achieves the highest F1.

**Table 4.2-1: Teacher metrics (Stage-A validation split, n = 18,165).**

| Model | Eval split (n) | Accuracy | Macro-F1 | Raw NLL | TS NLL | Raw ECE | TS ECE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RN18 | Stage-A val (18165) | 0.786182 | 0.780828 | 4.025883 | 0.880346 | 0.205298 | 0.148851 |
| B3 | Stage-A val (18165) | 0.796091 | 0.790988 | 3.221890 | 0.787123 | 0.198786 | 0.083927 |
| CNXT | Stage-A val (18165) | 0.794055 | 0.788959 | 3.101407 | 0.769976 | 0.200896 | 0.081701 |

**Table 4.2-2: Teacher per-class F1 (Stage-A validation).**

| Model | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RN18 | 0.7357 | 0.6940 | 0.7635 | 0.8970 | 0.7415 | 0.8186 | 0.8155 |
| B3 | 0.7521 | 0.7156 | 0.7576 | 0.9197 | 0.7479 | 0.8042 | 0.8399 |
| CNXT | 0.7687 | 0.7194 | 0.7395 | 0.9135 | 0.7313 | 0.8064 | 0.8439 |

#### 4.2.2 Hard gates (domain shift / mixed-domain)

For clarity: the Stage-A validation numbers above are **not** the same evaluation as the hard-gate tests.

Here we evaluate the **same three teacher checkpoints** on three hard/mixed-domain gate datasets to assess robustness under domain shift.

**Table 4.2-3: Teacher macro-F1 on hard-gate datasets.**

| Gate dataset | n (eval rows) | RN18 acc | RN18 macro-F1 | B3 acc | B3 macro-F1 | CNXT acc | CNXT macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eval_only | 11890 | 0.427250 | 0.372670 | 0.470816 | 0.392831 | 0.440875 | 0.388980 |
| expw_full | 9179 | 0.498747 | 0.374009 | 0.584486 | 0.406649 | 0.511167 | 0.382112 |
| test_all_sources | 48928 | 0.644723 | 0.617067 | 0.674297 | 0.645421 | 0.664936 | 0.638065 |

Interpretation (why this “drop” is a key negative result):

- The Stage-A validation metrics substantially overestimate performance under mixed-domain stress tests. For example, the best teacher (B3) drops from 0.791 macro-F1 on Stage-A validation to 0.393 on eval-only — a **50% relative decrease**. This gap is expected because Stage-A uses a filtered in-distribution mixture, while `eval_only` / `expw_full` / `test_all_sources` intentionally include harder domain shift + label noise.
- Practical implication: **teacher selection by Stage-A validation alone is insufficient** for deployment-facing robustness; hard-gate evaluation must be treated as a separate requirement.
- Working hypotheses for the hard-gate gap (to be tested, not assumed):
  - **Domain mismatch:** ExpW/webcam-like imagery differs from curated training sources in lighting, pose, expression intensity, and crop statistics.
  - **Label noise and class imbalance:** minority classes (Fear/Disgust) are disproportionately affected, lowering macro-F1 more than accuracy.
  - **Preprocessing alignment:** differences in face alignment/cropping/CLAHE policy can shift the evaluation distribution even when the backbone is unchanged.
- Evidence pointers for deeper analysis are tracked in the project's research logs.

### 4.3 Ensemble robustness benchmark (mixed-source)

The ensemble benchmark evaluates a weighted teacher combination on the full mixed-source test set (48,928 rows).

Selected ensemble configuration and results:

- Weights: RN18/B3/CNXT = 0.4/0.4/0.2
- Accuracy: 0.687255
- Macro-F1: 0.6596075

Additional metrics (same ensemble):

- NLL: 4.077156
- ECE: 0.287694
- Brier: 0.590869

**Table 4.3-1: Ensemble per-class F1 (mixed-source test, n = 48,928).**

| Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.6304 | 0.6034 | 0.5350 | 0.8389 | 0.5924 | 0.7205 | 0.6967 |

### 4.4 Student performance (HQ-train evaluation)

**Figure 4.4-0: Student CE training curves (MobileNetV3-Large, 10 epochs)**

![Student CE Training Curves](figures/fig5_training_curves_ce.png)

*Figure 4.4-0.* Training loss, validation accuracy, and validation macro-F1 over 10 epochs for the CE student. Loss decreases steadily; validation accuracy and macro-F1 plateau around epoch 7–8 at ~0.75 and ~0.74 respectively.

Student metrics from the Dec 2025 CE/KD/DKD runs:

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

**Figure 4.4-2: Student calibration comparison (Raw vs Temperature-Scaled)**

![Calibration: Raw ECE/NLL vs TS ECE/NLL](figures/fig3_calibration_comparison.png)

*Figure 4.4-2.* Calibration comparison showing Raw vs Temperature-Scaled ECE and NLL for the three student checkpoints. KD and DKD achieve substantially lower TS ECE (0.028 and 0.027 respectively) compared to CE (0.050), demonstrating the calibration benefit of distillation.

Interpretation:

- CE provides the best raw macro-F1 in this run.
- KD/DKD improve temperature-scaled calibration (TS ECE) substantially, but do not improve macro-F1 under this configuration.

Why this “no macro-F1 gain” is plausible (hypotheses):

- KD/DKD optimize a different objective (matching teacher distributions and often improving calibration after temperature scaling) that does not guarantee higher macro-F1 under noisy/mixed-domain evaluation.
- Distillation can shift probability **margins** and class confusions in ways that benefit NLL/ECE but do not translate into better decision boundaries for minority classes.
- Student capacity and hyperparameters (e.g., KD/DKD temperatures/weights, schedule, augmentation) can dominate whether KD/DKD improves macro-F1; the results show a clear calibration benefit here, but not a macro-F1 benefit.

#### 4.4.1 Overall sanity table (CE vs KD vs DKD across hard gates)

To provide a single consolidated view of offline performance, a consolidated table is generated across four stress-test datasets:

- Classification eval-only manifest (mixed domain)
- ExpW full manifest (in-the-wild)
- Mixed-source test (all sources combined)
- FER2013 folder split (non-official; stress-test only)

Note: **official FER2013** PublicTest/PrivateTest evaluation (from the original CSV) is reported separately in Section 4.12 because it uses a different protocol from the folder split.

**Figure 4.4-3: Student macro-F1 across hard gates (domain shift stress tests)**

![Cross-dataset Macro-F1: CE vs KD vs DKD](figures/fig6_crossdataset_macro_f1.png)

*Figure 4.4-3.* Student macro-F1 on four hard-gate datasets. CE consistently achieves the best macro-F1 across all mixed-source gates; the gap between HQ-train validation (~0.74) and these stress tests (~0.46–0.54) quantifies the domain shift challenge.

Summary:

- On the mixed-source gates (`eval_only`, `expw_full`, `test_all_sources`), CE has the best raw macro-F1 among the three checkpoints in this snapshot.
- On `fer2013_folder`, KD has the best raw macro-F1, while DKD has the best TS ECE.
- The consistent fragility remains minority classes (Fear/Disgust), supporting the domain-shift + label-noise framing rather than a conclusion that the model is fundamentally weak.

### 4.5 NL/NegL screening results (Jan 2026, offline)

The full experimental matrix and analysis are documented in an internal NL/NegL report. This section expands the key ideas and provides a larger, fully verified snapshot from the repo’s compare artifacts.

Interpretation note (protocol):

- All tables below are compiled from per-run reliability metrics (which also embed the exact run directories used).
- These are short-budget screening runs (KD 5 epochs; DKD resume runs) intended to detect regressions and characterize gating behavior, not to claim a final “best” configuration.

#### 4.5.1 KD-stage (5 epochs) comparisons

**Baseline KD vs NegL** (high entropy threshold; selective gate):

| Run | NegL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep) | off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 |
| KD+NegL (5ep) | entropy gate (w=0.05, ratio=0.5, ent=0.7) | 0.722364 | 0.719800 | 0.039770 | 0.808534 | 0.682749 |

NegL “bite” example (lower entropy threshold; higher activation but TS calibration worsens here):

| Run | NegL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep) | off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 |
| KD+NegL (5ep) | entropy gate (w=0.05, ratio=0.5, ent=0.3) | 0.728177 | 0.726967 | 0.046010 | 0.827339 | 0.698288 |

NL(proto) example (stable, but no clear improvement at these settings):

| Run | NL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep) | off | 0.728363 | 0.726648 | 0.027051 | 0.783856 | 0.697342 |
| KD+NL(proto) (5ep) | proto (dim=32, m=0.9, thr=0.2, w=0.1) | 0.729573 | 0.728076 | 0.042676 | 0.796150 | 0.694379 |

#### 4.5.2 DKD-stage (resume) comparisons

Baseline DKD vs NegL (entropy gate ent=0.7):

| Run | NegL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DKD baseline | off | 0.735711 | 0.736796 | 0.034764 | 0.783468 | 0.704458 |
| DKD+NegL | entropy gate (w=0.05, ratio=0.5, ent=0.7) | 0.735060 | 0.734752 | 0.034830 | 0.792553 | 0.702431 |

NL-only DKD example (regression at the tested weight):

| Run | NL | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| DKD baseline | off | 0.735711 | 0.736796 | 0.034764 | 0.783468 | 0.704458 |
| DKD+NL(proto) | proto (top-k=0.05, w=0.1) | 0.719807 | 0.717861 | 0.045183 | 0.844715 | 0.688264 |

Synergy example (DKD + NL + NegL; raw calibration improves but F1 does not):

| Run | NL + NegL | Raw acc | Raw macro-F1 | Raw ECE | Raw NLL | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DKD baseline | off | 0.735711 | 0.736796 | 0.211901 | 1.475317 | 0.034764 | 0.783468 | 0.704458 |
| DKD+NL+NegL | NL(proto) + NegL(ent=0.4) | 0.733712 | 0.733798 | 0.202779 | 1.412536 | 0.037443 | 0.786831 | 0.701544 |

#### 4.5.3 Mechanism sanity signals and failure modes

NegL gating example (why ent=0.7 can be too selective): NegL `applied_frac` starts at 0.031961 (epoch 0) and drops to below 1% by epoch 1 (0.00970), reaching 0.00583 by epoch 4.

Legacy instability counterexample (learned NegL gate interaction): raw accuracy drops to 0.5402 with macro-F1 0.5204 (notably Angry F1 = 0.1961 and Neutral F1 = 0.3563), indicating an unstable configuration under that variant.

Summary conclusion (evidence-backed across the existing compare tables):

- Under the tested short-budget configurations, NL/NegL does not provide a consistent macro-F1 or minority-F1 improvement over KD/DKD baselines; outcomes are sensitive to gating/weighting and can regress TS calibration.
- **Why this happens (Hypothesis):** As detailed in the NL/NegL screening reports, datasets like ExpW are strongly imbalanced. Applying auxiliary negative signals can be risky because they may disproportionately penalize and destabilize already-weak minority decision regions. Furthermore, teacher softlabels in KD/DKD already encode strong structural information; adding NegL can conflict with these teacher targets, especially when the entropy gate selects uncertain regions that overlap with minority class confusions.

#### 4.5.4 Feb 2026 addendum: NL/NegL status (what the offline evidence supports)

Building on the baseline CE → KD → DKD pipeline (ensemble-teacher softlabels), we conducted controlled short-budget screening runs to test whether Negative Learning (NL) and Complementary/Negative Learning variants (NegL) can improve offline accuracy, macro-F1, minority-F1 (lowest-3), and calibration (ECE/NLL, including temperature-scaled variants).

Across KD and DKD stages, the stored comparison tables show:

- **NL(proto)** was generally stable but did not consistently outperform the baseline under the tested configurations.
- **NegL with entropy gating** showed mixed effects: high entropy thresholds can apply too sparsely to strongly influence learning, while lower thresholds increase activation but can worsen temperature-scaled calibration in these runs.
- **Synergy (NL + NegL)** can improve raw loss/calibration signals under DKD in some settings, but did not translate into clear gains in macro-F1 or minority-F1 in the recorded comparisons.

Overall , NL/NegL as currently tuned are **not ready as drop-in improvements** for the existing KD/DKD pipeline.

Future work should focus on safer weighting and more reliable gating (including reporting/monitoring the NegL activation rate), and adding deployment-facing stability metrics (e.g., flip-rate/jitter) to these experiments.

### 4.6 Domain shift: live webcam scoring results (Jan 2026)

**Figure 4.6-1: Webcam demo — raw vs smoothed per-class F1**

![Webcam Raw vs Smoothed Per-class F1](figures/fig8_webcam_raw_vs_smoothed.png)

*Figure 4.6-1.* Per-class F1 for raw vs EMA-smoothed predictions during the webcam demo session (20260126_205446, n=4,154). Smoothing improves most classes but Fear remains at 0.00 F1 (393 samples), revealing a critical deployment gap..

**Figure 4.6-2: Webcam confusion matrix (smoothed predictions)**

![Confusion Matrix — Webcam Demo](figures/fig4_confusion_matrix_webcam.png)

*Figure 4.6-2.* Confusion matrix from the webcam demo session (smoothed predictions, n=4,154). Fear samples are overwhelmingly misclassified as Disgust (304/393); Sad samples are almost entirely predicted as Disgust (479/542). This reveals the structural confusion between minority/negative-valence classes under webcam domain shift..


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

#### 4.6.1 Feb 2026 addendum: qualitative real-time checkpoint preference (objective mismatch)

Deployment-facing qualitative observation (not a controlled, artifact-backed comparison):

- During informal interactive webcam use, the **CE** student checkpoint (Dec 2025 main run) appeared more stable (fewer visibly frequent class switches) than the tested KD+LP and DKD checkpoints under the same demo pipeline settings.

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
4) Store run artifacts and compare scoring results (raw vs smoothed metrics + jitter flips/min).


### 4.7 Domain shift: conservative adaptation and offline safety gate


Table 4.7-1 (Offline safety gate). Split: test. Protocol: single-crop.

| Model | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline (CE20251223) | 0.567368 | 0.485878 | 0.059181 | 1.228754 | baseline |
| Head-only FT | 0.548024 | 0.450845 | 0.060011 | 1.289419 | fail (macro-F1 drop) |
| BN-only FT | 0.548612 | 0.451277 | 0.060566 | 1.289044 | fail (macro-F1 drop) |

Interpretation:

- Both head-only and BN-only fine-tuning (as configured in these **early Jan-2026 FT runs**) reduce offline macro-F1 on a broader eval-only distribution; these checkpoints should not be promoted beyond experiments.
- **Why this happens (Hypothesis):** While BN-only tuning is often safer, it can still shift feature statistics enough to hurt generalization if the adaptation buffer is narrow, if the pseudo-label signal is noisy, or if BatchNorm running statistics drift (which can occur even in “head-only” intent if BN layers remain in `train()` mode).
- Note: Section 4.9.2 documents a later 2026-02-21 adaptation attempt where preprocessing (`use_clahe`) and BatchNorm running-stat updates are controlled; that later candidate can pass the offline gate within rounding but still fails the deployment-facing webcam A/B.

### 4.8 Domain shift: ExpW cross-dataset evaluation (Jan 2026)

This subsection reports a controlled **cross-dataset** evaluation on ExpW (static images). It complements the webcam live-scoring evidence in Section 4.6 by providing a repeatable, manifest-defined “in-the-wild” test.


Table 4.8-1 (ExpW cross-dataset gate). Split: test. Protocol: single-crop.

| Label | Mode | Epochs | NegL | NL | Raw acc | Raw macro-F1 | Raw ECE | Raw NLL | TS ECE | TS NLL | Minority-F1 (lowest-3) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| DKD_20251229_223722 | dkd | - | off | off | 0.622944 | 0.459529 | 0.254957 | 1.820043 | 0.036783 | 1.102717 | 0.259545 |
| DKD_20260101_204953 | dkd | - | off | off | 0.579039 | 0.431158 | 0.236954 | 1.633517 | 0.025713 | 1.194914 | 0.249118 |
| DKD_20260101_212203 | dkd | - | off | off | 0.616843 | 0.451807 | 0.248753 | 1.733742 | 0.036499 | 1.116522 | 0.250905 |
| DKD_20260101_214949 | dkd | - | off | off | 0.615644 | 0.450946 | 0.246569 | 1.723473 | 0.032534 | 1.119614 | 0.251363 |
| DKD_20260101_221602 | dkd | - | off | off | 0.617714 | 0.452596 | 0.241540 | 1.721193 | 0.041437 | 1.120937 | 0.252719 |

Interpretation:

- On this ExpW test, DKD_20251229_223722 is the strongest among the listed checkpoints by raw macro-F1 (0.460).
- Temperature scaling produces low TS ECE across these ExpW evaluations (≈0.026–0.041), while raw ECE remains high (≈0.237–0.255), reinforcing the “calibration benefit” pattern observed elsewhere in this project.

### 4.9 Detailed extension summary (Jan 2026)

#### 4.9.1 NL / NegL / complementary learning (offline screening)

Hypothesis:

- Adding NegL (and/or NL(proto)) on top of KD/DKD can improve calibration and potentially improve minority performance by preventing confident mistakes.

What the artifacts show (Sections 4.5.1–4.5.3):

- Under the tested short-budget configurations, **macro-F1 improvements are not consistent**, and some configurations regress.
- NegL gating has a measurable “strength” via `applied_frac` in `history.json`. When the gate is too selective, NegL likely has limited effect; when too strong, it can destabilize training.
- A configuration can improve **raw calibration metrics** (raw ECE / raw NLL) without improving macro-F1 (the DKD+NL+NegL synergy compare in Section 4.5.2 is the clearest example currently in-repo).

Practical takeaway:

- Treat NL/NegL as a sensitive regularizer: it requires careful tuning and stronger safety checks than plain KD/DKD.

#### 4.9.2 Domain shift improvement via Self-Learning + NegL (webcam loop)

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

- A reproducible webcam scoring protocol (raw vs smoothed, plus jitter flips/min) with per-session scoring artifacts.
- An offline regression gate on the eval-only manifest; in the current experiments, conservative fine-tuning candidates failed this gate and were rejected (Section 4.7).

2026-02-21 update (this report version): issues found, fixes applied, and A/B outcome

Observed issues:

- **Preprocessing mismatch can cause false “regressions”:** the baseline checkpoint used `use_clahe=True` while early adaptation checkpoints were trained with `use_clahe=False`. Since evaluation defaults to the checkpoint’s stored settings, the gate check was not initially apples-to-apples.
- **BatchNorm running-stat drift under small-buffer tuning:** even when tuning “head-only”, calling `model.train()` updates BatchNorm running mean/variance, which can cause large distribution shifts from a tiny webcam buffer.

Fixes implemented (repo changes):

- Added a conservative training safeguard: when `--tune` is not `all`, BatchNorm layers are forced to `eval()` during training to freeze running stats.
- Added a replay inference utility to enable fair webcam A/B scoring on the *same* recorded session while preserving manual labels/time.

Method detail (what “Self-Learning + NegL” means in this A/B):

- Buffer source: the self-learning buffer was built from the recorded session's per-frame log by selecting stable frames and using the model's predicted label as the pseudo-label (not the manual label).
- Confidence-banded supervision in the manifest:
  - High-confidence frames ($p_{max} \ge \tau_{high}$) become pseudo-labeled positives with `weight=1`.
  - Medium-confidence frames ($\tau_{mid} \le p_{max} < \tau_{high}$) become **NegL-only** samples with `weight=0` and `neg_label=<predicted_label>` (i.e., discourage probability mass on the model’s own uncertain prediction).
  - Low-confidence frames are excluded.
- Training consumption: the adapted run consumes `weight` as weighted CE and consumes `neg_label` as the explicit NegL target. This is intentionally conservative (no positive CE update on medium-confidence frames) but can still be harmful if the buffer is narrow or the negative target policy is mis-specified.

Gate check result (eval-only manifest):

- Baseline checkpoint: CE student (Dec 2025 main run)
- Adapted checkpoint: head-only + CLAHE + BN-stats frozen (Feb 2026)

On the eval-only manifest (test split, single-crop), the adapted checkpoint matches the baseline within rounding in macro-F1, indicating the offline safety gate can be passed when preprocessing and BN behavior are controlled.

Webcam A/B result (same recorded session, same manual labels):

**Webcam A/B comparison** (same recorded session, same manual labels):

Summary (smoothed predictions):

- Baseline: accuracy 0.5879, macro-F1 0.5248, minority-F1(lowest-3) 0.1609, jitter 14.86 flips/min
- Adapted: accuracy 0.5269, macro-F1 0.4667, minority-F1(lowest-3) 0.1384, jitter 14.16 flips/min

**Figure 4.9-1: Domain shift adaptation A/B — baseline vs adapted (webcam replay)**

![Adaptation A/B Comparison](figures/fig10_adaptation_ab.png)

*Figure 4.9-1.* A/B comparison on the same labeled webcam session. The adapted checkpoint (Self-Learning + NegL) slightly reduces jitter (−0.7 flips/min) but regresses accuracy (−0.061) and macro-F1 (−0.058). This is the key negative result NR-1: offline gate pass does not imply deployment improvement..

Interpretation:

- The gate-passing adaptation is **not yet beneficial** for the labeled webcam session: it slightly reduces jitter but regresses accuracy and macro-F1.
- **Why this happens (Hypothesis):** Several mechanisms can produce a webcam A/B regression even when an offline gate is passed:
  - **Small, correlated buffer:** a single-session buffer can be narrow (subject-specific lighting/pose), amplifying overfitting and catastrophic forgetting.
  - **Pseudo-label noise + transition frames:** “stable frame” heuristics can still include ambiguous transitions; a small number of wrong pseudo-labels can move decision boundaries.
  - **NegL target policy risk:** in this implementation, medium-confidence samples apply NegL with `neg_label=<predicted_label>` (discourage mass on the model’s own uncertain prediction). If that predicted label is often correct-but-uncertain, this rule can push probability away from the right class and reduce macro-F1.
  - **Objective mismatch:** eval-only / ExpW gates are broad distribution checks; webcam replay is a *specific deployment objective* with temporal stabilization. A model can pass offline non-regression while still degrading the probability margin patterns that matter for webcam scoring.
- This supports the “safety-gated adaptation” framing: passing an offline gate is necessary, but not sufficient, to claim an improvement on deployment-facing metrics.
- Scope limitation (academic interpretation): this is evidence from a single recorded session and a single conservative adaptation candidate. It demonstrates a failure mode (gate pass does not imply webcam improvement), but it does not prove that self-learning + NegL cannot help under other buffer policies, thresholds, multi-session data, or alternative negative-target definitions.

Immediate next steps (to turn this negative result into a controlled research loop):



1. Run one-knob ablations that isolate the cause of the regression:
    - self-learning positives only (disable NegL) vs NegL-only vs combined.
    - vary buffer thresholds ($\tau_{high}$, $\tau_{mid}$) and the ratio of positives vs NegL-only.
2. Audit the buffer:
    - report class distribution, confidence histogram, and an estimated pseudo-label error rate against available manual labels (for analysis only; still train on pseudo-labels).
3. Revisit the NegL definition for medium-confidence frames:
    - compare `neg_label=<predicted_label>` vs a complementary-class policy (select a plausible wrong class) vs “ignore medium confidence”.
4. Promote candidates only if they satisfy both:
    - non-regression on eval-only/ExpW, and
    - improvement on the replayed labeled webcam session under fixed demo parameters.

How ExpW fits this extension:

- ExpW cross-dataset evaluation (Section 4.8) provides an additional controlled in-the-wild test bed to evaluate whether domain-shift adaptations generalize beyond a single webcam session.

### 4.10 Consolidated Negative Results (Single-Report Submission Version)

This section consolidates the key negative results into a single reference for submission.

#### 4.10.1 Negative result matrix (evidence-backed)

| ID | Negative result statement | Status | Main evidence |
| --- | --- | --- | --- |
| NR-1 | 2026-02-21 Self-Learning + manifest-driven NegL candidate passed offline eval-only non-regression (within rounding) but regressed on identical-session webcam replay | Evidence-backed | Section 4.9, per-session scoring artifacts |
| NR-2 | Early head-only and BN-only adaptation candidates failed the offline eval-only gate (macro-F1 drop) | Evidence-backed | Eval-only gate metrics (Appendix A.1) |
| NR-3 | NL/NegL short-budget KD/DKD screening did not produce consistent macro-F1 or minority-F1 gains | Evidence-backed | Per-run training history and reliability metrics (Section 4.5) |
| NR-4 | KD/DKD improved calibration (TS ECE/TS NLL) but did not outperform CE macro-F1 in the Dec-2025 main snapshot | Evidence-backed | Artifacts listed in Section 4.4 |
| NR-5 | Strong Stage-A teacher validation did not transfer to hard-gate robustness (eval-only, ExpW) | Evidence-backed | Teacher benchmark summary and hard-gate analysis (Section 4.2) |
| NR-6 | Hard-gate weakness concentrates in minority/confusable classes (especially Fear/Disgust); calibration correction alone is insufficient | Evidence-backed | Per-class F1 analysis (Section 4.8, Section 9.3.3) |
| NR-7 | LP-loss short-budget screening improved some calibration terms but did not show a clear ExpW macro-F1 gain | Evidence-backed | Section 9.3.2 gate tables |

#### 4.10.2 Detailed negative result analysis

#### NR-1: Offline gate pass, webcam replay regression (critical)

The most consequential negative result involves the 2026-02-21 Self-Learning + manifest-driven NegL adaptation candidate. The adaptation loop was designed to improve target-domain webcam behavior while avoiding broad-distribution regression: a session-specific self-learning buffer was built, conservative fine-tuning with confidence-banded supervision was applied, and promotion was conditioned on offline non-regression and deployment-facing replay improvement.

Protocol controls ensured fairness: the same recorded session and same manual labels were used, preprocessing was harmonised (`use_clahe` mismatch corrected), and BatchNorm running-stat updates were prevented during non-`all` tuning.

**Outcome (evidence-backed).** Smoothed metrics on the same scored frames ($n = 4{,}154$):

| | Accuracy | Macro-F1 | Minority-F1 (lowest 3) | Jitter (flips/min) |
|---|---|---|---|---|
| Baseline | 0.5879 | 0.5248 | 0.1609 | 14.86 |
| Adapted | 0.5269 | 0.4667 | 0.1384 | 14.16 |

The adapted checkpoint slightly reduced jitter but materially regressed deployment-facing accuracy and macro-F1. Therefore, adaptation is a fail for this session under the pre-defined webcam objective.

**Evidence-backed explanation:** Offline non-regression and deployment replay are different objectives; satisfying one does not guarantee the other.

**Hypotheses requiring direct ablation:** (1) narrow, correlated session buffer induced localised overfit; (2) transition-frame pseudo-label noise shifted class boundaries; (3) medium-confidence NegL policy ($\text{neg\_label} = \langle\text{predicted\_label}\rangle$) may suppress correct-but-uncertain classes; (4) probability-margin changes degraded EMA/hysteresis behaviour without large offline static-loss signals.

#### NR-2: Early head-only and BN-only adaptation failed eval-only safety gate

Against baseline (raw macro-F1 = 0.4859 on eval-only):

- Head-only FT: 0.4508 macro-F1 (fail).
- BN-only FT: 0.4513 macro-F1 (fail).

This negative result justifies the project's strict "gate-first, promote-later" discipline and demonstrates that adaptation can regress broad-distribution macro-F1 even under seemingly conservative update scopes.

#### NR-3: NL/NegL screening did not provide consistent offline gains

Across KD/DKD screening comparisons, no robust, repeatable macro-F1/minority-F1 lift over baseline was observed. Several settings regressed. Mechanism signals from `history.json` indicate: high-threshold NegL can become too sparse (`applied_frac` very low); threshold-based NL can decay toward inactivity; top-$k$ NL keeps activity but can over-regularise at tested weights. NL/NegL cannot be claimed as drop-in improvements in current tuning regimes; they remain conditional research components.

#### NR-4: KD/DKD improved calibration but not CE macro-F1

In the Dec-2025 HQ-train evaluation snapshot, CE achieved the best raw macro-F1 while KD/DKD achieved better temperature-scaled calibration metrics. This constitutes a negative result under a macro-F1-improvement objective and indicates optimisation target mismatch between calibration and decision-boundary quality.

#### NR-5: Teacher Stage-A validation did not predict hard-gate robustness

Teachers with strong Stage-A validation (~0.78–0.79 macro-F1) dropped substantially on `classification_manifest_eval_only` and `expw_full_manifest`, while performing better on `test_all_sources`. This negative finding demonstrates that in-distribution validation is insufficient as a deployment proxy; evaluation distribution mismatch drives the gap.

#### NR-6: Hard-gate failures concentrate in Fear/Disgust

From the low-performance investigation: ExpW/eval-only class distribution and confusion profile consistently penalise Fear/Disgust; FER2013 uniform-7 is balanced yet Fear remains weak, so imbalance alone is an insufficient explanation; temperature scaling improved ECE/NLL but did not change macro-F1 ranking. Class-wise representational mismatch under domain shift, not only confidence calibration, is the dominant issue.

#### NR-7: LP-loss did not show clear ExpW macro-F1 gain

In the Feb-2026 short-budget KD baseline vs KD+LP study, eval-only macro-F1 changed slightly upward for KD+LP, but ExpW macro-F1 did not improve (slight decline). Some calibration terms improved. LP-loss is not validated as a domain-shift macro-F1 improver under current short-budget settings.

#### 4.10.3 Cross-cutting causal structure

The negative results above form a coherent causal structure that can be decomposed into evidence-backed statements and testable hypotheses.

**Evidence-backed causal statements:**

1. **Objective mismatch exists:** offline static gates and webcam replay can diverge.
2. **Gate necessity is validated:** without eval-only gate, regressive candidates could be mistakenly promoted.
3. **Class-specific fragility dominates:** Fear/Disgust are persistent weak links across hard gates.
4. **Calibration is not enough:** lower ECE/NLL does not guarantee macro-F1 or replay gain.

**Hypothesis map (to be tested, not asserted):**

1. **Buffer quality hypothesis:** pseudo-label noise and transition-frame contamination drive adaptation harm.
2. **NegL policy hypothesis:** medium-confidence negative-target definition may be mis-specified.
3. **Margin dynamics hypothesis:** adaptation changes top-1/top-2 margins in ways that destabilise temporal post-processing.
4. **Training-budget hypothesis:** short-run screens reveal risk but under-sample stable improvement regimes.

#### 4.10.4 Threats to validity

1. **External validity:** some conclusions rely on single-session replay or limited run counts. The strongest adaptation negative result (NR-1) is based on one recorded session and one conservative candidate; generalisation to all users/cameras cannot be claimed.
2. **Protocol variance risk:** historical runs can differ in preprocessing defaults, crop pipeline, or temperature policy.
3. **Selection bias risk:** mitigated by retaining both favourable and unfavourable outcomes in artifact-backed summaries.
4. **Underpowered screening:** several studies are intentionally short-budget for fast triage; this prioritises risk detection over final optimisation.
5. **Distribution dependence:** class support can differ substantially across runs, so raw macro-F1 differences must be read with support context.
6. **Gate objective mismatch:** static offline gates and temporal deployment objectives are related but not identical.

#### 4.10.5 Evaluation protocol for domain-shift claims

Based on the negative results above, the following evaluation protocol is adopted for any future domain-shift adaptation claim:

1. **Offline safety first:** no candidate can be promoted if eval-only macro-F1/minority-F1 violate non-regression thresholds.
2. **Replay requirement:** no deployment improvement claim without identical-session replay gain using same labels and fixed scorer settings.
3. **Causal isolation:** change one variable per experiment to preserve interpretability.
4. **Replication:** a candidate must pass at least two distinct labelled sessions before any generalised claim.
#### 4.10.6 Summary

The project's negative results are coherent, reproducible, and scientifically meaningful. They do not indicate failed research; they establish a rigorous boundary between offline metric optimisation and true deployment-facing improvement. The most important methodological conclusion is: **a model update is not successful unless it passes both broad-distribution safety gates and fixed-protocol webcam replay criteria.** This principle, supported by the 2026-02-21 adaptation counterexample (NR-1), is a central contribution of the project.

### 4.11 Additional evaluations (Feb 2026)

The following Feb-2026 evaluations extend the results above. Full details are in Section 9.3:

- **LP-loss screening (Section 9.3.2):** KD+LP slightly improved eval-only macro-F1 (0.439→0.441) but did not improve ExpW macro-F1; calibration improved on both gates.
- **Offline diagnostics (Section 9.3.3):** per-source breakdown revealed that the `expw_hq` subset drives low aggregate eval-only performance; CLAHE ablation confirmed CLAHE is not the root cause of low benchmark scores.
- **FER2013 official split (Section 9.3.4):** on the protocol-matched PublicTest, the best student (DKD) achieves 61.4% single-crop accuracy. This is below paper SOTA (73.3%) but expected given the MobileNetV3 capacity–latency trade-off and training recipe differences.
- **Paper comparison (Section 9.3.4):** RAF-DB accuracy (86.3%) is competitive; AffectNet balanced macro-F1 (0.823) is strong but uses a different evaluation protocol; cross-domain stress tests remain the weakest results.

## 5. Demo and Application

### 5.1 Real-time demo pipeline

The demo system implements a multi-stage real-time inference pipeline targeting CPU deployment on a consumer laptop (Intel i9-13900HX @ 2.20 GHz). The pipeline architecture is:

```
Webcam → YuNet Face Detection → Crop + Resize (224×224) → Optional CLAHE
  → FER Model (MobileNetV3-Large, ONNX-exported) → Temperature Scaling
  → EMA Smoothing → Hysteresis → Optional Vote Window → Display + CSV Log
```

Key implementation details (from `demo/realtime_demo.py`):

1. **Face detection:** YuNet [18] provides per-frame bounding boxes; crop jitter between frames is a source of prediction instability.
2. **Preprocessing:** input crops are resized to 224×224; optional CLAHE histogram normalisation matches training-time preprocessing.
3. **Inference:** the student checkpoint (MobileNetV3-Large) is loaded as a PyTorch model or ONNX export. ONNX export was validated for CPU deployment (Jan 2026).
4. **Temperature scaling:** logits are divided by a stored global temperature $T$ (from `calibration.json`) before softmax. This preserves argmax but changes probability sharpness, affecting downstream smoothing behaviour.
5. **Temporal stabilisation:**
   - EMA smoothing over probability vectors (parameter `ema_alpha`)
   - Hysteresis on the predicted class index (parameter `hysteresis_delta`): a new class must exceed the current class probability by at least `hysteresis_delta` to trigger a switch
   - Optional vote window (parameters `vote_window` / `vote_min_count`)
6. **Logging:** every frame is logged to a per-frame CSV with timestamp, raw/smoothed predictions, confidence, and optional manual labels for later scoring.

Device support: the demo supports CPU, CUDA, and DirectML backends for repeatable benchmarking across hardware configurations. A backup deployment package was published to a GitHub demo repository with Git LFS for model weights.

### 5.2 Deployment KPIs (current status)

What is already measured and stored:

- Live classification behavior: raw vs smoothed macro-F1/accuracy and jitter flips/min from per-session scoring artifacts.

What remains required for a complete deployment report:

- A timed benchmark run on the target device (CPU-only and GPU if applicable) to report FPS and latency distribution.

Note: as of the Dec-2024 mini-report, FPS/latency/flip-rate numbers were not yet claimed because those require a dedicated timed demo run with an attached per-frame CSV log.

### 5.3 Subjective checkpoint preference finding (deployment-facing)

During live webcam demonstrations (Feb 2026), three student checkpoints were compared subjectively:

- CE (Dec-2025 main run)
- KD + LP-loss (Feb-2026 screening)
- DKD (Jan-2026 best-of-screening candidate)

**Subjective finding:** The CE checkpoint exhibited the most stable behaviour (fewest noticeable label flickers) and the best perceived accuracy during live webcam use, compared to the KD+LP and DKD checkpoints.

This observation, while subjective, is consistent with the project's quantitative findings and can be explained by several evidence-aware mechanisms:

1. **Probability margin dynamics.** The demo applies EMA smoothing over probability vectors and hysteresis-based switch suppression. Even when two checkpoints have similar offline macro-F1, they produce different probability margins frame-to-frame, directly affecting how often hysteresis permits label switching. CE may produce larger, more stable margins on the webcam domain.

2. **Temperature scaling interaction.** The demo applies $\text{logits}/T$ before softmax using stored calibration parameters. Temperature scaling does not change argmax labels but does change probability sharpness, which in turn modifies EMA + hysteresis behaviour. KD/DKD checkpoints, optimised to match teacher distributions, may exhibit different calibrated probability geometry that interacts unfavourably with these temporal filters.

3. **KD/DKD uncertainty inheritance.** Students trained via KD/DKD are optimised to reproduce teacher logit/probability distributions. If the teacher ensemble is uncertain or biased for webcam-like frames (variable lighting, pose, motion), the student inherits smaller top-1/top-2 margins ("near-ties"), which produce visible label flicker when face detection crops shift slightly between frames.

4. **Face detection crop jitter.** Per-frame face bounding boxes from YuNet [18] can shift even slightly, introducing input perturbation. A checkpoint more sensitive to crop/illumination changes exhibits greater probability variance. CE may have learned decision boundaries less sensitive to these small perturbations on the user's camera domain.

5. **Domain shift preference reversal.** Offline tables (HQ-train, eval-only, ExpW) represent different distributions from the user's live webcam stream. It is plausible for CE to be strongest on the webcam domain even if KD/DKD show better calibration on offline splits.

**Validation protocol (recommended).** To convert this subjective observation into an artifact-backed claim, a controlled labelled webcam session should be recorded and replayed with each checkpoint under identical demo parameters (`ema_alpha`, `hysteresis_delta`, vote settings, CLAHE, temperature), producing the standard `per_frame.csv` and `score_results.json` artifacts for traceable comparison.

## 6. Discussion and Limitations

This project demonstrates that a lightweight student model (MobileNetV3-Large) can achieve competitive in-domain performance (0.742 macro-F1; 86.3% RAF-DB accuracy) while operating within real-time CPU constraints. However, three systematic tensions emerge from the results that have broader implications for deployed FER systems.

First, **distillation improves calibration but not decision boundaries**: KD/DKD consistently lower temperature-scaled ECE (0.050 to 0.027) without improving macro-F1, confirming that teacher-distribution matching optimises a fundamentally different objective than hard-label classification. Second, **offline evaluation is necessary but insufficient for deployment claims**: the NR-1 adaptation counterexample demonstrates that a candidate can pass all offline safety gates yet regress on the same webcam session it was adapted from. Third, **minority-class fragility is the dominant failure mode under domain shift**: Fear and Disgust degrade to near-zero F1 on cross-domain gates, and this fragility persists across backbones, distillation methods, and preprocessing variants (including CLAHE ablation), suggesting a representational rather than preprocessing cause.

These findings motivate the dual-gate evaluation framework proposed in this work: both broad-distribution offline non-regression and deployment-facing replay improvement must be satisfied before any checkpoint is promoted. The remainder of this section situates these results within the broader FER literature and clarifies the limitations of the current evaluation protocol.



### 6.0 Discussion refinement (what results mean, and what they do not)

- **Different evaluation regimes answer different questions:**
  - Training-time validation (teacher Stage-A val; student HQ-train val) measures *in-distribution* model selection.
  - Offline gates (eval-only / ExpW / mixed-source) measure *deployment-aligned stress* under domain shift and label noise.
  - Official-split tests (e.g., FER2013 PublicTest/PrivateTest from `fer2013.csv`) support *paper-style* comparison.
- **Therefore, numbers are not interchangeable:** a high Stage-A teacher val macro-F1 does not imply high mixed-source macro-F1, and vice versa.

#### 6.0.1 Clear limitations (protocol + data + deployment)

- **Protocol mismatch is the #1 paper-compare risk:** single-crop vs ten-crop, preprocessing, and label mapping can move accuracy materially. This is why we label comparisons as `Comparable: Yes/Partial/No`.
- **Dataset packaging vs official splits:** folder-packaged datasets can differ from papers’ official split definitions. This report treats official-split evaluation (when available) as the anchor, and uses folder/uniform-7 as stress tests.
- **Domain shift dominates deployment:** webcam lighting, motion blur, and crop jitter can cause failures that are invisible on curated offline tests.
- **Temporal stabilization changes the measured objective:** raw vs smoothed metrics are both needed; smoothing can improve perceived stability while hiding instantaneous errors.

#### 6.0.2 Adaptation risks (why safety gates exist)

- **Small-buffer fine-tuning is risky:** it can improve the target session while harming broad generalization.
- **NL/NegL is sensitive:** gating/weighting choices can produce calibration changes without macro-F1 gains.
- **KD/DKD can reduce probability margins under shift:** increased near-ties can amplify flicker under EMA/hysteresis.

Practical safeguard used in this project:

- Adaptation candidates must pass offline regression gates (eval-only / ExpW) before any “improved” claim.

### 6.1 Analytical comparison vs papers (trade-off analysis, not “winning”)

Supervisor clarification: in this FYP, “comparison” is primarily an **analytical comparison**.

Goal:

- Not to compete with SOTA papers on a single benchmark number.
- Instead, to explain **where and why** performance differs, whether the gap is reasonable under our constraints, and what engineering trade-offs were chosen to meet real-time deployment goals.

#### 6.1.1 Our system objective vs typical paper objective

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

#### 6.1.2 Why our numbers should not be compared 1:1 to paper SOTA

Direct numeric comparison is only fair when the following are matched:

- Dataset split/protocol (official train/test, crop policy, etc.)
- Label mapping (7-class vs 8-class vs compound)
- Metric definition (accuracy vs macro-F1, averaging, test-time augmentation)

FER2013 example (common mismatch): many papers report **ten-crop** accuracy, while many baselines (including earlier versions of this project) report **single-crop** evaluation. In this report, we explicitly separate and report **both** single-crop and ten-crop results for the official FER2013 PublicTest/PrivateTest splits, and still avoid claiming strict numeric equivalence unless preprocessing and training protocol are also matched.

In this project, we keep comparisons evidence-backed and protocol-aware:


#### 6.1.3 Where performance differs and why that is expected

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

#### 6.1.4 What we optimize that papers often do not

This repo measures and engineers for deployment-facing behaviors:

- **Calibration** via temperature scaling (ECE/NLL) and artifact logging per run.
- **Stability** via smoothing + hysteresis (reported separately from raw classifier metrics).
- **Reproducibility** via manifests, validation, and stored metrics artifacts.

Concrete “overall sanity” snapshot (CE vs KD vs DKD on four hard gates) is provided as a single artifact-backed table:

- The overall CE vs KD vs DKD sanity snapshot (four hard gates) is tabulated in Section 4.4.

Interpretation rule:

- Use SOTA papers as a **reference point** to explain gaps and constraints.
- Only claim a “paper-comparable” number when the protocol is truly matched.

### 6.2 FYP requirements checklist (evidence audit)

The following checklist maps common FYP requirements (as discussed with the supervisor) to concrete evidence already stored in this project.

| Requirement | Status | Evidence in repo |
| --- | --- | --- |
| 1) Study deep learning methods for FER | Met | Paper study notes, background and method discussion in this report and interim reports. |
| 2) Study attention mechanisms + apply attention / knowledge distillation | Met (KD/DKD) / Met (attention as used in backbones) | KD/DKD implemented and evaluated with per-run reliability metrics. Attention mechanisms are included in used architectures (e.g., SE-style attention in MobileNetV3 / EfficientNet) and discussed in interim reports. |
| 3) Identify at least two datasets for experimentation | Met | Multi-source manifests covering FERPlus, AffectNet, RAF-DB, ExpW, and FER2013 (official splits). Six validated CSV manifests with integrity checks. |
| 4) Investigate and evaluate at least three methods | Met | Student: CE vs KD vs DKD comparisons with artifact-backed summaries (Sections 4.4, 9.3.4). Teacher backbones also provide additional method variety (RN18/B3/CNXT). |
| 5) Explore techniques to enhance FER performance (optional) | Met | Domain-shift loop + NL/NegL screening documented in Sections 4.5, 4.9, and Feb-2026 addendum (Section 9.3). |
| 6) Present at least one paper (seminar / paper study presentation) | Met | Presentation delivered (Dec 2025 interim presentation and Jan 2026 paper study presentation). |

## 7. Project Timeline (updated)

Jan 2026 (completed):

- Domain shift measurement loop (record → score → buffer → fine-tune → offline gate).
- NL/NegL offline screening runs and documentation.
- Demo engineering for repeatable device benchmarking and backup packaging.
- ONNX export validation and GitHub demo repo publication with Git LFS.

Feb 2026 (completed):

- Implemented safe self-learning + NegL ablation in the domain-shift pipeline with strict default-off gating; first A/B executed on 2026-02-21 (did not improve the labeled webcam replay score — documented as NR-1).
- Tightened buffer construction (stable-only sampling, per-class caps, and replay anchors); diagnosed CLAHE mismatch and BN running-stat drift as root causes of early adaptation regressions.
- Paper-study-driven extensions (Week 1): implemented LP-loss as an optional supervised auxiliary loss in student training with artifact logging and post-training gate evaluation.
- Established fresh KD baseline, then ran KD+LP screening with eval-only and ExpW gates.
- Ran offline benchmark suite and challenging-benchmark diagnostics; performed per-source breakdown analysis and CLAHE ablation.
- Protocol-aware FER2013 official split evaluation (single-crop + ten-crop).
- Academic audit of final report; consolidated negative result report (NR-1–NR-7).

Mar 2026 (in progress):

- Report consolidation and final write-up (this document).
- Timed demo KPI reporting (FPS, latency, jitter/flip-rate) on the target CPU.

Apr 2026 (planned):

- Consolidated final evaluation on a controlled benchmark manifest, final presentation, and packaging.

## 8. Lessons Learned from Development

- **Artifact-grounded workflow prevents silent drift:** manifests, JSON metrics, and checkpoints make results auditable.
- **Validation gates save time:** manifest path/label validation prevents wasted training.
- **Resume semantics matter:** DKD resume must guarantee `total_epochs > start_epoch` to avoid no-op runs.
- **Deployment metrics must be explicit:** real-time stability (flip-rate/jitter) is not implied by offline macro-F1.
- **Domain adaptation needs safety rails:** without an offline gate, small-buffer adaptation can silently harm generalization.

## 9. Conclusion and Next Steps

### 9.1 Conclusion

This project delivers a reproducible, end-to-end real-time FER pipeline spanning teacher training, ensemble distillation, and deployment-facing evaluation — with artifact-backed evidence at every stage.

**Main findings.** (1) KD/DKD consistently improve temperature-scaled calibration (TS ECE: 0.050→0.027) but do not surpass CE macro-F1 (0.742) in the primary evaluation, demonstrating that calibration and decision-boundary quality are distinct optimisation targets. (2) Domain shift causes severe performance degradation: teacher macro-F1 drops from 0.791 (in-distribution) to 0.393 (eval-only), with minority classes Fear and Disgust degrading to near-zero F1 under webcam conditions. (3) NL/NegL auxiliary losses, as currently tuned, do not provide consistent macro-F1 improvements and can destabilise training. (4) The 2026-02-21 Self-Learning + NegL adaptation candidate passed the offline regression gate but regressed on the same-session webcam replay (macro-F1: 0.525→0.467), establishing the project's most important methodological finding: **offline non-regression is necessary but insufficient for deployment improvement claims.**

**Methodological contribution.** The dual-gate evaluation framework — requiring both broad-distribution offline non-regression and fixed-protocol deployment replay improvement — is the central contribution. This principle, supported by seven formally catalogued negative results (NR-1–NR-7), provides a reusable evaluation discipline for safety-critical model updates in deployed FER systems.

### 9.2 Future Work

Several avenues remain for extending this work:

1. **Deployment benchmarking:** run a timed demo session to report FPS, latency distribution, and jitter/flip-rate on the target CPU device.
2. **LP-loss tuning:** establish a stable KD baseline on the HQ-train pipeline and systematically vary LP-loss weight and embedding layer to identify configurations that improve cross-dataset macro-F1 without harming calibration.
3. **Domain adaptation refinement:** tighten the self-learning buffer construction (stable-only sampling, per-class caps, replay anchors) and complete the BN-only webcam gate to isolate BatchNorm running-stat drift as a failure mode.
4. **Protocol-matched evaluation:** for each paper comparison target, complete a gap checklist covering split, crop, preprocessing, resolution, label mapping, backbone capacity, and training settings to enable strict apples-to-apples reporting.
5. **Capacity scaling:** investigate larger student backbones (e.g., ResNet-50) to quantify the capacity-latency-accuracy trade-off against the current MobileNetV3-Large student.

### 9.3 Feb 2026 addendum: Paper-study-driven implementation and evaluation update

*Note: The results in Sections 9.3.2--9.3.4 are fully evaluated findings produced in Feb 2026 after the main experimental pipeline was established. They are summarised in Section 4.11; the full analysis is retained here to preserve chronological ordering.*

The main body of this report covers Aug 2025 – Jan 2026. Section 9.3 documents Feb-2026 updates that extend the codebase and experimental plan; evaluation results are only stated when backed by the listed artifacts.

#### 9.3.1 Implementation context

In Feb 2026, two extensions were implemented to support the LP-loss screening and paper-comparison evaluations below:

1. **LP-loss [21]:** a locality-preserving auxiliary loss was added to the student training script with a default-off safety posture. When enabled, it applies a within-class compactness objective in embedding space (controlled by weight, layer selection, and neighbour count k).
2. **Post-training evaluation hook:** the training entrypoint supports an optional `--post-eval` flag that automatically runs the standalone evaluation script on eval-only and ExpW manifests after training finishes, generating per-evaluation reliability metrics and a summary artifact.

The experiment order followed a risk-managed sequence: KD baseline first to confirm gate stability, then KD+LP, with DKD and other extensions deferred until the above were stable.
#### 9.3.2 Feb 2026 addendum (Week 1): KD baseline vs KD + LP-loss (evidence-backed)

This subsection records the first short-budget screening results produced after the LP-loss implementation, using the repo’s standard artifacts.

Runs (training outputs):

- KD baseline (5-epoch screening run, Feb 2026)
- KD + LP-loss (w=0.01, k=20, penultimate embedding; 5-epoch screening, Feb 2026)

#### A) HQ-train validation split (from each run’s `reliabilitymetrics.json`)

Table 9.3.2-A (HQ-train validation). Split: val. Protocol: single-crop.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Global T |
| --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep) | 0.7297586 | 0.7281613 | 0.0373908 | 0.7926007 | 4.4717526 |
| KD + LP (w=0.01, k=20, penultimate; 5ep) | 0.7296656 | 0.7276670 | 0.0252364 | 0.7612492 | 3.4970691 |

Interpretation:

- On HQ-train val, KD+LP did not improve raw macro-F1 vs KD baseline in this 5-epoch screening.
- Calibration signals improved on this split (TS ECE and TS NLL both decrease), but this does not imply better cross-domain performance.

#### B) Offline gates

Eval-only (safety gate):

Table 9.3.2-B1 (Offline safety gate: eval-only). Split: test. Protocol: single-crop.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.5162321 | 0.4385411 | 0.0217606 | 1.2961859 |
| KD + LP | 0.5207738 | 0.4411229 | 0.0374865 | 1.2773255 |

ExpW (target-domain proxy):

Table 9.3.2-B2 (Cross-dataset gate: ExpW). Split: test. Protocol: single-crop.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.6311145 | 0.4595847 | 0.0276567 | 1.0635237 |
| KD + LP | 0.6356902 | 0.4583109 | 0.0197645 | 1.0421315 |

Interpretation:

- In this screening, KD+LP slightly increases eval-only macro-F1 but slightly decreases ExpW macro-F1.
- ExpW calibration improves (TS ECE and TS NLL decrease), but raw macro-F1 does not improve here.



#### 9.3.3 Feb 2026 addendum (Week 2): Offline benchmark suite + challenging-benchmark diagnostics (evidence-backed)

This subsection documents Week-2 Feb-2026 diagnostic work aimed at explaining unexpectedly low offline results on (1) eval-only, (2) ExpW, and (3) FER2013 (uniform-7 stress-test). The intent is not to “fix” the numbers in reporting, but to localize failure modes and confirm whether obvious preprocessing mismatches (e.g., CLAHE) are responsible.

Primary evidence artifacts (full path inventory moved to Appendix A.4):

- Offline suite export (canonical scoring snapshot)
- Investigation write-up (manifest integrity + per-class comparisons)

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

Interpretation:

- The low aggregate eval-only result is not uniform across all sources; it is driven primarily by source composition (notably `expw_hq`) and minority-class fragility.

#### B) CLAHE ablation (CE checkpoint): ExpW + FER2013

To test whether preprocessing mismatch is a root cause, ExpW and FER2013 uniform-7 were re-evaluated with CLAHE disabled.

ExpW full manifest:

| Setting | Raw acc | Raw macro-F1 | Fear F1 | Disgust F1 |
| --- | ---: | ---: | ---: | ---: |
| CLAHE on | 0.6576969 | 0.4821205 | 0.2152466 | 0.1631505 |
| CLAHE off | 0.6506155 | 0.4689109 | 0.2040816 | 0.1418021 |

FER2013 uniform-7:

| Setting | Raw acc | Raw macro-F1 | Fear F1 |
| --- | ---: | ---: | ---: |
| CLAHE on | 0.5241429 | 0.4973563 | 0.1881356 |
| CLAHE off | 0.4882857 | 0.4565744 | 0.1543860 |

Interpretation:

- Disabling CLAHE makes both ExpW and FER2013 worse on macro-F1, so CLAHE is not the main cause of the low benchmark results.
- FER2013 Fear remains a dominant failure mode even in the balanced setting, pointing to domain/preprocessing mismatch beyond calibration alone.

#### C) Interpreting “mixed-source” offline tests (why results can appear low)

This project uses several offline tests that intentionally mix multiple sources/domains (e.g., `classification_manifest_eval_only.csv`, and historically `test_all_sources.csv`). These are valuable, but they are also where aggregate scores can appear low even when the model is strong on cleaner single-dataset tests.

Key reasons:

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

#### 9.3.4 Feb 2026 addendum (Week 2): Evidence-backed comparison with provided papers (PDFs)

This subsection records a bounded comparison against published papers. Because papers often differ in label space, split definition, preprocessing/alignment, and metrics (accuracy vs macro-F1; balanced vs imbalanced evaluation), this comparison is used to contextualize our results rather than claim strict SOTA equivalence.

Primary comparison artifacts (full evidence index moved to Appendix A.5):

- A protocol-aware comparison table with comparability flags was constructed for each target paper
- Paper protocol and metric extraction notes (quotable lines + limitations) are documented internally
- The FER2013 official split summary is the primary anchor for gap analysis

#### A) RAF-DB (accuracy)

- Paper (face-regions analysis): reports RAF-DB **whole-face** testing accuracy **82.69%** *with padding* (Table 5).
- Ours (student CE, `test_rafdb_basic`): raw accuracy **86.28%**, raw macro-F1 **0.792**.

Interpretation: this is competitive, though exact split and protocol details may differ from the original study.

#### B) FER2013 (accuracy; split mismatch warning)

- Paper (“State of the Art Performance on FER2013”): reports test accuracy **73.28%** on the **FER2013 public test set**.

Two relevant evaluation regimes exist in this project:

1) **Non-official** stress-test split (FER2013 uniform-7 / folder datasets): useful as a hard gate, but not protocol-matched to the paper.

2) **Official** FER2013 split from `fer2013.csv` (Usage=PublicTest/PrivateTest): protocol-matched on split definition, but still a protocol mismatch if the paper uses ten-crop.

**Figure 9.3-1: FER2013 official split evaluation (CE vs KD vs DKD)**

![FER2013 Official Split Accuracy](figures/fig9_fer2013_official.png)

*Figure 9.3-1.* Accuracy on the official FER2013 PublicTest and PrivateTest splits (n=3,589 each) under single-crop and ten-crop protocols. DKD achieves the highest PublicTest single-crop accuracy (0.614), while all three students fall in the 0.60–0.61 range..

Table 9.3.4-B (FER2013 official split). Split: PublicTest and PrivateTest (n=3,589 each). Protocol: single-crop and ten-crop reported separately.

| Split | Protocol | n | Accuracy | Macro-F1 | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| PublicTest | single-crop | 3589 | **0.614099** | **0.553776** | Appendix A.5 |
| PublicTest | ten-crop | 3589 | **0.609083** | **0.557332** | Appendix A.5 |
| PrivateTest | single-crop | 3589 | **0.608247** | **0.539025** | Appendix A.5 |
| PrivateTest | ten-crop | 3589 | **0.612148** | **0.547634** | Appendix A.5 |

Interpretation: even with the official split and protocol-aware reporting, a strict 1:1 numeric comparison still depends on details that vary across studies (and are not always fully specified): exact preprocessing/alignment, image resolution, training schedule, augmentation, and whether extra data or pretraining was used. Therefore, we treat the official-split table as the strongest **anchor** for gap analysis, but we avoid claiming strict SOTA equivalence without matching those additional variables.

Additional evidence (partial comparable; different split): we also evaluated on a Kaggle FER2013 folder dataset (msambare). This uses a different train/test split and is **not** a strict match to the “public test” protocol, but is useful as a second external sanity check:

- Manifest: FER2013 folder dataset (msambare packaging)
- Metrics artifact (student DKD checkpoint on `test` split):
  - DKD student reliability metrics on FER2013 folder test split

#### C) AffectNet (macro-F1; balanced-subset warning)

- Paper (AffectNet database): Table 7 provides per-class F1 for multiple training approaches. From the weighted-loss approach (Top-1), the macro-average across the eight classes is:
  - Top-1 macro-F1 (Orig): **0.555**
  - Top-1 macro-F1 (skew-normalized): **0.625**
- Repro note: macro-F1 derived from Table 7 values of the original AffectNet paper
- Ours: on `test_affectnet_full_balanced`, student CE achieves raw macro-F1 **0.823** (raw acc **0.822**).

Interpretation: our evaluation uses an explicitly balanced subset, while the paper reports on the original (skewed) AffectNet test set; direct comparison is not appropriate.

#### D) Summary interpretation

- In-domain datasets (e.g., RAF-DB basic, FERPlus) show strong performance for a real-time student.
- The weakest results concentrate in cross-domain/mixed-source scenarios (ExpW/eval-only/FER2013 stress tests) and minority classes, consistent with domain shift + label ambiguity + class fragility rather than uniformly weak modeling.
- The official-split FER2013 table is the strongest anchor for gap analysis, but strict cross-paper comparisons still depend on preprocessing/alignment, training recipe, and protocol details.

#### E) Future work

- Expand protocol-matched evaluations on official splits and document acquisition constraints for licensed datasets.
- Run structured gap analysis across protocol variables (crop/TTA, preprocessing, resolution, label mapping) and training settings.
- Maintain strict protocol separation in reporting; do not mix stress-test splits with official-split claims.
- Continue conservative safety-gated Self-Learning + NegL with standardised deployment-facing metrics.

Appendix pointers (to keep the main report concise):

- Week-2 diagnostics artifact paths + error sampling lists: Appendix A.4
- Paper-comparison evidence index (manifests/checkpoints/metrics artifacts): Appendix A.5

## 10. References

Selected references relevant to the methods used in this project (IEEE format).

[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv preprint arXiv:1503.02531, 2015.

[2] B. Zhao, Q. Cui, R. Song, Y. Qiu, and J. Liang, "Decoupled knowledge distillation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11953–11962.

[3] C. Deng, D. Huang, X. Wang, and M. Tan, "Nested learning: A new paradigm for machine learning," arXiv preprint arXiv:2303.10576, 2023.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4690–4699.

[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.

[6] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1314–1324.

[7] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.

[8] K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.

[9] K. He, X. Zhang, S. Ren, and J. Sun, "Identity mappings in deep residual networks," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2016, pp. 630–645.

[10] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 3–19.

[11] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.

[12] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 9268–9277.

[13] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, "Long-tail learning via logit adjustment," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[14] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.

[15] Y. Kim, J. Yim, J. Yun, and J. Kim, "NLNL: Negative learning for noisy labels," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 101–110.

[16] T. Ishida, G. Niu, W. Hu, and M. Sugiyama, "Learning from complementary labels," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 5639–5649.

[17] A. Mollahosseini, B. Hasani, and M. H. Mahoor, "AffectNet: A database for facial expression, valence, and arousal computing in the wild," *IEEE Trans. Affect. Comput.*, vol. 10, no. 1, pp. 18–31, Jan.–Mar. 2019.

[18] W. Wu, Y. Peng, S. Wang, and Y. He, "YuNet: A tiny millisecond-level face detector," *Mach. Intell. Res.*, vol. 20, pp. 656–665, 2023.

[19] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019. [Online]. Available: https://github.com/huggingface/pytorch-image-models

[20] I. J. Goodfellow, D. Erhan, P. L. Carrier, A. Courville, M. Mirza, B. Hamner, W. Chou, Y. Dauphin, D. Warde-Farley, T. Berg, A. Courville, Y. Bengio, and C. Pal, "Challenges in representation learning: A report on three machine learning contests," in *Proc. Int. Conf. Neural Inf. Process. (ICONIP)*, 2013, pp. 117–124.

[21] S. Li, W. Deng, and J. Du, "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2852–2861.

[22] D. Wang, A. Shelhamer, J. Hoffman, X. Yu, and T. Darrell, "Tent: Fully test-time adaptation by entropy minimization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

## 11. Appendix

### A.0 Interim Report v4 figure-to-artifact mapping

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

### A.1 Evidence inventory (key artifacts)

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

- The interim report (v4, Dec 2025) is available as a separate submission
- Dec-24 mini-report pack index: `research/report of project restart/mini report 24-12-2025/mini report md file/00_index.md`

### A.2 Metric definitions

- **Macro-F1**: unweighted mean of per-class F1.
- **ECE**: Expected Calibration Error (binned absolute gap between accuracy and confidence).
- **NLL**: Negative Log-Likelihood.
- **Temperature scaling**: global temperature $T$ applied to logits; accuracy/macro-F1 remain unchanged, while NLL/ECE can change.

### A.3 Manifest distribution tables

The following distribution summaries are generated directly from the CSV manifests under `Training_data_cleaned/`:

- Generator: `scripts/summarize_manifest_counts.py`
- Outputs: `outputs/manifest_counts_summary.md` and `outputs/manifest_counts_summary.json`

### A.4 Feb 2026 Week-2 diagnostics: artifact inventory

This appendix subsection contains the long-form artifact-path inventory for Section 9.3.3.

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

### A.5 Paper comparison: evidence index and source notes

This appendix subsection contains the long-form evidence index for Section 9.3.4.

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
| FER2013 official (PublicTest / PrivateTest) | Official split from `fer2013.csv` (Usage=PublicTest/PrivateTest); protocol-aware (single-crop + ten-crop) | `Training_data/FER2013_official_from_csv/manifest__publictest.csv` and `.../manifest__privatetest.csv` | Student CE/KD/DKD (best DKD) | Summary: `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`; raw JSONs: `outputs/evals/students/fer2013_official__*__*test__20260212__{singlecrop,tencrop}/reliabilitymetrics.json` | Partial (closest match; remaining differences may include preprocessing/alignment/model/training) |
| FER2013 (folder dataset; msambare) | Non-official folder packaging (stress-test) | `Training_data/fer2013_folder_manifest.csv` | Student DKD | `outputs/evals/students/DKD_20251229_223722__fer2013_folder__test__20260208/reliabilitymetrics.json` | No |
| RAF-DB basic (student) | Single-dataset test (protocol may differ vs paper) | (from offline suite index) | Student CE | Offline suite CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | Partial |
| FERPlus (student) | Single-dataset test (protocol may differ vs paper) | (from offline suite index) | Student CE/KD/DKD | Offline suite CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | Partial |
| AffectNet balanced (student) | Balanced subset test (not paper’s original skew) | (from offline suite index) | Student CE | Offline suite CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | No (balanced-subset mismatch) |
| ExpW (cross-dataset gate) | In-the-wild proxy test | `Training_data_cleaned/expw_full_manifest.csv` | Student DKD variants (incl. ) | `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md` and per-run `outputs/evals/students/*/reliabilitymetrics.json` | No |
| Eval-only (safety gate) | Mixed-source stress test (deployment-aligned) | `Training_data_cleaned/classification_manifest_eval_only.csv` | Student CE baseline and adapted runs | Gate metrics JSONs: `outputs/evals/students/*__eval_only__test__*/reliabilitymetrics.json` | No |
| Mixed-source benchmark (teachers / ensemble) | Mixed-domain benchmark (48,928 rows) | `Training_data_cleaned/test_all_sources.csv` | Teacher ensemble RN18/B3/CNXT (0.4/0.4/0.2) | `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json` | No |

### A.6 MathJax and Docx Equation Compatibility Checklist

This report uses LaTeX-style math notation that is compatible with MathJax rendering and markdown-to-docx conversion workflows.

Recommended conversion-safe rules used in this file:

1. Inline symbols use `$...$`, for example $T$, $p_{max}$, $\tau_{high}$, $\tau_{mid}$.
2. Display equations use `$$...$$` blocks when needed.
3. This notation is compatible with both MathJax rendering and pandoc markdown-to-docx conversion.

Representative equations used by this project (notation summary):

$$
\operatorname{logits}_{\text{scaled}} = \frac{\operatorname{logits}}{T}
$$

$$
\text{NegL gate active} \iff \tau_{mid} \le p_{max} < \tau_{high}
$$

These equations are consistent with the narrative definitions in Sections 2.5 and 4.9.2.
