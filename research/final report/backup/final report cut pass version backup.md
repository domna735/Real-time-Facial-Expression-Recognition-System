# Real-time Facial Expression Recognition System: Final Report

Project Title: Real-time Facial Expression Recognition System via Knowledge Distillation and Self-Learning + Negative Learning (NegL)

Author: Donovan Ma  
Institution: The Hong Kong Polytechnic University (PolyU)  
Supervisor: Prof. Lam  
Report Period: Aug 2025 – Mar 2026  
Document Date: Mar 7, 2026  
Report Version: 2

---

## Abstract

Real-time facial expression recognition (FER) demands not only classification accuracy but also prediction stability, confidence calibration, and robustness to domain shift — requirements that standard offline benchmarks do not adequately capture. This project develops a reproducible, end-to-end real-time FER system for the canonical 7-class emotion space (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using a teacher–student knowledge distillation pipeline. Three teacher backbones (ResNet-18, EfficientNet-B3, ConvNeXt-Tiny) are trained with ArcFace-style margins on 466,284 validated multi-source samples and distilled into a lightweight MobileNetV3-Large student via cross-entropy (CE), knowledge distillation (KD), and decoupled KD (DKD).

Three principal findings emerge. First, KD/DKD consistently improve temperature-scaled calibration (ECE: 0.050 → 0.027) but do not surpass CE macro-F1 (0.742), demonstrating that calibration quality and decision-boundary quality are distinct optimisation targets. Second, domain shift causes persistent minority-class fragility — Fear and Disgust degrade to near-zero F1 under webcam conditions — identifying a structural representational challenge rather than a tuning failure. Third, a safety-gated adaptation candidate combining Self-Learning and Complementary-Label Negative Learning (NegL) passes offline regression gates yet regresses on same-session webcam replay, establishing that offline non-regression is necessary but insufficient for deployment improvement claims.

This third finding motivates the project's key engineering contribution: a **dual-gate evaluation protocol** requiring both broad-distribution offline non-regression and fixed-protocol deployment replay improvement before promoting any checkpoint. Seven negative results (NR-1–NR-7) are formally catalogued with artifact-backed evidence, and a protocol-aware comparison methodology contextualises performance against published benchmarks while respecting split, preprocessing, and metric differences.

## Executive Summary

**One-sentence summary:** A reproducible real-time FER pipeline is implemented using teacher–student knowledge distillation, evaluated with protocol-aware offline benchmarks and deployment-facing domain shift scoring.

**Deliverables:**

| Deliverable | Scope |
| --- | --- |
| Multi-source data pipeline | 466,284 validated samples, 7 emotion classes, CSV manifests with integrity checks |
| Teacher training | ResNet-18 / EfficientNet-B3 / ConvNeXt-Tiny; best teacher macro-F1 0.791 |
| Student distillation | MobileNetV3-Large trained via CE → KD → DKD for CPU inference |
| Real-time demo system | Webcam loop with EMA, hysteresis, vote-window stabilisation; per-frame CSV logging |
| Dual-gate evaluation protocol | Offline non-regression + deployment replay gates with pre-registered thresholds |
| Protocol-aware paper comparison | Comparability flags for split, preprocessing, and metric differences |

**Headline results:**

| Metric | Value | Source |
| --- | --- | --- |
| Student macro-F1 (HQ-train val) | 0.742 (CE best) | Section 4.4 |
| Calibration after distillation (TS ECE) | 0.027 (DKD) | Section 4.4 |
| RAF-DB test accuracy | 86.3% | Section 4.12 |
| FER2013 official-split accuracy | 61.4% | Section 4.12 |
| Domain-shift teacher drop (in-dist → eval-only) | 0.791 → 0.393 macro-F1 | Section 4.2 |
| Negative results formally catalogued | NR-1 to NR-7 | Section 4.9 |

**Key insight:** Offline non-regression is necessary but insufficient for deployment improvement — a finding that motivates the dual-gate evaluation protocol used throughout this project (see Abstract for the three principal findings).

---
## Table of Contents

1. [Introduction and Background](#1-introduction-and-background)
   - 1.1 Research Questions and Objectives
   - 1.2 Contributions
   - 1.3 Relationship to Interim Report v4
2. [Literature Review](#2-literature-review)
   - 2.1 Problem setting  |  2.2 Datasets and protocols  |  2.3 Backbones  |  2.4 KD/DKD  |  2.5 Calibration  |  2.6 Real-time FER  |  2.7 Domain shift  |  2.8 Synthesis and research gap
3. [Methodology](#3-methodology)
   - 3.1 Pipeline overview  |  3.2 Label space  |  3.3 Data cleaning  |  3.4 Teacher training  |  3.5 Ensemble export  |  3.6 Student training  |  3.7 NL/NegL screening  |  3.8 Domain shift track  |  3.9 Feb 2026 extensions
4. [Results & Analysis](#4-results--analysis)
   - 4.1 Dataset integrity  |  4.2 Teacher performance  |  4.3 Ensemble benchmark  |  4.4 Student performance  |  4.5 NL/NegL screening  |  4.6 Webcam scoring  |  4.7 Adaptation & safety gate  |  4.8 Domain shift experiment  |  4.9 Negative results  |  4.10 LP-loss screening  |  4.11 Offline diagnostics  |  4.12 Paper comparison
5. [Demo and Application](#5-demo-and-application)
   - 5.1 Real-time demo pipeline  |  5.2 Deployment KPIs  |  5.3 Checkpoint preference  |  5.4 Temporal stabilisation
6. [Discussion and Limitations](#6-discussion-and-limitations)
   - 6.1 Discussion of key findings  |  6.2 Analytical comparison with published results  |  6.3 Ethical considerations
7. [Project Timeline](#7-project-timeline)
8. [Lessons Learned](#8-lessons-learned)
9. [Conclusion and Next Steps](#9-conclusion-and-next-steps)
   - 9.1 Conclusion  |  9.2 Future Work
10. [References](#10-references)
11. [Appendix](#11-appendix)

---

## 1. Introduction and Background

Real-time FER aims to classify facial expressions from a live stream while meeting constraints that are not captured by typical offline benchmarks:

- Latency and throughput (FPS) constraints
- Stability (avoid flickering predictions)
- Calibration (confidence should be meaningful)
- Domain shift (webcam lighting, sensor noise, motion blur, user-specific effects)

This project adopts a teacher–student knowledge distillation design to reconcile accuracy and speed: train strong teachers offline on a validated multi-source dataset, ensemble them for robustness, distill knowledge into a compact student suitable for real-time inference, and evaluate using both offline metrics (macro-F1, per-class F1, calibration) and deployment-facing metrics (smoothed vs raw performance, jitter flips/min).

### 1.1 Research Questions and Objectives

This project addresses the following research questions:

- **RQ1:** Can knowledge distillation (KD/DKD) from a multi-teacher ensemble improve both classification performance and calibration quality in a lightweight real-time FER student model?
- **RQ2:** How does domain shift (from curated training data to live webcam deployment) affect per-class recognition performance, and which emotion categories are most vulnerable?
- **RQ3:** Can self-learning and negative learning (NegL) techniques safely adapt a deployed FER model to target-domain conditions without regressing broad-distribution performance?
- **RQ4:** What evaluation protocol is appropriate for validating model updates in a real-time FER system where offline metrics may not reflect deployment-facing behaviour?

The corresponding project objectives are:

1. Implement a reproducible teacher–student FER pipeline with artifact-grounded provenance at every stage.
2. Systematically evaluate CE, KD, and DKD distillation strategies on classification accuracy, calibration, and deployment-facing stability.
3. Characterise domain shift effects on minority-class performance and diagnose root causes.
4. Design and validate a safety-gated adaptation loop with dual-gate evaluation (offline non-regression + deployment replay).
5. Document negative results transparently to establish evidence-backed boundaries on what works and what does not.

### 1.2 Contributions

This project makes the following technical and methodological contributions:

1. **Reproducible teacher–student FER pipeline.** An end-to-end, artifact-grounded pipeline covering multi-source data cleaning, teacher training (3 backbones), ensemble softlabel export, and student distillation (CE/KD/DKD) with stored JSON metrics at every stage.

2. **Dual-gate evaluation protocol.** A deployment-aware evaluation protocol requiring both (a) offline non-regression on broad-distribution gates (eval-only, ExpW) and (b) improvement on fixed-protocol webcam replay, before promoting any checkpoint. This is motivated by the empirical finding that offline gate pass does not imply webcam improvement (NR-1).

3. **Systematic negative result documentation.** Seven formally catalogued negative results (NR-1–NR-7) with evidence-backed vs hypothesis classifications, covering adaptation failures, auxiliary loss instability, and calibration–accuracy decoupling.

4. **Domain shift characterisation.** Quantitative analysis of the teacher→student→deployment transfer gap, including per-class fragility analysis (Fear F1 = 0.00 under webcam shift) and root-cause diagnosis (CLAHE mismatch, BN running-stat drift).

5. **Real-time stabilisation analysis.** Deployment-facing metrics (EMA-smoothed accuracy/F1, jitter flips/min) reported alongside offline metrics, with analysis of how probability margin dynamics interact with temporal smoothing.

**Novelty in one line:** not a new SOTA architecture, but a **deployment-grade evaluation protocol** (dual-gate + negative results) applied to a real-time FER pipeline with full artifact provenance.

### 1.3 Relationship to Interim Report v4 (Dec 25, 2025)

This final report extends the Dec-2025 interim deliverable (v4, submitted separately). The interim report’s framing—deployment constraints, domain shift risk, and artifact-grounded provenance—is preserved as the motivation for Sections 1–3; all numeric results remain sourced from run artifacts (Sections 3–4). The Jan-Feb 2026 extensions (NL/NegL screening, webcam scoring protocol, offline safety gate, LP-loss screening, domain shift adaptation) go beyond the interim scope.

## 2. Literature Review

This section reviews the key research areas that motivate the system design choices in this project. The review is organised around seven themes: the inherent challenges in FER as a classification task, the role of datasets and evaluation protocols in enabling fair comparison, backbone architectures and the capacity-efficiency trade-off, knowledge distillation as a model compression strategy, calibration and its relationship to deployment quality, real-time deployment constraints and temporal stabilisation, and domain shift and adaptation risks. The section concludes with a synthesis that identifies the specific research gaps this project addresses.

### 2.1 Problem setting: FER is noisy, imbalanced, and ambiguous

Facial expression recognition (FER) typically maps a face crop to a discrete set of emotion categories (commonly 7 basic emotions). In practice, this mapping is inherently ambiguous:

- **Subtlety and intensity variation:** low-intensity expressions can be visually close to Neutral.
- **Class overlap:** Fear vs Surprise, Disgust vs Angry, and Sad vs Neutral are common confusions in-the-wild.
- **Annotation noise:** crowd-labelled in-the-wild datasets contain label noise and context ambiguity.

Because of these factors, accuracy alone can hide important failure modes when the dataset is class-imbalanced. Techniques such as focal loss [11], class-balanced loss [12], and logit adjustment [13] have been proposed to address long-tailed distributions. For this reason, macro-F1 and per-class F1 are commonly used to reveal minority-class brittleness.

Recent FER-specific methods have targeted these challenges directly. SCN [23] introduces a self-cure network that suppresses uncertain samples via a relabelling module, achieving strong results on RAF-DB and AffectNet. EAC [27] addresses annotation noise by erasing attention consistency between an image and its flipped version, forcing the model to learn noise-robust features rather than memorising noisy labels. These noise-handling approaches are orthogonal to the knowledge distillation strategy used in this project and represent potential extensions for future work.

### 2.2 Datasets and protocols: why "fair comparison" is hard

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

Several recent architectures have been designed specifically for FER. RAN [24] uses region attention to handle pose variation and partial occlusion by learning importance weights for different face regions. DAN [26] applies multi-head cross-attention to distract from noisy features and focus on expression-relevant regions. MA-Net [28] combines global multi-scale features with local attention to capture both holistic expression patterns and fine-grained local cues. POSTER V2 [25] achieves current state-of-the-art on RAF-DB (92.21%) and AffectNet-7 (67.49%) by combining a cross-fusion transformer with landmark-guided attention, using a two-stream design that processes facial landmarks and image features jointly. These methods consistently employ heavier backbones (ResNet-50, ViT, or Swin Transformer) and single-dataset protocols, representing a different design trade-off from the lightweight, multi-source, deployment-facing approach in this project.

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

- **NLL (Negative Log-Likelihood):** penalises overconfident wrong predictions more strongly.

Temperature scaling [14] is a simple post-hoc calibration method that rescales logits by a single scalar $T$ (without changing argmax labels):

- The system uses confidence thresholds (e.g., pseudo-label acceptance for self-learning).
- The UI or downstream logic depends on probability margins.
- Real-time smoothing/hysteresis reacts differently to “peaky” vs “flat” distributions.

In this project, calibration is reported alongside classification metrics because “real-time usable” behaviour often depends on the confidence profile, not only top-1 accuracy.

### 2.6 Real-time FER: temporal stabilisation and deployment KPIs

Real-time FER differs from offline image classification because predictions are made on a stream. A model that is accurate frame-by-frame can still be frustrating if it flickers rapidly between classes due to noise.

Common stabilisation techniques include:

- **Exponential moving average (EMA) smoothing** over probability vectors.
- **Hysteresis** to resist switching classes unless the new class is sufficiently stronger.
- **Voting windows** over recent predictions.

These techniques change what is being optimised and what should be measured. Therefore, reporting both **raw** and **smoothed** behaviour (plus stability metrics like flip-rate/jitter) is important for deployment-aligned evaluation.

### 2.7 Domain shift and adaptation: self-learning and negative learning need safety rails

Cross-domain generalisation is a key challenge for FER. Domain shift can be caused by:

- camera sensor differences and compression artifacts,
- lighting and white balance,
- pose and occlusion,
- face detector/crop jitter,
- subject-specific appearance and expression style.

Two families of techniques are commonly discussed for improving robustness:

- **Test-time adaptation (TTA):** update a subset of parameters at test time using unsupervised objectives (e.g., entropy minimization on confident predictions).
- **Self-learning / pseudo-labeling:** treat confident predictions as pseudo-labels and fine-tune on them.

Both approaches can fail under label noise and distribution drift. Negative (or complementary) learning adds auxiliary objectives that discourage probability mass on likely-wrong classes, but it can also destabilise training if applied too aggressively.

For this reason, this project treats adaptation as a safety-gated process:

- Only update on high-confidence or stable samples.
- Keep parameter updates conservative.
- Require passing an offline regression gate (e.g., eval-only / ExpW) before accepting an adapted checkpoint.

This framing supports the supervisor’s emphasis on fair comparison and analytical gap explanation: improvements must be evidenced by stored artifacts, and claims must respect protocol differences.

### 2.8 Synthesis and Research Gap

The literature above motivates several non-obvious design decisions in this project. First, the persistent failure of minority classes (Fear/Disgust) under domain shift is a known structural challenge in FER [20, 21], not merely a tuning failure — this justifies treating per-class F1 and macro-F1 as primary metrics rather than overall accuracy. Second, while KD/DKD [1, 2] are well-established for model compression, their effect on calibration vs decision-boundary quality is not guaranteed to be aligned, which motivates our dual reporting of macro-F1 alongside ECE/NLL. Third, the test-time adaptation literature [22] warns that naïve entropy-based updates can fail under distribution shift; concrete failure mode analysis (e.g., SAR's entropy threshold $E_0 = 0.4 \times \ln(C)$ for $C$-class problems) informed our conservative gating design.

Fourth, LP-loss [21] was selected as the most promising auxiliary loss for FER because it explicitly preserves local structure in the embedding space, addressing the class-similarity confusions (Fear↔Surprise, Disgust↔Angry) that dominate our error analysis. Finally, the gap between in-distribution validation and deployment-facing evaluation is not unique to this project; the domain adaptation literature consistently emphasises that proxy metrics must be complemented by target-domain evaluation, which motivates our dual-gate protocol.

**Identified research gap.** Despite the maturity of knowledge distillation and domain adaptation techniques individually, no existing work in the FER literature — including recent high-performing methods such as POSTER V2 [25], DAN [26], and EAC [27] — provides (a) a systematic empirical comparison of CE, KD, and DKD distillation strategies evaluated jointly on classification, calibration, *and* deployment-facing stability metrics, or (b) a formal evaluation protocol that gates model promotion on both offline non-regression and deployment-environment replay. Most FER papers, including recent state-of-the-art methods, optimise for a single offline benchmark number (typically RAF-DB or AffectNet accuracy) and do not evaluate temporal prediction stability, probability-margin dynamics, or the failure modes that emerge when an adapted model is tested on the same domain it was adapted from. This project addresses these gaps by implementing and empirically evaluating a dual-gate protocol across all three dimensions.

## 3. Methodology

This section summarises the implemented pipeline and the artifacts that make every claim traceable.

### 3.1 Pipeline overview (figure)

Figure 3.1 (System overview; all metrics are artifact-backed and tied to manifests):

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
  webcam loop + stabilisation]
  H --> I[Live scoring
  (raw vs smoothed + flip-rate)]
  I --> J[Adaptation candidates
  (Self-learning + NegL)]
  J --> K[Offline safety gate
  (eval-only / ExpW)]
  K -->|pass| L[Promote checkpoint]
  K -->|fail| M[Reject + tighten update]
```

### 3.2 Canonical label space

All training and evaluation in this report uses a 7-class mapping:

- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

### 3.3 Data cleaning and manifest validation

All training and evaluation data are managed through CSV manifests with automated integrity validation (466,284 rows, 0 missing paths, 0 bad labels). Split sizes and label distributions are computed directly from the manifests and stored as reproducibility artifacts.

#### 3.3.1 Dataset provenance snapshots (Kaggle / Drive packaging)

Some datasets used in this project are downloaded via Kaggle or shared drives, and their folder packaging may not exactly match the “official split” definitions used in papers.

To make evaluation reproducible and audit-ready, the workflow snapshots the exact local dataset copies used in this workspace. Each snapshot records file counts, total bytes, extension counts, and a stable SHA256 fingerprint over the relative file list (without relying on external URLs).

Each dataset snapshot records file counts, byte totals, and a stable SHA256 fingerprint. For FER2013, an official-split evaluation was performed using the original ICML-format CSV (Usage=Training/PublicTest/PrivateTest); derived manifests and per-split metrics are stored for reproducibility (see Section 4.12).

### 3.4 Teacher training (Stage A, img224)

Teachers are trained using an ArcFace-style protocol with additive angular margin loss and cosine-annealing learning rate scheduling. All teachers use 224x224 input resolution with ImageNet-pretrained weights, and are saved with full provenance metadata.

**Training configuration:**

| Parameter | Value |
| --- | --- |
| Input resolution | 224x224 |
| Pretraining | ImageNet-1K |
| Loss | ArcFace-style (margin + scale) |
| Optimiser | SGD with momentum |
| LR schedule | Cosine annealing |
| Augmentation | Random horizontal flip, colour jitter, random crop |
| Backbone variants | ResNet-18 [8], EfficientNet-B3 [5], ConvNeXt-Tiny [7] |

Each teacher is trained on a source-filtered subset of the primary manifest (e.g., FERPlus + AffectNet balanced + RAF-DB basic, ~226k rows after filtering). Each run produces a standardised output folder containing the best checkpoint, training history, and reliability metrics (accuracy, macro-F1, per-class F1, ECE/NLL with temperature-scaled variants).

### 3.5 Ensemble selection and softlabels export

Teacher ensembles are constructed by weighted logit fusion. The ensemble weights are selected based on performance on a mixed-source benchmark (48,928 rows), and the selected ensemble exports per-sample softlabel probability vectors for subsequent student training.

The selected ensemble (RN18/B3/CNXT = 0.4/0.4/0.2) is evaluated on the mixed-source test set. Its softlabel outputs --- probability vectors over the 7-class label space --- are stored as a reusable CSV artifact for student KD/DKD training.

### 3.6 Student training (CE -> KD -> DKD)

The student model uses MobileNetV3-Large (`mobilenetv3_large_100` via the timm library [19]), chosen for its favourable accuracy-latency trade-off on CPU inference.

**Training configuration:**

| Parameter | Value |
| --- | --- |
| Backbone | MobileNetV3-Large (5.4M params) |
| Input resolution | 224x224 |
| Pretraining | ImageNet-1K |
| Training stages | CE -> KD (T=5) -> DKD |
| Optimiser | Adam |

**HQ training manifest** (verified size and splits): 259,004 rows with split sizes train=213,144 / val=18,020 / test=27,840.

The three student stages are trained sequentially:
1. **CE stage:** standard cross-entropy training on hard labels.
2. **KD stage:** soft targets from the teacher ensemble combined with hard labels (Equation in Section 2.4), with distillation temperature T.
3. **DKD stage:** decoupled KD with separate target-class and non-target-class weighting (Equation in Section 2.4).

Each student run produces the same standardised output structure as teachers, with reliability metrics stored as JSON artifacts.
### 3.7 NL/NegL screening experiments (offline)

NL/NegL screening experiments are documented in dedicated planning and report files, with results summarised via standardised comparison tables. Each compared run is backed by stored reliability metrics.

Terminology note (to avoid confusion in academic reading):

- **NL(proto)** in this project refers to the *Nested Learning* [3] prototype-style auxiliary mechanism used in the Jan-2026 screening runs (an auxiliary objective with a gating/applied fraction).
- **NegL** in this project refers to an entropy-gated *complementary-label negative learning* loss (a “not-this-class” auxiliary loss).

These are distinct mechanisms; throughout this report, “NL” means Nested Learning, and “NegL” means complementary-label negative learning.

#### 3.7.1 Negative Learning (NegL): objective and gating

Intended objective: improve calibration (ECE/NLL) and reduce overconfident mistakes during KD/DKD by adding a **complementary-label negative learning (NegL)** term [16] that discourages probability mass on likely-wrong classes. Formally, given a complementary label $\bar{y}$ (a class that the sample is believed *not* to belong to), the NegL loss is:

$$
\mathcal{L}_{\text{NegL}} = -\log\!\left(1 - p(\bar{y} \mid x)\right)
$$

where $p(\bar{y} \mid x)$ is the model's predicted probability for the complementary label $\bar{y}$. This encourages the model to assign low probability to the specified wrong class.

Key design choice in this project is **gating**:

- NegL is not applied to every sample. A gate decides when NegL is active.
- Gate behaviour is logged into the run’s `history.json` so it can be audited (example summary in Section 4.5).

Practical implication:

- A high entropy threshold can make NegL too selective (low `applied_frac`), reducing its effect.
- A low threshold / high weight can destabilise training and reduce macro-F1 (an instability counterexample is documented in Section 4.5).

#### 3.7.2 Nested Learning (NL(proto)): intent and configuration transparency

In the Jan-2026 screening runs, “NL(proto)” refers to the prototype-style auxiliary signal used in the compare tables in Section 4.5.

All NL(proto) configurations reported below (e.g., `dim`, `m`, `thr`, `top-k`) are taken verbatim from the compare artifact and run identifier to ensure reproducibility.

### 3.8 Domain shift track (webcam + real-time scoring + conservative adaptation)

The domain shift track is documented in dedicated planning and report files. Live scoring artifacts (per labelled webcam run) and offline safety gate evaluations (on the eval-only manifest, 110,333 rows) are stored as JSON artifacts.

#### 3.8.1 Pre-registered evaluation thresholds

Before any result is generated, the following pass/fail thresholds are fixed (derived from domain_shift/evaluation_plan.md):

| Gate | FAIL condition | WIN (improvement) condition |
| --- | --- | --- |
| Offline non-regression | macro-F1 drops > 0.01 **or** any minority-class F1 drops > 0.02 vs baseline | minority-F1 improves ≥ 0.01 with no macro-F1 regression |
| Deployment replay | smoothed macro-F1 decreases **or** jitter increases > 10 % | smoothed macro-F1 improves **and** jitter does not increase > 10 % |

A checkpoint must pass **both** gates to be promoted; passing one gate alone is insufficient (see NR-1).

#### 3.8.2 Domain shift improvement via Self-Learning + Negative Learning (NegL)

This extension targets **webcam domain shift** by adding a safe adaptation loop. The design principle is: only accept a target-domain update if it passes an offline regression gate.

Implemented components already evidenced in this project:

- **Webcam measurement protocol** (raw vs smoothed metrics, jitter) as reported in Section 4.6.
- **Offline safety gate** on the eval-only manifest, as reported in Section 4.7.

Planned (but not yet claimed as successful results here):

- Add self-learning using high-confidence pseudo-labels plus NegL for medium-confidence samples, with strict defaults and rollback.

Executed evidence update (Feb 2026):

- A Self-Learning + manifest-driven NegL adaptation attempt passed the offline eval-only gate within rounding when preprocessing and BatchNorm behaviour were controlled, but regressed on same-session webcam replay (see Section 4.8).

### 3.9 Feb 2026 implementation extensions

The following implementation extensions were added in Feb 2026 to support the evaluations reported in Sections 4.10–4.12.

1. **LP-loss [21]:** a locality-preserving auxiliary loss was added to the student training script with a default-off safety posture. When enabled, it applies a within-class compactness objective in embedding space (controlled by weight, layer selection, and neighbour count $k$).
2. **Post-training evaluation hook:** the training entrypoint supports an optional `--post-eval` flag that automatically runs the standalone evaluation script on eval-only and ExpW manifests after training finishes, generating per-evaluation reliability metrics and a summary artifact.

The experiment order followed a risk-managed sequence: KD baseline first to confirm gate stability, then KD+LP, with DKD and other extensions deferred until the above were stable. Results are reported in Sections 4.10–4.12.

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
| RN18 | Stage-A val (18165) | 0.786 | 0.781 | 4.026 | 0.880 | 0.205 | 0.149 |
| B3 | Stage-A val (18165) | 0.796 | 0.791 | 3.222 | 0.787 | 0.199 | 0.084 |
| CNXT | Stage-A val (18165) | 0.794 | 0.789 | 3.101 | 0.770 | 0.201 | 0.082 |

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
| eval_only | 11890 | 0.427 | 0.373 | 0.471 | 0.393 | 0.441 | 0.389 |
| expw_full | 9179 | 0.499 | 0.374 | 0.584 | 0.407 | 0.511 | 0.382 |
| test_all_sources | 48928 | 0.645 | 0.617 | 0.674 | 0.645 | 0.665 | 0.638 |

Interpretation (significance of the performance gap):

- The Stage-A validation metrics substantially overestimate performance under mixed-domain stress tests. For example, the best teacher (B3) drops from 0.791 macro-F1 on Stage-A validation to 0.393 on eval-only — a **50% relative decrease**. This gap is expected because Stage-A uses a filtered in-distribution mixture, while `eval_only` / `expw_full` / `test_all_sources` intentionally include harder domain shift + label noise.
- Practical implication: **teacher selection by Stage-A validation alone is insufficient** for deployment-facing robustness; hard-gate evaluation must be treated as a separate requirement.
- Hypothesised causes of the hard-gate gap:
  - **Domain mismatch:** ExpW/webcam-like imagery differs from curated training sources in lighting, pose, expression intensity, and crop statistics.
  - **Label noise and class imbalance:** minority classes (Fear/Disgust) are disproportionately affected, lowering macro-F1 more than accuracy.
  - **Preprocessing alignment:** differences in face alignment/cropping/CLAHE policy can shift the evaluation distribution even when the backbone is unchanged.

### 4.3 Ensemble robustness benchmark (mixed-source)

The ensemble benchmark evaluates a weighted teacher combination on the full mixed-source test set (48,928 rows).

Selected ensemble configuration and results:

- Weights: RN18/B3/CNXT = 0.4/0.4/0.2
- Accuracy: 0.687
- Macro-F1: 0.660

Additional metrics (same ensemble):

- NLL: 4.077
- ECE: 0.288
- Brier: 0.591

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
| CE | 0.750 | 0.742 | 1.315 | 0.778 | 0.131 | 0.050 | 3.228 |
| KD | 0.735 | 0.733 | 2.093 | 0.768 | 0.215 | 0.028 | 5.000 |
| DKD | 0.737 | 0.738 | 1.512 | 0.765 | 0.209 | 0.027 | 3.348 |

Per-class F1 (raw; rounded to 4 d.p.; copied from each run’s `reliabilitymetrics.json`):

| Stage | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.7263 | 0.6428 | 0.7640 | 0.8014 | 0.7170 | 0.7871 | 0.7550 |
| KD | 0.7237 | 0.6782 | 0.7447 | 0.7610 | 0.7234 | 0.7801 | 0.7224 |
| DKD | 0.7255 | 0.6828 | 0.7561 | 0.7596 | 0.7286 | 0.7915 | 0.7185 |

**Figure 4.4-2: Student calibration comparison (Raw vs Temperature-Scaled)**

![Calibration: Raw ECE/NLL vs TS ECE/NLL](figures/fig3_calibration_comparison.png)

*Figure 4.4-2.* Calibration comparison showing Raw vs Temperature-Scaled ECE and NLL for the three student checkpoints. KD and DKD achieve substantially lower TS ECE (0.028 and 0.027 respectively) compared to CE (0.050), demonstrating the calibration benefit of distillation.

Interpretation: CE provides the best raw macro-F1 (0.742); KD/DKD improve TS ECE substantially (0.050 → 0.027) but do not improve macro-F1. This decoupling arises because teacher-distribution matching optimises probability quality rather than decision boundaries, and student capacity/hyperparameters can dominate macro-F1 outcomes.

#### 4.4.1 Consolidated cross-gate comparison (CE vs KD vs DKD)

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
- On the ExpW full cross-dataset gate, the best DKD checkpoint achieves macro-F1 0.460 with TS ECE 0.037, reinforcing the calibration benefit pattern.
- The consistent fragility remains minority classes (Fear/Disgust), supporting the domain-shift + label-noise framing rather than a conclusion that the model is fundamentally weak.

### 4.5 NL/NegL screening results (Jan 2026, offline)

This section reports the NL/NegL screening outcome in compact form; full run-by-run tables are available in the internal report and comparison artifacts.

**Table 4.5-1: NL/NegL screening summary (key checkpoints)**

| Stage | Configuration | Raw macro-F1 | TS ECE | Minority-F1 (lowest-3) | Outcome |
| --- | --- | ---: | ---: | ---: | --- |
| KD (5 ep) | baseline | 0.7266 | 0.0271 | 0.6973 | reference |
| KD (5 ep) | +NegL (ent=0.7) | 0.7198 | 0.0398 | 0.6827 | regression |
| KD (5 ep) | +NegL+NL(proto) | 0.5204 | - | - | severe instability |
| DKD (resume) | baseline | 0.7368 | 0.0348 | 0.7045 | reference |
| DKD (resume) | +NegL (ent=0.7) | 0.7348 | 0.0348 | 0.7024 | slight regression |
| DKD (resume) | +NL(proto) | 0.7179 | 0.0452 | 0.6883 | regression |

Key interpretation:

- NL/NegL did not provide consistent macro-F1 or minority-F1 gains under tested short-budget settings.
- NegL gating is highly sensitive: with high thresholds, `applied_frac` quickly drops to near-zero; with more aggressive settings, instability risk increases.
- Conclusion for this submission: NL/NegL are retained as conditional research components, not promoted as default improvements.

### 4.6 Domain shift: live webcam scoring results (Jan 2026)

**Temperature policy for replay comparisons.** All webcam replay metrics in this section and in Section 4.8 are computed using the same post-hoc temperature-scaling value (optimised on the HQ-train validation set). When comparing checkpoints trained with different distillation stages (CE vs KD vs DKD), each checkpoint's own calibrated temperature is used. No temperature re-tuning is performed between replay sessions, ensuring that differences in smoothed metrics reflect model behaviour rather than calibration artefacts.

**Figure 4.6-1: Webcam demo — raw vs smoothed per-class F1**

![Webcam Raw vs Smoothed Per-class F1](figures/fig8_webcam_raw_vs_smoothed.png)

*Figure 4.6-1.* Per-class F1 for raw vs EMA-smoothed predictions during the webcam demo session (20260126_205446, n=4,154). Smoothing improves most classes but Fear remains at 0.00 F1 (393 samples), revealing a critical deployment gap.

**Figure 4.6-2: Webcam confusion matrix (smoothed predictions)**

![Confusion Matrix — Webcam Demo](figures/fig4_confusion_matrix_webcam.png)

*Figure 4.6-2.* Confusion matrix from the webcam demo session (smoothed predictions, n=4,154). Fear samples are overwhelmingly misclassified as Disgust (304/393); Sad samples are almost entirely predicted as Disgust (479/542). This reveals the structural confusion between minority/negative-valence classes under webcam domain shift.


Key deployment-aligned metrics (raw vs smoothed):

| Run | Raw acc | Raw macro-F1 (present) | Raw minority-F1 (lowest-3) | Smoothed acc | Smoothed macro-F1 (present) | Smoothed minority-F1 (lowest-3) | Jitter flips/min |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20260126_205446 | 0.528 | 0.472 | 0.156 | 0.588 | 0.525 | 0.161 | 14.864 |
| 20260126_215903 | 0.464 | 0.493 | 0.337 | 0.514 | 0.555 | 0.362 | 17.910 |

Per-class highlights (smoothed metrics; includes supports so interpretation is distribution-aware):

| Run | Fear F1 | Fear support | Sad F1 | Sad support |
| --- | ---: | ---: | ---: | ---: |
| 20260126_205446 | 0.0000 | 393 | 0.0301 | 542 |
| 20260126_215903 | 0.3432 | 1316 | 0.4372 | 1540 |

The two runs have different per-class supports (emotion mix), so the comparison should be treated as behavioural evidence rather than a strict A/B test.

### 4.7 Domain shift: conservative adaptation and offline safety gate


Table 4.7-1 (Offline safety gate). Split: test. Protocol: single-crop.

| Model | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Baseline (CE20251223) | 0.567 | 0.486 | 0.059 | 1.229 | baseline |
| Head-only FT | 0.548 | 0.451 | 0.060 | 1.289 | fail (macro-F1 drop) |
| BN-only FT | 0.549 | 0.451 | 0.061 | 1.289 | fail (macro-F1 drop) |

Interpretation:

- Both head-only and BN-only fine-tuning (as configured in these **early Jan-2026 FT runs**) reduce offline macro-F1 on a broader eval-only distribution; these checkpoints should not be promoted beyond experiments.
- **Hypothesis:** While BN-only tuning is often considered safer, it can still shift feature statistics enough to reduce generalisation if the adaptation buffer is narrow, if the pseudo-label signal is noisy, or if BatchNorm running statistics drift (which can occur even in “head-only” intent if BN layers remain in `train()` mode).
- Note: Section 4.8 documents a later 2026-02-21 adaptation attempt where preprocessing (`use_clahe`) and BatchNorm running-stat updates are controlled; that later candidate can pass the offline gate within rounding but still fails the deployment-facing webcam A/B.

### 4.8 Domain shift adaptation experiment (Feb 2026)

#### Issues identified and corrections applied

Two issues were identified and fixed during the Feb-2026 iteration: (1) a **preprocessing mismatch** (baseline used use_clahe=True while adapted checkpoint used use_clahe=False) caused false regression signals, fixed by enforcing consistent CLAHE settings; (2) **BatchNorm running-stat drift** under small-buffer tuning, fixed by forcing BN layers to eval() when ``--tune`` does not specify last-layer-only mode. A replay inference utility was also added to enable fair webcam A/B scoring on the same recorded session.

**Self-Learning + NegL method summary.** The adaptation buffer was built from a recorded session's per-frame log by selecting stable frames with the model's predicted label as pseudo-label. High-confidence frames ($p_{\max} \ge \tau_{\text{high}}$) served as pseudo-labelled positives; medium-confidence frames ($\tau_{\text{mid}} \le p_{\max} < \tau_{\text{high}}$) received NegL-only supervision (discouraging probability mass on the model's own uncertain prediction); low-confidence frames were excluded.

Gate check result (eval-only manifest):

- Baseline checkpoint: CE student (Dec 2025 main run)
- Adapted checkpoint: head-only + CLAHE + BN-stats frozen (Feb 2026)

On the eval-only manifest (test split, single-crop), the adapted checkpoint matches the baseline within rounding in macro-F1, indicating the offline safety gate can be passed when preprocessing and BN behaviour are controlled.

Webcam A/B result (same recorded session, same manual labels):

**Webcam A/B comparison** (same recorded session, same manual labels):

Summary (smoothed predictions):

- Baseline: accuracy 0.5879, macro-F1 0.5248, minority-F1(lowest-3) 0.1609, jitter 14.86 flips/min
- Adapted: accuracy 0.5269, macro-F1 0.4667, minority-F1(lowest-3) 0.1384, jitter 14.16 flips/min

**Figure 4.9-1: Domain shift adaptation A/B — baseline vs adapted (webcam replay)**

![Adaptation A/B Comparison](figures/fig10_adaptation_ab.png)

*Figure 4.9-1.* A/B comparison on the same labelled webcam session. The adapted checkpoint (Self-Learning + NegL) slightly reduces jitter (−0.7 flips/min) but regresses accuracy (−0.061) and macro-F1 (−0.058). This is the key negative result NR-1: offline gate pass does not imply deployment improvement.

Interpretation: The adapted checkpoint slightly reduces jitter but regresses accuracy and macro-F1, supporting the safety-gated adaptation framing: **passing an offline gate is necessary but not sufficient** for deployment improvement. Hypothesised mechanisms include small/correlated buffer amplifying overfitting, pseudo-label noise from transition frames, NegL target policy risk (discouraging correct-but-uncertain predictions), and objective mismatch between broad offline gates and deployment-specific temporal scoring. This is evidence from a single session and candidate; it demonstrates a failure mode but does not prove that self-learning + NegL cannot help under other configurations. For this FYP, we therefore **do not** claim any positive adaptation result; we only claim a documented failure mode.

### 4.9 Consolidated Negative Results (Single-Report Submission Version)

This section consolidates the key negative results into a single reference for submission.

#### 4.9.1 Negative result matrix (evidence-backed)

| ID | Negative result statement | Status | Main evidence |
| --- | --- | --- | --- |
| NR-1 | 2026-02-21 Self-Learning + manifest-driven NegL candidate passed offline eval-only non-regression (within rounding) but regressed on identical-session webcam replay | Evidence-backed | Section 4.8, per-session scoring artifacts |
| NR-2 | Early head-only and BN-only adaptation candidates failed the offline eval-only gate (macro-F1 drop) | Evidence-backed | Eval-only gate metrics (Appendix A.1) |
| NR-3 | NL/NegL short-budget KD/DKD screening did not produce consistent macro-F1 or minority-F1 gains | Evidence-backed | Per-run training history and reliability metrics (Section 4.5) |
| NR-4 | KD/DKD improved calibration (TS ECE/TS NLL) but did not outperform CE macro-F1 in the Dec-2025 main snapshot | Evidence-backed | Artifacts listed in Section 4.4 |
| NR-5 | Strong Stage-A teacher validation did not transfer to hard-gate robustness (eval-only, ExpW) | Evidence-backed | Teacher benchmark summary and hard-gate analysis (Section 4.2) |
| NR-6 | Hard-gate weakness concentrates in minority/confusable classes (especially Fear/Disgust); calibration correction alone is insufficient | Evidence-backed | Per-class F1 analysis (Section 4.4.1, Section 4.11) |
| NR-7 | LP-loss short-budget screening improved some calibration terms but did not show a clear ExpW macro-F1 gain | Evidence-backed | Section 4.10 gate tables |

#### 4.9.2 Interpretation

The matrix above supports one central conclusion: **offline non-regression is necessary but insufficient for deployment improvement.**

Most critical example (NR-1, same labelled replay session):

| Model | Accuracy | Macro-F1 | Minority-F1 (lowest-3) | Jitter (flips/min) |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.5879 | 0.5248 | 0.1609 | 14.86 |
| Adapted | 0.5269 | 0.4667 | 0.1384 | 14.16 |

Although jitter slightly improved, deployment-facing accuracy and macro-F1 regressed. This motivates the dual-gate policy used throughout the report: pass offline safety gates and deployment replay before promotion. Detailed per-NR analysis, causal map, and full validity threats are retained in the project notes and artifact-backed appendices.

### 4.10 LP-loss screening: KD baseline vs KD + LP-loss (Feb 2026)

This subsection records the first short-budget screening results produced after the LP-loss implementation, using the repo’s standard artifacts.

Runs (training outputs):

- KD baseline (5-epoch screening run, Feb 2026)
- KD + LP-loss (w=0.01, k=20, penultimate embedding; 5-epoch screening, Feb 2026)

#### A) HQ-train validation split (from each run’s `reliabilitymetrics.json`)

Table 4.10-A (HQ-train validation). Split: val. Protocol: single-crop.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL | Global T |
| --- | ---: | ---: | ---: | ---: | ---: |
| KD baseline (5ep) | 0.730 | 0.728 | 0.037 | 0.793 | 4.472 |
| KD + LP (w=0.01, k=20, penultimate; 5ep) | 0.730 | 0.728 | 0.025 | 0.761 | 3.497 |

Interpretation:

- On HQ-train val, KD+LP did not improve raw macro-F1 vs KD baseline in this 5-epoch screening.
- Calibration signals improved on this split (TS ECE and TS NLL both decrease), but this does not imply better cross-domain performance.

#### B) Offline gates

Eval-only (safety gate):

Table 4.10-B1 (Offline safety gate: eval-only). Split: test. Protocol: single-crop.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.516 | 0.439 | 0.022 | 1.296 |
| KD + LP | 0.521 | 0.441 | 0.037 | 1.277 |

ExpW (target-domain proxy):

Table 4.10-B2 (Cross-dataset gate: ExpW). Split: test. Protocol: single-crop.

| Run | Raw acc | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: | ---: |
| KD baseline | 0.631 | 0.460 | 0.028 | 1.064 |
| KD + LP | 0.636 | 0.458 | 0.020 | 1.042 |

Interpretation:

- In this screening, KD+LP slightly increases eval-only macro-F1 but slightly decreases ExpW macro-F1.
- ExpW calibration improves (TS ECE and TS NLL decrease), but raw macro-F1 does not improve here.



### 4.11 Offline benchmark diagnostics (Feb 2026)

This diagnostic section is condensed for submission; full logs and per-source files are listed in Appendix A.4.

Key findings:

1. Low aggregate eval-only performance is composition-driven, with `expw_hq` materially weaker than `expw_full`.
2. CLAHE is not the main failure cause: disabling CLAHE reduced macro-F1 on both ExpW and FER2013 checks.
3. Mixed-source benchmarks are best used as regression gates, not stand-alone quality claims.

Representative diagnostic values (CE checkpoint):

| Check | Key result |
| --- | --- |
| Eval-only per-source | `expw_full` macro-F1 0.4895 vs `expw_hq` 0.2789 |
| ExpW CLAHE ablation | macro-F1 0.4821 (on) vs 0.4689 (off) |
| FER2013 CLAHE ablation | macro-F1 0.4974 (on) vs 0.4566 (off) |

Interpretation: domain-shift and minority-class fragility are primary contributors to low mixed-source scores; preprocessing mismatch alone does not explain the gap.

### 4.12 Protocol-aware paper comparison (Feb 2026)

This subsection presents a bounded comparison against published papers. Because papers often differ in label space, split definition, preprocessing/alignment, and metrics (accuracy vs macro-F1; balanced vs imbalanced evaluation), this comparison contextualises our results rather than claiming strict SOTA equivalence.

Primary comparison artifacts (full evidence index moved to Appendix A.5):

- A protocol-aware comparison table with comparability flags was constructed for each target paper
- Paper protocol and metric extraction notes (quotable lines + limitations) are documented internally
- The FER2013 official split summary is the primary anchor for gap analysis

#### A) RAF-DB (accuracy)

- Paper (face-regions analysis): reports RAF-DB **whole-face** testing accuracy **82.69%** *with padding* (Table 5).
- Ours (student CE, `test_rafdb_basic`): raw accuracy **86.28%**, raw macro-F1 **0.792**.

Interpretation: this is competitive, though exact split and protocol details may differ from the original study. **Caveat:** the paper_training_recipe_checklist notes that this paper reportedly selects the best model checkpoint by test-set performance rather than by a held-out validation set, which would inflate the reported number relative to a protocol that selects by validation loss.

#### B) FER2013 (accuracy; split mismatch warning)

- Paper (“State of the Art Performance on FER2013”): reports test accuracy **73.28%** on the **FER2013 public test set**.

Two relevant evaluation regimes exist in this project:

1) **Non-official** stress-test split (FER2013 uniform-7 / folder datasets): useful as a hard gate, but not protocol-matched to the paper.

2) **Official** FER2013 split from `fer2013.csv` (Usage=PublicTest/PrivateTest): protocol-matched on split definition, but still a protocol mismatch if the paper uses ten-crop.

**Figure 4.12-1: FER2013 official split evaluation (CE vs KD vs DKD)**

![FER2013 Official Split Accuracy](figures/fig9_fer2013_official.png)

*Figure 4.12-1.* Accuracy on the official FER2013 PublicTest and PrivateTest splits (n=3,589 each) under single-crop and ten-crop protocols. DKD achieves the highest PublicTest single-crop accuracy (0.614), while all three students fall in the 0.60–0.61 range.

Table 4.12-B (FER2013 official split). Split: PublicTest and PrivateTest (n=3,589 each). Protocol: single-crop and ten-crop reported separately.

| Split | Protocol | n | Accuracy | Macro-F1 | Evidence |
| --- | --- | ---: | ---: | ---: | --- |
| PublicTest | single-crop | 3589 | **0.614** | **0.554** | Appendix A.5 |
| PublicTest | ten-crop | 3589 | **0.609** | **0.557** | Appendix A.5 |
| PrivateTest | single-crop | 3589 | **0.608** | **0.539** | Appendix A.5 |
| PrivateTest | ten-crop | 3589 | **0.612** | **0.548** | Appendix A.5 |

Interpretation: even with the official split and protocol-aware reporting, a strict 1:1 numeric comparison still depends on details that vary across studies (and are not always fully specified): exact preprocessing/alignment, image resolution, training schedule, augmentation, and whether extra data or pretraining was used. Therefore, we treat the official-split table as the strongest **anchor** for gap analysis, but we avoid claiming strict SOTA equivalence without matching those additional variables.

#### C) AffectNet (not directly comparable)

Our student CE achieves macro-F1 0.823 on a balanced AffectNet subset, but the original paper reports on the skewed test set (macro-F1 0.555); direct comparison is not appropriate due to this distribution mismatch.

#### D) Consolidated landscape comparison

Table 4.12-D consolidates published results on RAF-DB (the most widely reported benchmark among recent FER methods) to contextualise where this project's lightweight student sits relative to the field. All reported numbers are single-crop accuracy on the RAF-DB test set unless noted otherwise.

**Table 4.12-D: RAF-DB accuracy landscape (7-class, single-crop unless noted).**

| Method | Year | Backbone | Params (approx.) | RAF-DB Accuracy | Protocol notes |
| --- | --- | --- | ---: | ---: | --- |
| SCN [23] | 2020 | ResNet-18 | 11M | 87.03% | Self-cure relabelling |
| RAN [24] | 2020 | ResNet-18 | 11M | 86.90% | Region attention + occlusion |
| LP-loss [21] | 2017 | VGG-16 | 138M | 84.12% | Locality-preserving embedding |
| MA-Net [28] | 2021 | ResNet-18 | 11M | 88.42% | Multi-scale + local attention |
| EAC [27] | 2022 | ResNet-18 | 11M | 89.99% | Erasing attention consistency |
| DAN [26] | 2023 | ResNet-18 | 11M | 89.70% | Multi-head cross-attention |
| POSTER V2 [25] | 2023 | ViT + ResNet-50 | ~100M | 92.21% | Cross-fusion transformer |
| **Ours** | 2026 | MobileNetV3-Large | **5.4M** | **86.28%** | Multi-source training; CPU-targeted |

Key observations:

- Our MobileNetV3-Large student (5.4M params) achieves 86.28%, competitive with the ResNet-18-based methods SCN (87.03%) and RAN (86.90%) that have 2× the parameter count.
- The gap to current SOTA (POSTER V2, 92.21%) is approximately 6 percentage points, attributable to the backbone capacity trade-off (5.4M vs ~100M parameters) and the multi-source noisy training regime vs single-dataset clean protocols.
- **Protocol caveat:** our evaluation uses the RAF-DB basic test split via the offline benchmark suite; differences in face alignment, crop policy, and augmentation may shift results by 1–2 percentage points in either direction.

#### E) Summary interpretation

- In-domain datasets (e.g., RAF-DB basic, FERPlus) show strong performance for a real-time student.
- The weakest results concentrate in cross-domain/mixed-source scenarios (ExpW/eval-only/FER2013 stress tests) and minority classes, consistent with domain shift + label ambiguity + class fragility rather than uniformly weak modelling.
- The official-split FER2013 table is the strongest anchor for gap analysis, but strict cross-paper comparisons still depend on preprocessing/alignment, training recipe, and protocol details.

Appendix pointers (to keep the main report concise):

- Week-2 diagnostics artifact paths + error sampling lists: Appendix A.4
- Paper-comparison evidence index (manifests/checkpoints/metrics artifacts): Appendix A.5

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

**Table 5.2-1: Deployment KPI summary**

| KPI | Target | Measured | Source |
| --- | --- | --- | --- |
| Classification accuracy (in-domain) | Competitive with published MobileNet-class FER | 0.742 macro-F1 (HQ-train val) | Section 4.4 |
| Calibration (ECE) | < 0.05 after temperature scaling | 0.027 (DKD, TS) | Section 4.4 |
| Smoothed webcam macro-F1 | > raw macro-F1 (smoothing benefit) | 0.525 smoothed vs 0.472 raw | Section 4.6 |
| Jitter (flips/min) | Low enough for comfortable use | 14.9–17.9 | Section 4.6 |
| FPS / latency (CPU) | ≥ 25 FPS for real-time feel | **Not yet measured** | Section 9.2 |

**Existing measurements.** Live classification behaviour (raw vs smoothed macro-F1/accuracy and jitter flips/min) has been captured from per-session scoring artefacts.

**Limitation: FPS/latency benchmarks not yet reported.** A timed benchmark run on the target device (CPU-only and GPU) has not been conducted as of this submission. FPS and latency-distribution numbers are therefore not claimed in this report. This gap is significant for a project titled "Real-time" FER and is prioritised as the first item in the future work plan (Section 9.2).

The demo pipeline already logs per-frame timestamps, so the benchmark procedure is straightforward: (1) replay a fixed 60-second video clip through the full pipeline (face detection + classification + smoothing), (2) compute median and 95th-percentile per-frame latency, and (3) report sustained FPS. Given that MobileNetV3-Large (5.4 M parameters, ~0.22 GFLOPs) was designed for mobile/edge deployment, achieving ≥ 25 FPS on a modern laptop CPU is expected, but this remains an empirical claim to be verified.

### 5.3 Checkpoint preference observation (deployment-facing)

During informal live webcam comparisons (Feb 2026), the CE student checkpoint appeared more stable than KD+LP-loss and DKD checkpoints (fewer visible label flickers). This is consistent with the hypothesis that KD/DKD students inherit teacher uncertainty patterns, producing smaller probability margins that interact poorly with EMA/hysteresis smoothing under webcam domain shift. Converting this observation into an artifact-backed claim requires a controlled labelled replay protocol (noted in Section 9.2).

### 5.4 Temporal stabilisation and model quality separation

An important interpretive caveat applies to all webcam replay metrics: **raw model quality and post-processing quality are confounded in deployment-facing scores.** The demo pipeline applies EMA smoothing, hysteresis, and optional vote-window filtering after the classifier's softmax output. These temporal filters can mask or amplify differences in underlying classifier accuracy, meaning that a higher smoothed macro-F1 does not necessarily indicate a better model — it may reflect a better interaction between the model's probability margin distribution and the filter parameters.

This confound was identified during the domain-shift evaluation planning (field_transfer_framework.md, Experiment 2) and motivates the following protocol discipline:

1. **Raw and smoothed metrics are always reported side-by-side** (Tables in Sections 4.6 and 4.8) so that readers can assess how much of the deployment score comes from the model vs the filter.
2. **Jitter (flips/min) is reported alongside F1** to surface cases where smoothing suppresses label switching without improving classification — a form of false stability.
3. **Future work (Section 9.2)** recommends a controlled ablation varying filter parameters per checkpoint to disentangle model quality from filter quality.

## 6. Discussion and Limitations

### 6.1 Discussion of key findings

This section first summarises what the project delivers, then maps the experimental results to the research questions and discusses limitations.

**What this project delivers.** The primary deliverable is a working, end-to-end real-time FER system: a MobileNetV3-Large student (5.4M parameters) that achieves 0.742 macro-F1 on the in-distribution evaluation and 86.3% accuracy on RAF-DB — competitive with ResNet-18-based methods that have twice the parameter count (Table 4.12-D). The student benefits from KD/DKD distillation for substantially improved calibration (TS ECE: 0.027), which is important for any downstream system that uses confidence thresholds. The pipeline is fully reproducible with stored manifests, JSON metrics, and checksummed artifacts at every stage. The real-time demo runs on CPU with temporal stabilisation (EMA, hysteresis, vote window) and per-frame logging for offline analysis.

The sections below discuss the boundaries and limitations of these results, organised by research question.

#### 6.1.1 RQ1: Distillation improves calibration but not decision boundaries

KD and DKD consistently improve temperature-scaled calibration (TS ECE: 0.050 to 0.027) compared to the CE baseline, but neither surpasses CE macro-F1 (0.742) in the primary HQ-train evaluation (Section 4.4). This demonstrates that teacher-distribution matching optimises a fundamentally different objective than hard-label classification. The implication for practitioners is that **KD/DKD should be evaluated on calibration and classification metrics independently**, rather than assuming that better soft-target matching translates to better decision boundaries.

#### 6.1.2 RQ2: Domain shift causes systematic minority-class fragility

Domain shift from curated training data to deployment conditions causes severe and systematic performance degradation. Teacher macro-F1 drops from 0.791 (in-distribution) to 0.393 (eval-only), a 50% relative decrease (Section 4.2.2). This degradation is not uniform: Fear and Disgust consistently degrade to near-zero F1 under webcam conditions (Section 4.6), while majority classes (Happy, Neutral) remain relatively robust. The CLAHE ablation (Section 4.11) confirms this is not primarily a preprocessing issue, pointing instead to a structural representational mismatch for low-intensity, ambiguous emotion categories.

#### 6.1.3 RQ3: Adaptation can pass offline gates yet fail deployment criteria

The NR-1 result (Section 4.8) is the most consequential finding: a Self-Learning + NegL adaptation candidate passed the offline eval-only non-regression gate yet regressed on same-session webcam replay (macro-F1: 0.525 to 0.467). NL/NegL auxiliary losses, as currently tuned, do not provide consistent macro-F1 improvements and can destabilise training (Section 4.5). This establishes that adaptation techniques require deployment-facing validation, not just offline non-regression.

#### 6.1.4 RQ4: A dual-gate evaluation protocol is necessary

The evidence from NR-1 through NR-7 collectively motivates the dual-gate evaluation protocol: both broad-distribution offline non-regression and fixed-protocol deployment replay improvement must be satisfied before promoting any checkpoint. Different evaluation regimes answer fundamentally different questions:

- **Training-time validation** (teacher Stage-A val; student HQ-train val) measures in-distribution model selection.
- **Offline gates** (eval-only / ExpW / mixed-source) measure deployment-aligned stress under domain shift and label noise.
- **Deployment replay** captures temporal stability, probability-margin dynamics, and the interaction between model confidence and post-processing filters.

These metrics are **not interchangeable**: a high Stage-A teacher validation macro-F1 does not imply high mixed-source macro-F1, and a model that passes offline gates can still regress on deployment-facing replay.

#### 6.1.5 Limitations

- **Single-seed experiments:** all CE/KD/DKD comparisons are based on a single random seed. Without repeated runs, observed differences (e.g., CE vs KD macro-F1) cannot be statistically distinguished from seed noise. Given FYP GPU budget and timeline, we prioritised breadth of experiments (KD/DKD/NegL/LP-loss/domain shift) over multi-seed repeats; future work would trade some breadth for 2–3 seeds on the final configuration.
- **Protocol mismatch** is the primary risk for paper comparison: single-crop vs ten-crop, preprocessing pipeline, and label mapping can all shift accuracy materially.
- **Dataset packaging vs official splits:** folder-packaged datasets can differ from papers' official split definitions. This report uses official-split evaluation as the anchor where available.
- **Domain shift dominates deployment:** webcam lighting, motion blur, and face-crop jitter cause failures invisible on curated offline tests.
- **Temporal stabilisation changes the optimisation target:** raw vs smoothed metrics are both necessary; smoothing improves perceived stability while potentially hiding instantaneous errors.
- **Adaptation safety:** small-buffer fine-tuning can improve the target session while harming broad generalisation; KD/DKD can reduce probability margins under shift, amplifying flicker under EMA/hysteresis.

### 6.2 Analytical comparison with published results

In this FYP context, comparison is primarily **analytical**: the goal is to explain where and why performance differs rather than claim strict SOTA equivalence.

**Key constraint differences.** This project targets a deployable real-time FER pipeline (MobileNetV3-Large on CPU, multi-source noisy data, explicit domain shift focus) whereas most FER papers optimise offline benchmark objectives with heavy backbones (ResNet-50/ViT/Swin) on cleaner single-dataset protocols. A lower offline score is therefore both expected and acceptable when the trade-off is explicit.

**Why 1:1 comparison is inappropriate.** Direct numeric comparison requires matching: (a) dataset split/protocol, (b) label mapping (7-class vs 8-class), (c) metric definition (accuracy vs macro-F1, single-crop vs ten-crop), and (d) training recipe (epochs, resolution, optimiser, augmentation). A systematic checklist comparison reveals hidden recipe advantages in published SOTA numbers, e.g., FER2013 methods commonly use 40x40 centre crops with 300-epoch SGD, while this project uses 224x224 with AdamW for 30 epochs; some papers reportedly select best checkpoints by test-set performance.

**Where performance differs.** The system underperforms paper-reported SOTA mainly due to: model capacity trade-off (MobileNetV3 vs heavy backbones), dataset difference (466k multi-source noisy mixture vs single curated set), deployment constraint (stability/calibration objectives rarely reported in papers), explicit domain shift scope (webcam drift as first-class problem), and training recipe differences (hidden variables).

**What we optimise beyond papers.** This project additionally measures calibration (ECE/NLL with temperature scaling), stability (smoothing + hysteresis + jitter), and reproducibility (manifests + stored metrics). SOTA papers are used to contextualise gaps; a result is only labelled paper-comparable when the evaluation protocol is matched.

### 6.3 Ethical considerations

Facial expression recognition raises several ethical concerns that informed the design of this project.

**Privacy and consent.**  All live webcam data used during development were captured by the author in controlled settings with informed consent.  No webcam imagery is transmitted externally; all inference runs locally on the user's device.  Recorded sessions are stored only as per-frame CSV logs (predicted labels, confidence scores, and timestamps) — raw video frames are never persisted by default.

**Bias and fairness.**  The training corpora (FERPlus, AffectNet, RAF-DB, ExpW) are known to exhibit demographic imbalances in age, ethnicity, and gender.  This project does not claim demographic fairness; the domain-shift and calibration analyses in Sections 4.7–4.8 partially expose performance gaps across data sources, but a dedicated demographic audit was not conducted and is noted as future work.

**Cultural validity of emotion categories.**  The seven-class taxonomy used throughout this project (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) derives from Ekman's basic-emotions theory.  This framework has been critiqued for imposing Western-centric categories that may not generalise across cultures — for example, Barrett et al. [29] argue that facial configurations are not universally mapped to discrete emotion categories.  The system's outputs should therefore be interpreted as learned statistical associations within the training distribution rather than ground-truth emotion states.

**Institutional ethics.**  This project was conducted under the university's standard undergraduate project guidelines.  No formal ethics committee approval was required because no new human-subjects data were collected (all training data are from existing published datasets), and webcam testing involved only the author.

**Intended use.**  The system is developed as an academic prototype for emotion-aware HCI research.  It is not intended for surveillance, hiring decisions, or any high-stakes automated decision-making.  Any downstream deployment should undergo further bias testing and obtain appropriate institutional ethics approval.

## 7. Project Timeline

The following table summarises the project milestones across the Aug 2025 - Apr 2026 development period.

| Period | Phase | Key Deliverables |
| --- | --- | --- |
| Aug–Oct 2025 | Literature review & data prep | Paper study (KD, DKD, FER datasets); multi-source manifest (466 k rows); codebase setup |
| Nov 2025 | Teacher training | Stage-A training (RN18, B3, CNXT); ensemble selection; softlabel export |
| Dec 2025 | Student distillation | CE/KD/DKD student training; reliability-metrics pipeline; interim report submitted |
| Jan 2026 | Domain shift & NL/NegL | Domain-shift measurement loop; NL/NegL screening; ONNX export; demo repo published |
| Feb 2026 (Wk 1–2) | LP-loss & diagnostics | KD+LP screening; offline benchmark suite; per-source breakdown; FER2013 official-split eval |
| Feb 2026 (Wk 3–4) | Adaptation & audit | Self-Learning + NegL A/B (NR-1); BN-stat drift diagnosed; negative results consolidated |
| Mar 2026 | Final report | Report consolidation; structural improvements; timed demo KPI benchmarking |
| Apr 2026 | Evaluation & presentation | Final evaluation; presentation; project packaging |
## 8. Lessons Learned

This section distils the key engineering and methodological lessons that emerged during the project, which may be valuable for future work in deployed ML systems.

1. **Artifact-grounded workflows prevent silent metric drift.** By storing manifests, JSON reliability metrics, and checkpoints at every stage, results become auditable and reproducible. This discipline caught several instances where metric changes were caused by manifest composition shifts rather than model improvements.

2. **Manifest validation gates save significant debugging time.** Automated path/label validation on CSV manifests before training prevented multiple potential wasted GPU-hours from silent data corruption or path misalignment.

3. **Resume semantics must be explicitly verified.** During DKD resume training, a subtle bug where `total_epochs <= start_epoch` caused no-op (zero-step) training runs. Explicit epoch-count assertions are necessary for multi-stage training pipelines.

4. **Deployment-facing metrics cannot be inferred from offline metrics.** Offline macro-F1 does not predict real-time stability (flip-rate/jitter). The probability-margin dynamics that determine whether EMA/hysteresis filters produce stable predictions are invisible to standard classification metrics. This motivated the dual-gate evaluation protocol.

5. **Domain adaptation requires explicit safety controls.** Without an offline regression gate, small-buffer adaptation can silently degrade broad-distribution performance. The BatchNorm running-stat drift issue (Section 4.8) demonstrated that even "head-only" fine-tuning can unexpectedly modify feature statistics if BN layers remain in training mode.

6. **Preprocessing consistency is a hidden variable.** The CLAHE on/off mismatch between training and evaluation (Section 4.8) caused a false "regression" signal that was initially attributed to the adaptation method. This underscores the importance of storing preprocessing configuration alongside model checkpoints.
## 9. Conclusion and Next Steps

### 9.1 Conclusion

This project delivers a reproducible, end-to-end real-time FER pipeline spanning multi-source data cleaning, teacher training, ensemble distillation, and deployment-facing evaluation --- with artifact-backed evidence at every stage. The results address the four research questions posed in Section 1.1:

**RQ1 (Distillation effect).** KD/DKD consistently improve temperature-scaled calibration (TS ECE: 0.050 to 0.027) but do not surpass CE macro-F1 (0.742) in the primary evaluation. This demonstrates that calibration quality and decision-boundary quality are distinct optimisation targets that must be evaluated independently.

**RQ2 (Domain shift impact).** Domain shift causes severe and systematic performance degradation: teacher macro-F1 drops from 0.791 (in-distribution) to 0.393 (eval-only), with minority classes Fear and Disgust degrading to near-zero F1 under webcam conditions. This fragility persists across backbones, distillation methods, and preprocessing variants, identifying it as a structural representational challenge.

**RQ3 (Adaptation safety).** NL/NegL auxiliary losses, as currently tuned, do not provide consistent macro-F1 improvements and can destabilise training. The 2026-02-21 Self-Learning + NegL adaptation candidate passed the offline regression gate but regressed on same-session webcam replay (macro-F1: 0.525 to 0.467), directly demonstrating that offline non-regression is necessary but insufficient for deployment improvement claims.

**RQ4 (Evaluation protocol).** The evidence from NR-1 through NR-7 collectively supports the project's key engineering contribution: a **dual-gate evaluation protocol** requiring both broad-distribution offline non-regression and fixed-protocol deployment replay improvement before promoting any checkpoint. This provides a reusable evaluation discipline for safety-critical model updates in deployed FER systems.

In summary, this FYP **intentionally stops** at a safe, well-understood baseline rather than over-claiming fragile adaptation gains. The primary contribution is not a higher benchmark number but a **documented evaluation discipline** — dual-gate promotion, artifact-backed negative results, and protocol-aware comparison — that makes the gap between offline metrics and deployment reality explicit and measurable.

### 9.2 Future Work

Several avenues remain for extending this work:

1. **Deployment benchmarking:** run a timed demo session to report FPS, latency distribution, and jitter/flip-rate on the target CPU device.
2. **LP-loss tuning:** establish a stable KD baseline on the HQ-train pipeline and systematically vary LP-loss weight and embedding layer to identify configurations that improve cross-dataset macro-F1 without harming calibration.
3. **Domain adaptation refinement:** tighten the self-learning buffer construction (stable-only sampling, per-class caps, replay anchors) and complete the BN-only webcam gate to isolate BatchNorm running-stat drift as a failure mode.
4. **Protocol-matched evaluation:** for each paper comparison target, complete a gap checklist covering split, crop, preprocessing, resolution, label mapping, backbone capacity, and training settings to enable strict apples-to-apples reporting.
5. **Capacity scaling:** investigate larger student backbones (e.g., ResNet-50) to quantify the capacity-latency-accuracy trade-off against the current MobileNetV3-Large student.



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

[23] K. Wang, X. Peng, J. Yang, S. Lu, and Y. Qiao, "Suppressing uncertainties for large-scale facial expression recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2020, pp. 6897–6906.

[24] K. Wang, X. Peng, J. Yang, D. Meng, and Y. Qiao, "Region attention networks for pose and occlusion robust facial expression recognition," *IEEE Trans. Image Process.*, vol. 29, pp. 4057–4069, 2020.

[25] Z. Mao, Z. Xu, C. Lu, L. Liu, J. Yan, and Z. Liu, "POSTER V2: A simpler and stronger facial expression recognition network," arXiv preprint arXiv:2301.12149, 2023.

[26] Z. Wen, W. Lin, T. Wang, and G. Xu, "Distract your attention: Multi-head cross attention network for facial expression recognition," *Biomimetics*, vol. 8, no. 2, art. 199, 2023.

[27] Y. Zhang, C. Wang, X. Ling, and W. Deng, "Learn from all: Erasing attention consistency for noisy label facial expression recognition," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2022, pp. 418–434.

[28] Z. Zhao, Q. Liu, and F. Zhou, "Learning deep global multi-scale and local attention features for facial expression recognition in the wild," *IEEE Trans. Image Process.*, vol. 30, pp. 6544–6556, 2021.

[29] L. F. Barrett, R. Adolphs, S. Marsella, A. M. Martinez, and S. D. Pollak, "Emotional expressions reconsidered: Challenges to inferring emotion from human facial movements," *Psychol. Sci. Public Interest*, vol. 20, no. 1, pp. 1–68, 2019.

## 11. Appendix

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

Condensed pointer list:

- Offline benchmark suite index: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_index.json`
- Offline benchmark CSV: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
- Diagnostic write-up: `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`
- Per-source eval-only breakdown folder: `outputs/diagnostics/bad_datasets/source_breakdown__CE__eval_only__clahe_on/`
- Error-sample folder: `outputs/diagnostics/bad_datasets/`

### A.5 Paper comparison: evidence index and source notes

Evidence index:

| Evaluation target | Manifest / split | Metrics artifact(s) | Comparable? |
| --- | --- | --- | --- |
| FER2013 official (PublicTest/PrivateTest) | `Training_data/FER2013_official_from_csv/manifest__publictest.csv`, `.../manifest__privatetest.csv` | `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`; `outputs/evals/students/fer2013_official__*__*test__20260212__{singlecrop,tencrop}/reliabilitymetrics.json` | Partial |
| RAF-DB basic (student CE) | Offline-suite protocol | `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | Partial |
| AffectNet balanced (student CE) | Offline-suite protocol | `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` | No |
| ExpW gate (DKD variants) | `Training_data_cleaned/expw_full_manifest.csv` | `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md`; `outputs/evals/students/*/reliabilitymetrics.json` | No |
| Eval-only safety gate | `Training_data_cleaned/classification_manifest_eval_only.csv` | `outputs/evals/students/*__eval_only__test__*/reliabilitymetrics.json` | No |

Reference text sources:

- `outputs/paper_extract/AffectNet A Database for Facial Expression, Valence, and Arousal Computing in the Wild.txt`
- `outputs/paper_extract/Facial Emotion Recognition State of the Art Performance on FER2013.txt`
- `outputs/paper_extract/Expression Analysis Based on Face Regions in Read-world Conditions.txt`

### A.6 FYP requirements checklist (evidence audit)

The following checklist maps common FYP requirements (as discussed with the supervisor) to concrete evidence already stored in this project.

| Requirement | Status | Evidence in repo |
| --- | --- | --- |
| 1) Study deep learning methods for FER | Met | Paper study notes, background and method discussion in this report and interim reports. |
| 2) Study attention mechanisms + apply attention / knowledge distillation | Met (KD/DKD) / Met (attention as used in backbones) | KD/DKD implemented and evaluated with per-run reliability metrics. Attention mechanisms are included in used architectures (e.g., SE-style attention in MobileNetV3 / EfficientNet) and discussed in interim reports. |
| 3) Identify at least two datasets for experimentation | Met | Multi-source manifests covering FERPlus, AffectNet, RAF-DB, ExpW, and FER2013 (official splits). Six validated CSV manifests with integrity checks. |
| 4) Investigate and evaluate at least three methods | Met | Student: CE vs KD vs DKD comparisons with artifact-backed summaries (Sections 4.4, 4.12). Teacher backbones also provide additional method variety (RN18/B3/CNXT). |
| 5) Explore techniques to enhance FER performance (optional) | Met | Domain-shift loop + NL/NegL screening documented in Sections 4.5, 4.8, and Feb-2026 extensions (Section 3.9). |
| 6) Present at least one paper (seminar / paper study presentation) | Met | Presentation delivered (Dec 2025 interim presentation and Jan 2026 paper study presentation). |
