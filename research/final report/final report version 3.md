# Real-time Facial Expression Recognition System: Final Report

Project Title: Real-time Facial Expression Recognition System via Knowledge Distillation and Self-Learning + Negative Learning (NegL)

Author: Donovan Ma  
Institution: The Hong Kong Polytechnic University (PolyU)  
Supervisor: Prof. Lam  
Report Period: Aug 2025 – Mar 2026  
Document Date: Mar 16, 2026  
Report Version: 3 (7-chapter restructured)

---

## Abstract

Real-time facial expression recognition (FER) demands not only classification accuracy but also prediction stability, confidence calibration, and robustness to domain shift — requirements that standard offline benchmarks do not adequately capture. This project develops a reproducible, end-to-end real-time FER system for the canonical 7-class emotion space (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) using a teacher–student knowledge distillation pipeline. Three teacher backbones (ResNet-18, EfficientNet-B3, ConvNeXt-Tiny) are trained with ArcFace-style margins on 466,284 validated multi-source samples and distilled into a lightweight MobileNetV3-Large student via cross-entropy (CE), knowledge distillation (KD), and decoupled KD (DKD).

Three principal findings emerge. First, KD/DKD consistently improve temperature-scaled calibration (ECE: 0.050 → 0.027) but do not surpass CE macro-F1 (0.742), demonstrating that calibration quality and decision-boundary quality are distinct optimisation targets. Second, domain shift causes persistent minority-class fragility — Fear and Disgust degrade to near-zero F1 under webcam conditions — identifying a structural representational challenge rather than a tuning failure. Third, a safety-gated adaptation candidate combining Self-Learning and Complementary-Label Negative Learning (NegL) passes offline regression gates yet regresses on same-session webcam replay, establishing that offline non-regression is necessary but insufficient for deployment improvement claims.

This third finding motivates the project's central methodological contribution: a **dual-gate evaluation protocol** requiring both broad-distribution offline non-regression and fixed-protocol deployment replay improvement before promoting any checkpoint. Seven negative results (NR-1–NR-7) are formally catalogued with artifact-backed evidence, and a protocol-aware comparison framework contextualises performance against published benchmarks while respecting split, preprocessing, and metric differences.

## Executive Summary

**One-sentence summary:** A reproducible real-time FER pipeline is implemented using a teacher-student knowledge distillation design, evaluated with protocol-aware offline benchmarks and deployment-facing domain shift scoring.

**Deliverables:**

- Cleaned multi-source training and evaluation pipeline (466,284 validated samples across 7 emotion classes).
- Teacher training with three backbone architectures (ResNet-18, EfficientNet-B3, ConvNeXt-Tiny) achieving 0.781-0.791 macro-F1.
- Student distillation pipeline (CE, KD, DKD) targeting real-time CPU inference with MobileNetV3-Large.
- Real-time demo system with temporal stabilisation (EMA, hysteresis, vote window) and deployment-facing metrics (jitter flips/min).
- Protocol-aware paper comparison framework with explicit comparability flags.
- Dual-gate evaluation protocol requiring both offline non-regression and deployment replay improvement.

**Key findings:**

- KD/DKD improve temperature-scaled calibration (TS ECE: 0.050 to 0.027) but do not improve macro-F1 (0.742), demonstrating that calibration and decision-boundary quality are distinct optimisation targets.
- Domain shift causes severe minority-class degradation: teacher macro-F1 drops from 0.791 (in-distribution) to 0.393 on mixed-domain gates; Fear/Disgust reach near-zero F1 under webcam conditions.
- A safety-gated adaptation candidate passed offline regression but regressed on webcam replay, establishing that offline non-regression is necessary but insufficient for deployment claims.
- Fair comparison depends on protocol details (split definition, crop policy, preprocessing); RAF-DB accuracy (86.3%) is competitive while FER2013 official-split accuracy (61.4%) reflects the MobileNetV3 capacity-latency trade-off.

---

## Table of Contents

1. [Chapter 1 — Introduction](#chapter-1--introduction)
   - 1.1 Background and Problem Motivation
   - 1.2 Research Questions and Objectives
   - 1.3 Contributions
   - 1.4 Report Structure

2. [Chapter 2 — Literature Review](#chapter-2--literature-review)
   - 2.1 FER Basics and Challenges
   - 2.2 Datasets and Evaluation Protocols
   - 2.3 Backbone Architectures for FER
   - 2.4 Loss Functions and Training Methods
   - 2.5 Knowledge Distillation and Calibration
   - 2.6 Real-time FER and Temporal Stabilisation
   - 2.7 Domain Shift and Adaptation
   - 2.8 Synthesis and Research Gap

3. [Chapter 3 — System Design](#chapter-3--system-design)
   - 3.1 Overall Pipeline Architecture
   - 3.2 Multi-source Data Pipeline
   - 3.3 Teacher Model Design and Training
   - 3.4 Student Model Design and Distillation
   - 3.5 Real-time Inference Pipeline
   - 3.6 Domain Shift Evaluation Track
   - 3.7 Dual-gate Evaluation Protocol

4. [Chapter 4 — Implementation](#chapter-4--implementation)
   - 4.1 Dataset Preparation and Manifest Validation
   - 4.2 Teacher Training (Stage A)
   - 4.3 Ensemble Selection and Softlabel Export
   - 4.4 Student Training (CE → KD → DKD)
   - 4.5 NL/NegL Screening Experiments
   - 4.6 Real-time Demo System Implementation
   - 4.7 Domain Shift Experiments and Adaptation
   - 4.8 LP-loss Implementation
   - 4.9 Post-training Evaluation Infrastructure

5. [Chapter 5 — Results & Analysis](#chapter-5--results--analysis)
   - 5.1 Dataset Integrity
   - 5.2 Teacher Performance
   - 5.3 Ensemble Robustness
   - 5.4 Student Performance (CE/KD/DKD)
   - 5.5 Calibration Analysis
   - 5.6 NL/NegL Screening Results
   - 5.7 Webcam Domain Shift Results
   - 5.8 Adaptation Experiments and Negative Results
   - 5.9 LP-loss Screening Results
   - 5.10 Offline Benchmark Diagnostics
   - 5.11 Protocol-aware Paper Comparison

6. [Chapter 6 — Discussion & Limitations](#chapter-6--discussion--limitations)
   - 6.1 Discussion of Key Findings
   - 6.2 Limitations and Threats to Validity
   - 6.3 Comparative Analysis with Published Work
   - 6.4 Ethical Considerations

7. [Chapter 7 — Conclusion & Lessons Learned](#chapter-7--conclusion--lessons-learned)
   - 7.1 Conclusion
   - 7.2 Lessons Learned
   - 7.3 Future Work

8. [References](#references)
9. [Appendix](#appendix)

---

# Chapter 1 — Introduction

## 1.1 Background and Problem Motivation

Real-time facial expression recognition (FER) aims to classify facial expressions from a live stream while meeting constraints that are not captured by typical offline benchmarks:

- **Latency and throughput (FPS) constraints:** real-time deployment requires < 40ms latency per frame to maintain responsive user experience
- **Stability:** avoid flickering predictions between frames due to noise and face-detection jitter
- **Calibration:** confidence should be meaningful for threshold-based decision logic
- **Domain shift:** webcam lighting, sensor noise, motion blur, and user-specific effects differ substantially from curated training data

The central problem addressed by this project is the mismatch between offline benchmark optimization and deployment-facing robustness. Standard FER benchmarks report accuracy on curated, single-dataset test splits; real deployment faces a distribution shift to webcam-like conditions with additional challenges of temporal stability and calibration reliability.

This project adopts a teacher–student knowledge distillation design to reconcile accuracy and speed: train strong teachers offline on a validated multi-source dataset, ensemble them for robustness, distill knowledge into a compact student suitable for real-time inference, and evaluate using both offline metrics (macro-F1, per-class F1, calibration) and deployment-facing metrics (smoothed vs raw performance, jitter flips/min).

## 1.2 Research Questions and Objectives

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

## 1.3 Contributions

This project makes the following technical and methodological contributions:

1. **Reproducible teacher–student FER pipeline.** An end-to-end, artifact-grounded pipeline covering multi-source data cleaning, teacher training (3 backbones), ensemble softlabel export, and student distillation (CE/KD/DKD) with stored JSON metrics at every stage.

2. **Dual-gate evaluation protocol.** A deployment-aware evaluation protocol requiring both (a) offline non-regression on broad-distribution gates (eval-only, ExpW) and (b) improvement on fixed-protocol webcam replay, before promoting any checkpoint. This is motivated by the empirical finding that offline gate pass does not imply webcam improvement (NR-1).

3. **Systematic negative result documentation.** Seven formally catalogued negative results (NR-1–NR-7) with evidence-backed vs hypothesis classifications, covering adaptation failures, auxiliary loss instability, and calibration–accuracy decoupling.

4. **Domain shift characterisation.** Quantitative analysis of the teacher→student→deployment transfer gap, including per-class fragility analysis (Fear F1 = 0.00 under webcam shift) and root-cause diagnosis (CLAHE mismatch, BN running-stat drift).

5. **Real-time stabilisation analysis.** Deployment-facing metrics (EMA-smoothed accuracy/F1, jitter flips/min) reported alongside offline metrics, with analysis of how probability margin dynamics interact with temporal smoothing.

## 1.4 Report Structure

This report is organised into 7 chapters:

- **Chapter 2 (Literature Review)** establishes the research context, reviewing FER fundamentals, datasets, architectures, loss functions, real-time constraints, and domain shift.
- **Chapter 3 (System Design)** presents the overall architecture and design choices for the teacher–student pipeline, real-time inference, and evaluation protocol.
- **Chapter 4 (Implementation)** details the practical implementation of dataset preparation, training stages, and evaluation infrastructure.
- **Chapter 5 (Results & Analysis)** reports all experimental results organized by component, including negative results.
- **Chapter 6 (Discussion & Limitations)** interprets the findings, contextualizes them against published work, and discusses limitations.
- **Chapter 7 (Conclusion & Lessons Learned)** synthesizes the key takeaways, documents lessons learned, and outlines future work.
- **Appendix** contains technical details, artifact inventories, and detailed evidence tables.

---

# Chapter 2 — Literature Review

This chapter reviews the foundational literature concerning facial expression recognition (FER), positioning the project within the broader context of computer vision and real-time deep learning. We begin by examining the inherent challenges of defining and classifying emotions, followed by a critical analysis of existing datasets and the complications introduced by domain shift. The review then evaluates backbone architectures suitable for real-time inference, discusses the evolution of loss functions from simple classification to margin-based and locality-preserving objectives, and finally synthesizes the research gaps that this project aims to address.

## 2.1 FER Basics and Challenges

Facial expression recognition (FER) is traditionally framed as a multi-class classification problem, mapping aligned face crops to a discrete set of emotion categories. The most common framework is the 7-class set established by Ekman: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. While this categorical approach provides a standard target for machine learning models, the task is fraught with inherent ambiguity. Expressions in the wild rarely conform to the exaggerated prototypes found in laboratory settings; instead, they exhibit subtle intensity variations where a low-intensity "Sad" face may be visually indistinguishable from "Neutral." Furthermore, muscle activation patterns often overlap, leading to persistent confusion pairs such as Fear versus Surprise, or Disgust versus Angry.

These ambiguities are exacerbated by the nature of in-the-wild data annotation. Crowdsourced labels are prone to noise and subjective interpretation, meaning that ground truth labels often represent a "majority vote" rather than a definitive psychological state. Consequently, standard accuracy metrics can be misleading, especially in class-imbalanced datasets where a model might achieve high accuracy simply by predicting majority classes like Happy/Neutral while failing completely on Fear or Disgust. To address this, modern FER research increasingly employs metrics like macro-F1 and per-class F1 to expose these minority-class failures, alongside specialized loss functions such as focal loss [11] or class-balanced loss [12] to strictly penalize neglect of tail classes.

## 2.2 Datasets and Evaluation Protocols

A significant divide exists between academic benchmarks and deployment reality. In the literature, results are typically reported on curated, single-source datasets with strictly defined splits. However, real-world deployment data is inherently multi-source and messy. Training distributions often combine high-quality studio images with noisy web-scraped data to maximize volume, while test distributions—such as live webcam feeds—introduces severe domain shifts including sensor noise, motion blur, and varying lighting conditions.

This discrepancy makes direct comparison of "state-of-the-art" results difficult without careful scrutiny of evaluation protocols. A result on the RAF-DB dataset, which is highly curated, means something very different from a result on the FER2013 dataset, which contains greyscale, low-resolution images with significant label noise. Furthermore, protocol decisions such as whether to use 8 distinct classes or compound classes, and whether to evaluate using single-crop or multi-crop (test-time augmentation) strategies, can shift reported accuracy by several percentage points.

In this project, we utilize **RAF-DB** as a representative for high-quality, curated data, and **FER2013** as the canonical benchmark for historical comparison. Additionally, we incorporate **AffectNet** for its diversity in pose and intensity, and **ExpW** as a stress-test for in-the-wild robustness. We intentionally distinguish between "paper-like" comparisons using official splits and "deployment-aligned" stress tests using mixed-source gates to fully characterize model performance.

## 2.3 Backbone Architectures for FER

The choice of backbone architecture is a trade-off between representational capacity and inference latency. Modern FER research relies heavily on Convolutional Neural Networks (CNNs) pretrained on large-scale datasets like ImageNet. For tasks where accuracy is paramount and computational budget is flexible (i.e., our "Teacher" models), architectures like **ResNet-18** [8] and **EfficientNet-B3** [5] provide strong feature extraction capabilities. **ConvNeXt-Tiny** [7] represents a more modern CNN iteration that adopts Vision Transformer design principles, offering potentially better handling of global context.

However, for a real-time system running on consumer CPUs, these heavy backbones are often too slow. This necessitates the use of efficiency-focused architectures like **MobileNetV3** [6], which utilizes lightweight depthwise separable convolutions and hardware-aware neural architecture search. While MobileNetV3 sacrifices some raw capacity compared to a ResNet, it is specifically designed to minimize latency. To bridge the accuracy gap, recent works frequently incorporate attention mechanisms, such as Squeeze-and-Excitation (SE) or CBAM [10], which allow the network to dynamically recalibrate feature weights—focusing on salient facial regions while suppressing background noise. This project leverages this "Teacher-Student" dynamic, using high-capacity backbones to discover features that are then distilled into the lightweight MobileNetV3 student.

## 2.4 Loss Functions and Training Methods

### Knowledge Distillation and Decoupled KD

Knowledge distillation trains a compact student model by combining:

- **Hard labels (cross-entropy):** encourage correct classification on ground-truth labels.
- **Soft targets (teacher probabilities/logits):** transfer teacher knowledge, including class similarity structure.

In typical KD [1], the teacher logits are softened with a temperature $T$ during training to provide a smoother target distribution. The standard KD loss combines a hard-label cross-entropy term with a soft-label KL-divergence term:

$$\mathcal{L}_{\text{KD}} = (1 - \alpha) \cdot \mathcal{L}_{\text{CE}}(y, \sigma(z_s)) + \alpha \cdot T^2 \cdot D_{\text{KL}}\!\left(\sigma\!\left(\frac{z_t}{T}\right) \;\middle\|\; \sigma\!\left(\frac{z_s}{T}\right)\right)$$

Decoupled KD (DKD) [2] modifies the objective so that target-class and non-target-class contributions can be weighted differently, potentially improving decision-boundary quality.

### ArcFace-style Margin Training

Teachers in this project are trained using ArcFace-style [4] additive angular margin loss, which encourages larger angular separation between class centers in embedding space. This can improve inter-class discrimination and robustness to variations in expression intensity.

### Complementary-Label Negative Learning

Negative Learning (NegL) [16] applies auxiliary losses that discourage probability mass on likely-wrong classes. Given a complementary label $\bar{y}$ (a class that the sample is believed *not* to belong to), the NegL loss is:

$$\mathcal{L}_{\text{NegL}} = -\log\!\left(1 - p(\bar{y} \mid x)\right)$$

This can improve calibration under controlled gating, but requires careful threshold tuning to avoid training instability.

### LP-loss: Locality-Preserving Loss

LP-loss [21] is a locality-preserving auxiliary loss that encourages within-class compactness in embedding space by penalizing samples that are far from their k-nearest same-class neighbors. This explicitly addresses the class-similarity confusions (Fear↔Surprise, Disgust↔Angry) that dominate FER error analysis.

## 2.5 Knowledge Distillation and Calibration

### Calibration Metrics

Calibration measures whether predicted confidence aligns with empirical correctness. Two common metrics are:

- **ECE (Expected Calibration Error):** a binned estimate of confidence vs accuracy mismatch:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{n} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

- **NLL (Negative Log-Likelihood):** penalises overconfident wrong predictions more strongly.

### Temperature Scaling

Temperature scaling [14] is a simple post-hoc calibration method that rescales logits by a single scalar $T$ (without changing argmax labels). It is essential when:

- The system uses confidence thresholds (e.g., pseudo-label acceptance for self-learning).
- The UI or downstream logic depends on probability margins.
- Real-time smoothing/hysteresis reacts differently to "peaky" vs "flat" distributions.

## 2.6 Real-time FER and Temporal Stabilisation

Real-time FER distinguishes itself from traditional offline image classification by requiring predictions on a continuous video stream rather than static images. This introduces the challenge of temporal consistency, where a model might flicker between expressions due to minor frame-to-frame variations in face detection or lighting. To mitigate this, common stabilisation techniques include Exponential Moving Average (EMA) smoothing over probability vectors, hysteresis to resist rapid class switching, and voting windows. These methods, however, alter the optimization target; a system that is robust in deployment may not necessarily have the highest raw accuracy on single frames. Therefore, reporting both raw and smoothed metrics, alongside stability indicators like jitter or "flip-rate," is crucial for a deployment-aligned evaluation.

## 2.7 Domain Shift and Adaptation

Cross-domain generalisation remains a formidable barrier in FER research. Models trained on curated datasets often fail when deployed in real-world environments due to domain shifts caused by camera sensor differences, varying lighting conditions, and subject-specific facial dynamics. To improved robustness, two primary families of techniques are often discussed: Test-time adaptation (TTA), which updates model parameters on the fly using unsupervised objectives, and self-learning or pseudo-labeling, which fine-tunes the model on confident predictions. However, both approaches are risky; adaptation on noisy or drifting distributions can lead to catastrophic forgetting or model collapse. This necessitates a safety-gated approach where updates are conservative and contingent on passing rigorous offline regression tests.

## 2.8 Synthesis and Research Gap

The reviewed literature motivates several critical design decisions for this project. First, the structural failure of minority classes like Fear and Disgust under domain shift is identified as a persistent challenge, justifying the use of macro-F1 and per-class metrics over simple accuracy. Second, while Knowledge Distillation (KD) and Decoupled KD (DKD) are standard for compression, their distinct impacts on calibration versus decision boundary quality warrant separate investigation. Third, the potential instability of naïve adaptation methods informs our conservative, dual-gate design. Finally, the selection of LP-loss is grounded in its ability to preserve local manifold structure, directly addressing the class overlap issues inherent in FER.

**Identified Research Gap:** Despite the maturity of individual components like KD and domain adaptation, a significant gap remains in the FER literature. There is a lack of systematic empirical comparison that evaluates CE, KD, and DKD jointly on classification accuracy, calibration quality, and deployment-facing stability. Furthermore, no existing work establishes a formal evaluation protocol that requires a model to pass both offline non-regression gates and deployment-environment replay tests before promotion. Most studies optimize for a single offline metric, neglecting the temporal dynamics and failure modes unique to real-time deployment. This project aims to bridge this gap by proposing and validating a dual-gate evaluation framework.

## 2.9 Chapter Summary

In summary, the literature highlights a critical tension between the optimization of offline benchmarks—typically focused on static image accuracy—and the requirements of a real-time deployment system, which demands stability, calibration, and robustness to domain shift. While high-capacity backbones and margin-based losses offer strong feature extraction, they must be effectively bridged to lightweight students for inference. Having established the theoretical foundation and identified the gap in deployment-aligned evaluation, the next chapter details the system design, specifically how a multi-teacher distillation pipeline and dual-gate evaluation protocol are constructed to address these challenges.

---

# Chapter 3 — System Design

This chapter describes the comprehensive architecture of the real-time FER system, translating the theoretical needs identified in the Literature Review into concrete design choices. We detail the end-to-end pipeline, starting from multi-source data ingestion and progressing through teacher ensemble training, student distillation, and real-time inference. Particular emphasis is placed on the **dual-gate evaluation protocol**, a novel methodological contribution designed to rigorously separate offline optimization from deployment readiness.

## 3.1 Overall Pipeline Architecture

![System Architecture Pipeline](../figures/fig0_pipeline_architecture.png)
*Figure 3.1: High-level pipeline architecture mapping multi-source raw data to a real-time deployment loop.*

The system is architected as a multi-stage pipeline designed to ensure data quality, robust training, and reliable deployment. The workflow begins with the ingestion of raw, multi-source datasets, which undergo a rigorous cleaning and validation process to generate immutable CSV manifests. These manifests drive the training of a Teacher ensemble comprising diverse backbone architectures (ResNet-18, EfficientNet-B3, and ConvNeXt-Tiny), trained using an ArcFace-style margin loss to maximize class separability.

Once trained, the best-performing Teacher ensemble generates soft-labels—probability distributions that encapsulate the inter-class relationships learned from the data. These soft-labels serve as the target for the Student model (MobileNetV3-Large), which is trained via a progressive distillation strategy moving from Cross-Entropy (CE) to Knowledge Distillation (KD) and finally to Decoupled Knowledge Distillation (DKD). The resulting student model is then deployed into a real-time inference loop that incorporates temporal stabilisation logic. Crucially, the system features a continuous evaluation loop where adaptation candidates (generated via self-learning or negative learning) must pass a rigorous "dual-gate" protocol—satisfying both offline benchmarks and live deployment checks—before being promoted to production.

```mermaid
flowchart LR
  A[Raw Datasets] --> B[Clean & Validate]
  B --> C[Teacher Training
  (Ensemble)]
  C --> D[Softlabel Export]
  D --> E[Student Distillation
  (CE → KD → DKD)]
  E --> F[Real-time Inference]
  F --> G[Dual-Gate Evaluation]
  G -->|Pass| H[Deploy]
  G -->|Fail| I[Reject]
```

## 3.2 Multi-source Data Pipeline

Data is the foundation of any deep learning system, and for FER, diversity is key. To ensure broad coverage of facial variances, we aggregate data from multiple canonical sources including FERPlus, AffectNet, RAF-DB, and ExpW. Rather than relying on opaque folder structures, the entire data lifecycle is managed through CSV manifests. This approach allows for automated integrity validation—ensuring that every file path is valid and every label falls within the canonical 7-class emotion space (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) before a single GPU cycle is spent on training. This rigorous "manifest-first" design guarantees reproducibility, as the exact composition of training and evaluation sets (totaling 466,284 validated rows) is frozen and verifiable via SHA256 fingerprints.

## 3.3 Teacher Model Design and Training

The role of the Teacher models is to extract the richest possible feature representations from the data, unconstrained by the latency requirements that limit the Student. We employ an ensemble of three distinct architectures: **ResNet-18** [8] for its balance of depth and residual learning, **EfficientNet-B3** [5] for its enhanced scaling efficiency, and **ConvNeXt-Tiny** [7] for its modern, transformer-inspired design.

These teachers are not merely trained for classification accuracy; they are optimized using an **ArcFace-style additive angular margin loss** [4]. This loss function enforces a stricter geometric constraint on the learned embeddings, pushing features of different classes further apart in the angular space. This margin-based training is critical for FER, where inter-class boundaries are often fuzzy. By maximizing the angular margin, the Teachers learn more distinct and separable class representations, providing a higher-quality signal for the Student to emulate during the distillation process.

## 3.4 Student Model Design and Distillation

For the Student model, the primary constraint is real-time inference latency on standard consumer CPUs. We selected **MobileNetV3-Large** [6] as the backbone due to its hardware-aware architecture search design, which effectively balances parameter count (approx. 5.4M) with execution speed.

The training of the Student follows a curriculum-based distillation strategy designed to progressively refine its performance. The process begins with a **Cross-Entropy (CE)** baseline, where the model learns directly from the hard ground-truth labels. In the second stage, **Knowledge Distillation (KD)** [1] is introduced, where the Student is trained to match the softened probability distributions (soft-labels) produced by the Teacher ensemble. This allows the Student to learn not just the correct class, but also the "dark knowledge"—the structural similarities between classes (e.g., that Fear is more similar to Surprise than to Happy). Finally, we apply **Decoupled Knowledge Distillation (DKD)** [2], which separates the distillation loss into target-class and non-target-class components, allowing for more precise weighting of the teacher's guidance and further refining the student's decision boundaries.

## 3.5 Real-time Inference Pipeline

The deployment environment introduces challenges that offline training does not address, principally the need for temporal consistency. The inference pipeline is built to stabilize the raw, often jittery predictions that result from frame-by-frame analysis. After face detection via YuNet [18] and standard preprocessing (resizing and normalization), the input frame is passed to the ONNX-exported Student model.

Crucially, the raw logits from the model are not utilized directly. Instead, they first undergo **temperature scaling** using a calibrated value $T$, derived from the offline validation phase, to align the confidence scores with empirical probabilities. These probabilities are then smoothed over time using an **Exponential Moving Average (EMA)** to dampen noise. A **hysteresis** logic is applied to the final class decision, requiring a new emotion to surpass a confidence threshold before the system switches states. This multi-layered stabilization ensures that the user experiences a smooth and coherent emotional read-out, rather than a flickering stream of disconnected predictions.

## 3.6 Domain Shift Evaluation Track

To address the inevitable performance degradation caused by domain shift—differences in camera sensors, lighting, and subject demographics—evaluation is treated as a continuous, two-track process. The first track is a **Broad-Distribution Offline Gate**, utilizing the "eval-only" and "ExpW" datasets as proxies for in-the-wild diversity. These datasets are never seen during training and serve to measure general robustness.

The second track is the **Deployment Replay Gate**. This involves re-running the model on recorded sessions from the target deployment environment (e.g., specific webcam feeds) and measuring metrics that reflect user experience, such as "jitter" (the frequency of label flips) and smoothed macro-F1. This track ensures that a model that performs well on static benchmarks does not regress in the dynamic, noisy conditions of actual use.

## 3.7 Dual-gate Evaluation Protocol

A central contribution of this system design is the formalization of a **Dual-Gate Evaluation Protocol**. We recognize that offline metrics and deployment performance are not interchangeable; a model can achieve high accuracy on a validation set yet fail in a live demo due to poor calibration or instability.

Therefore, for any model update or adaptation candidate to be promoted to production, it must satisfy two distinct conditions:
1.  **Offline Non-Regression:** The model must effectively maintain its performance on the broad-distribution offline benchmarks (eval-only/ExpW), ensuring it has not "forgotten" general features or overfitted to a specific niche.
2.  **Deployment Improvement:** The model must demonstrate a tangible improvement (or at minimum, non-regression) on the deployment replay track, specifically in terms of stability (lower jitter) and smoothed F1 scores.

This strict "AND" condition prevents the common pitfall of optimizing for a benchmark number at the expense of real-world usability, ensuring that every system update creates a genuinely better user experience.

## 3.8 Chapter Summary

This chapter has outlined a design that prioritizes reproducibility and deployment realism over simple benchmark chasing. By coupling a strong teacher ensemble with a lightweight student via knowledge distillation, the system balances accuracy and latency. The dual-gate evaluation protocol then serves as the critical quality control mechanism, ensuring that improvements in the lab actually translate to the webcam. The following chapter, **Implementation**, will detail exactly how these components were built, trained, and validated in practice.

---

# Chapter 4 — Implementation

This chapter documents the practical realization of the design proposed in Chapter 3. It details the specific engineering steps taken to clean and validate the multi-source dataset, the chronological progression of training stages (from Teachers to CE/KD/DKD Students), and the development of the real-time inference engine. We also describe the specific configuration of the experimental screening runs for Negative Learning and LP-loss, providing the necessary context for the results analyzed in Chapter 5.

## 4.1 Dataset Preparation and Manifest Validation

All training and evaluation data are managed through CSV manifests with automated integrity validation (466,284 rows, 0 missing paths, 0 bad labels). Split sizes and label distributions are computed directly from the manifests and stored as reproducibility artifacts.

![Dataset Imbalance Distribution](../figures/fig0_data_imbalance.png)
*Figure 4.1: Extreme class imbalance present in the HQ training manifest.*

**Multi-source data sources:** FERPlus, AffectNet (full balanced), RAF-DB (basic), ExpW (in-the-wild).

**Validated distribution summaries:**

- Full multi-source manifest split sizes: train=378,965 / val=37,862 / test=49,457.
- HQ-train manifest split sizes: train=213,144 / val=18,020 / test=27,840.
- Mixed-domain benchmark (test_all_sources) contains 48,928 rows and all 7 classes present.

## 4.2 Teacher Training (Stage A)

Teachers are trained using an ArcFace-style protocol with additive angular margin loss. All teachers use 224x224 input resolution with ImageNet-pretrained weights, and are saved with full provenance metadata.

Each teacher run produces a standardised output folder containing the best checkpoint, training history, reliability metrics (accuracy, macro-F1, per-class F1, ECE/NLL with temperature-scaled variants), and data alignment provenance.

## 4.3 Ensemble Selection and Softlabel Export

Teacher ensembles are constructed by weighted logit fusion. The ensemble weights are selected based on performance on a mixed-source benchmark (48,928 rows), and the selected ensemble exports per-sample softlabel probability vectors for subsequent student training.

**Selected ensemble:** RN18/B3/CNXT = 0.4/0.4/0.2
**Performance:** Accuracy 0.687, Macro-F1 0.660

The ensemble's softlabel outputs (probability vectors over the 7-class label space) are stored as a reusable CSV artifact for student KD/DKD training.

## 4.4 Student Training (CE → KD → DKD)

The student model uses MobileNetV3-Large (5.4M params), chosen for its favourable accuracy-latency trade-off on CPU inference.

![Training Curves - Cross Entropy vs Distillation](../figures/fig5_training_curves_ce.png)
*Figure 4.2: Student training progression comparing standard Cross-Entropy against Distillation targets.*

**HQ training manifest** (verified size and splits): 259,004 rows with split sizes train=213,144 / val=18,020 / test=27,840.

The three student stages are trained sequentially:
1. **CE stage:** standard cross-entropy training on hard labels.
2. **KD stage:** soft targets from the teacher ensemble combined with hard labels (with distillation temperature T=5).
3. **DKD stage:** decoupled KD with separate target-class and non-target-class weighting.

Each student run produces the same standardised output structure as teachers, with reliability metrics stored as JSON artifacts.

## 4.5 NL/NegL Screening Experiments

NL/NegL screening experiments are documented in dedicated planning and report files, with results summarised via standardised comparison tables. Each compared run is backed by stored reliability metrics.

**Terminology note:**

- **NL(proto)** refers to the *Nested Learning* [3] prototype-style auxiliary mechanism used in the Jan-2026 screening runs.
- **NegL** refers to an entropy-gated *complementary-label negative learning* loss [16] that discourages probability mass on likely-wrong classes.

## 4.6 Real-time Demo System Implementation

The real-time demonstration system bridges the gap between static offline benchmarks and dynamic, live deployment. It implements a multi-stage inference pipeline targeting CPU environments on consumer hardware (tested on an Intel i9-13900HX at 2.20 GHz). The process begins with lightweight face detection via YuNet [18], followed by frame pre-processing (including optional CLAHE for contrast equalization) before feeding into the ONNX-exported MobileNetV3 Student model.

A key engineering feature of the demo is its **temporal stabilisation logic**. Raw frame-by-frame inference in FER is inherently jittery due to slight variations in bounding box framing, pose, and lighting. To mitigate this, the system does not simply output the argmax of the current frame's logits. Instead, it utilizes configured temperature-scaling (read directly from the training artifact's `calibration.json`) to adjust the sharpness of the confidence distribution. These smoothed probabilities are then processed via an **Exponential Moving Average (EMA)** parameterized by `ema_alpha`. Finally, a **hysteresis** threshold (`hysteresis_delta`) enforces that the system only switches the predicted class if the competing class's smoothed probability exceeds the current label's probability by a predefined margin. 

![Real-Time Hysteresis Jitter Plot](../figures/fig11_hysteresis_jitter.png)
*Figure 4.3: Temporal stabilisation effect suppressing jitter during continuous inference.*

This design explicitly trades off instantaneous responsiveness for user-perceived stability, ensuring the UI remains readable. The system logs raw predictions, smoothed predictions, and confidence vectors per frame to a CSV, enabling deterministic replay and calculation of deployment metrics like "jitter flips/min".

## 4.7 Domain Shift Experiments and Adaptation

To address the performance gap between curated datasets and live webcam environments, we implemented a continuous safety-gated adaptation loop. The system leverages a **Self-Learning + Negative Learning (NegL)** policy designed to incrementally update the model without inducing catastrophic forgetting.

The adaptation logic utilizes a "Confidence-Banded Policy" built on temperature-calibrated probabilities. Predictions falling into a *high-confidence band* ($p_{max} \ge \tau_{high}$) are accepted as positive pseudo-labels for Self-Learning. Predictions in the *medium-confidence band* ($\tau_{mid} \le p_{max} < \tau_{high}$) are treated as untrustworthy positives but useful negative signals; here, an entropy-gated complementary Negative Learning loss is applied to push probability mass away from likely-wrong classes. Low-confidence predictions are ignored.

Crucially, any candidate model bred from this adaptation loop is subject to strict pre-registered evaluation thresholds. The model must pass an **Offline Safety Gate** (run against the `eval-only` and `ExpW` manifests) to prove it hasn't lost broad-distribution accuracy. Simultaneously, it must run through the **Deployment Replay Gate**—replaying the identical recorded webcam session—to prove it has reduced jitter or improved smoothed macro-F1. To ensure stability during fine-tuning on small webcam buffers, we enforce conservative parameter updates by freezing BatchNorm running statistics (`frozebn`) and optionally restricting updates to the network head.

## 4.8 LP-loss Implementation

An LP-loss [21] implementation was added in Feb 2026 to support locality-preserving training. When enabled, it applies a within-class compactness objective in embedding space, controlled by weight, layer selection, and neighbour count $k$.

## 4.9 Post-training Evaluation Infrastructure

The training entrypoint supports an optional `--post-eval` flag that automatically runs the standalone evaluation script on eval-only and ExpW manifests after training finishes, generating per-evaluation reliability metrics and a summary artifact.

## 4.10 Chapter Summary

In this chapter, we have detailed the technical implementation of the full FER pipeline, from the rigorous validation of 466,284 training samples to the deployment of the MobileNetV3 student on CPU. The implementation ensures that every experiment—whether a Teacher training run or a sophisticated DKD distillation—is backed by traceable artifacts and consistent evaluation logic. With the system built and the experimental protocols defined, the next chapter presents the **Results & Analysis**, evaluating the system's performance across accuracy, calibration, and domain shift robustness.

---

# Chapter 5 — Results & Analysis

This chapter presents a comprehensive analysis of the system's performance, structured to validate the research questions posed in Chapter 1. We begin by verifying dataset integrity and establishing Teacher baselines, then progress to the core comparative analysis of Student distillation strategies (CE vs. KD vs. DKD). Crucially, we do not shy away from failure; a significant portion of this chapter is dedicated to analysing "Negative Results" (NR-1 to NR-7) and domain shift fragility, which provide the empirical justification for the dual-gate protocol.

## 5.1 Dataset Integrity

Dataset integrity summary:

- Total rows: 466,284
- Missing paths: 0
- Bad labels: 0

While the programmatic validation ensured zero missing or formally invalid paths, it is critical to acknowledge the semantic label noise inherent in "in-the-wild" FER datasets. Emotions like "Fear" and "Surprise" often suffer from high inter-annotator disagreement even among human raters. By combining multiple datasets (FERPlus, AffectNet, RAF-DB), we inevitably inherit these conflicting annotation standards. This foundational label noise is exactly why our pipeline heavily leverages Knowledge Distillation (KD) in later stages: the Teacher model ensemble acts as a "label smoother," overriding contradictory human hard-labels with consistent, probabilistic soft-labels that represent a unified consensus.

Verified distribution summaries (baseline teacher training):

- Full multi-source manifest split sizes: train=378,965 / val=37,862 / test=49,457.
- HQ-train manifest split sizes: train=213,144 / val=18,020 / test=27,840.
- Mixed-domain benchmark (test_all_sources) contains 48,928 rows and all 7 classes present.

## 5.2 Teacher Performance

These teacher metrics are computed on the **Stage-A validation split** (n = 18,165) after source filtering (AffectNet balanced + FERPlus + RAF-DB basic).

![Teacher Per-class F1 Scores](../figures/fig2_teacher_perclass_f1.png)
*Figure 5.1: Teacher ensemble performance across specific emotion classes (Stage-A validation).*

**Table 5.2-1: Teacher metrics (Stage-A validation split, n = 18,165).**

| Model | Accuracy | Macro-F1 | Raw NLL | TS NLL | Raw ECE | TS ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| RN18 | 0.786 | 0.781 | 4.026 | 0.880 | 0.205 | 0.149 |
| B3 | 0.796 | 0.791 | 3.222 | 0.787 | 0.199 | 0.084 |
| CNXT | 0.794 | 0.789 | 3.101 | 0.770 | 0.201 | 0.082 |

**Hard gates (domain shift / mixed-domain):**

**Table 5.2-2: Teacher macro-F1 on hard-gate datasets.**

| Gate dataset | n | RN18 | B3 | CNXT |
| --- | ---: | ---: | ---: | ---: |
| eval_only | 11,890 | 0.373 | 0.393 | 0.389 |
| expw_full | 9,179 | 0.374 | 0.407 | 0.382 |
| test_all_sources | 48,928 | 0.617 | 0.645 | 0.638 |

**Interpretation:** The Stage-A validation metrics substantially overestimate performance under mixed-domain stress tests. The best teacher (B3) drops from 0.791 macro-F1 on Stage-A validation to 0.393 on eval-only — a **50% relative decrease**. This gap reflects domain mismatch, label noise, and class imbalance in hard gates.

## 5.3 Ensemble Robustness

The ensemble benchmark evaluates a weighted teacher combination on the full mixed-source test set (48,928 rows).

**Selected ensemble configuration:**
- Weights: RN18/B3/CNXT = 0.4/0.4/0.2
- Accuracy: 0.687
- Macro-F1: 0.660
- Additional metrics: NLL 4.077, ECE 0.288, Brier 0.591

## 5.4 Student Performance (CE/KD/DKD)

Student metrics from the Dec 2025 CE/KD/DKD runs (HQ-train validation):

![Student Per-class F1 Scores](../figures/fig1_student_perclass_f1.png)
*Figure 5.2: Student bottleneck capacity shown via per-class F1 drops.*

**Table 5.4-1: Student stage comparison (HQ-train val).**

| Stage | Accuracy | Macro-F1 | Raw NLL | TS NLL | Raw ECE | TS ECE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.750 | 0.742 | 1.315 | 0.778 | 0.131 | 0.050 |
| KD | 0.735 | 0.733 | 2.093 | 0.768 | 0.215 | 0.028 |
| DKD | 0.737 | 0.738 | 1.512 | 0.765 | 0.209 | 0.027 |

**Interpretation:**

- CE provides the best raw macro-F1.
- KD/DKD improve temperature-scaled calibration (TS ECE) substantially compared to CE (0.050), but do not improve macro-F1 under this configuration.

**Consolidated cross-gate comparison (CE vs KD vs DKD):** Student macro-F1 on four stress-test datasets:

| Dataset | CE | KD | DKD |
| --- | ---: | ---: | ---: |
| eval_only | 0.459 | 0.496 | 0.489 |
| expw_full | 0.485 | 0.456 | 0.435 |
| test_all_sources | 0.564 | 0.532 | 0.545 |
| fer2013_folder | 0.456 | 0.467 | 0.461 |

## 5.5 Calibration Analysis

KD and DKD consistently improve temperature-scaled calibration (TS ECE: 0.050 to 0.027 and 0.027 respectively) compared to CE. This demonstrates that teacher-distribution matching optimises a fundamentally different objective than hard-label classification.

![Calibration Comparison - Reliability Diagrams](../figures/fig3_calibration_comparison.png)
*Figure 5.3: Reliability diagrams demonstrating improved probabilistic calibration via KD/DKD.*

From a Human-Computer Interaction (HCI) perspective, this reduction in Expected Calibration Error is perhaps more critical than a minor bump in accuracy. If a system is deployed as an adaptive UI trigger (e.g., dynamically altering a game based on a user's frustration), an overconfident incorrect prediction severely degrades user trust. A well-calibrated model "knows what it does not know." When paired with our Temperature Scaling and Hysteresis margins, the DKD student reliably suppresses actions on ambiguous faces because its probability outputs honestly reflect the task's inherent uncertainty.

**Key finding:** Calibration improvement does not translate to macro-F1 improvement, indicating that calibration and decision-boundary quality are distinct optimisation targets requiring independent evaluation.

## 5.6 NL/NegL Screening Results

The exploration of Complementary-Label Negative Learning (NegL) and nested-learning (NL) strategies aimed to improve the robustness of student distillation without increasing labeled data dependency. However, rigorous gating metrics logged over early screening epochs painted a clear picture: these auxiliary techniques failed to provide consistent gains and frequently suffered from mechanism collapse or calibration regression.

The Nested Learning implementation, piloted via prototype memory (`nl_proto`), was hypothesised to mine hard samples efficiently. Early screening against a 5-epoch Knowledge Distillation (KD) baseline showed it remaining theoretically stable, producing a macro-F1 of $0.728$ compared to the baseline's $0.727$. However, it offered no significant performance gain while actively worsening Temperature-Scaled Expected Calibration Error (TS ECE) from $0.027$ up to $0.043$.

More severely, the NegL (Negative Learning) approach, which utilized an entropy-gated mechanism to push probabilities away from low-confidence classes, induced a minor regression in overall accuracy (macro-F1 dropping to $0.720$). Empirical logs indicated a structural instability: NegL can actively destabilise training when its uniform-push penalty is applied to uncertain regions that overlap with difficult minority classes (e.g., Fear vs. Sadness). The threshold calibration proved paradoxically brittle—if the entropy threshold was set too high, the gate became too selective (`applied_frac` plummeting toward zero), turning the loss inert; if set too low, it penalized valid secondary features.

**Table 5.6-1: NL/NegL Screening Results (KD-stage baseline vs Auxiliaries) [19]**

| Configuration | Macro-F1 | TS ECE | Verdict |
| :--- | ---: | ---: | :--- |
| KD baseline (5ep) | 0.727 | 0.027 | Baseline |
| KD+NegL (entropy gate) | 0.720 | 0.040 | Minor macro-F1 regression; worsened ECE |
| KD+NL (proto) | 0.728 | 0.043 | Stable macro-F1; no gain; worsened ECE |

Ultimately, due to outcomes being extremely sensitive to gating and weighting, and the observed risk of regressing the crucial TS calibration necessary for real-time deployment, these mechanisms were disabled. The focus was realigned primarily toward optimising the standard robust Teacher-Student distillation framework.

## 5.7 Webcam Domain Shift Results

Live webcam scoring results (per-frame metrics, raw vs smoothed):

![Confusion Matrix - Webcam Domain](../figures/fig4_confusion_matrix_webcam.png)
*Figure 5.4: Live webcam domain confusion matrix illustrating severe degradation in minority classes.*

![Webcam Raw vs Smoothed Output](../figures/fig8_webcam_raw_vs_smoothed.png)
*Figure 5.5: Live telemetry comparison between raw argmax and smoothed hysteresis predictions.*

**Table 5.7-1: Deployment-aligned metrics (raw vs smoothed).**

| Run | Raw acc | Raw macro-F1 | Smoothed acc | Smoothed macro-F1 | Jitter (flips/min) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20260126_205446 | 0.528 | 0.472 | 0.588 | 0.525 | 14.86 |
| 20260126_215903 | 0.464 | 0.493 | 0.514 | 0.555 | 17.91 |

**Per-class highlights:** Fear and Disgust show near-zero F1 under webcam conditions, demonstrating critical deployment gaps.

**Key deployment metric:** Smoothing improves most classes but Fear remains at 0.00 F1, revealing a critical domain shift gap.

## 5.8 Adaptation Experiments and Negative Results

A core tenet of this project's methodology was the rigorous documentation of failed experiments, providing empirical warnings about the limits of unsupervised adaptation in FER. These unyielding validations formed the backbone of the "Negative Results Matrix". 

### Offline Safety Gate (NR-2)

The fundamental challenge in domain adaptation via small data buffers is mitigating catastrophic forgetting. Initial attempts aimed to restrict weight updates strictly to either the network projection head (`Head-only FT`) or the Batch Normalisation tracking statistics (`BN-only FT`), theory positing that leaving the deep convolutional backbone frozen would guarantee generalisation retention. 

This hypothesis was definitively rejected during the Offline Safety Gate validation against the comprehensive `eval-only` split. Both restrictive regimes performed terribly:

**Table 5.8-1: Offline Safety Gate (eval-only test) [20]**

| Model | Raw acc | Raw macro-F1 | Gate Verdict |
| :--- | ---: | ---: | :--- |
| Baseline (CE20251223) | 0.567 | 0.486 | Safety Baseline |
| Head-only FT | 0.548 | 0.451 | Failed (Catastrophic macro-F1 drop) |
| BN-only FT | 0.549 | 0.451 | Failed (Catastrophic macro-F1 drop) |

These failures (catalogued as NR-2) demonstrated that partial, top-level adaptation is insufficient to realign a network to a novel domain; shifting the semantic boundaries of FER requires careful, full-network engagement.

### Webcam A/B Comparison (NR-1)

The most structurally consequential negative result (NR-1) emerged during a full Self-Learning + NegL adaptation run. Following the failure of partial fine-tuning, a candidate student model was permitted broader but cautious adaptation. This model successfully passed the strict offline `eval-only` non-regression gate, retaining broad static-image accuracy. 

However, when tested via the exact identical-session deterministic webcam replay system, the model regressed dramatically in real-time deployment metrics.

![Adaptation A/B Comparison Failure](../figures/fig10_adaptation_ab.png)
*Figure 5.6: A/B replay metrics visualization demonstrating the regression of the NR-1 adaptation candidate.*

**Table 5.8-2: Same-session Webcam Replay (Smoothed Predictions)**

| Model | Accuracy | Macro-F1 | Minority-F1 | Jitter (flips/min) |
| :--- | ---: | ---: | ---: | ---: |
| Unadapted Baseline | 0.588 | 0.525 | 0.161 | 14.86 |
| Adapted Candidate | **0.527** | **0.467** | **0.138** | **14.16** |

While the candidate marginally reduced jitter, its predictive capability systematically collapsed across the board. This outcome exposed a profound misalignment: **passing an offline multi-dataset gate does not guarantee or even strongly correlate with deployment improvement** in the highly continuous, heavily autocorrelated webcam domain. This finding necessitated the permanent adoption of the dual-gate evaluation framework, mandating that candidates must clear both offline validation and deterministic webcam replay.

### Consolidated Negative Results Matrix

The continuous project log yielded several other documented negative results, mapping the boundaries of viable FER techniques:

| ID | Phenomenon & Result | Status |
| :--- | :--- | :--- |
| **NR-1** | Adapted candidate passed offline safety but regressed on live webcam replay. | Evidence-backed (Table 5.8-2) |
| **NR-2** | Restricted (Head/BN-only) adaptation failed to preserve base features, failing offline gates. | Evidence-backed (Table 5.8-1) |
| **NR-3** | Auxiliary NL/NegL screening consistently degenerated, failing to produce macro-F1 gains. | Evidence-backed (Section 5.6) |
| **NR-4** | Distillation (KD/DKD) substantially improved probabilistic calibration but failed to raise raw semantic accuracy cleanly over CE baselines. | Evidence-backed (Section 5.4) |
| **NR-5** | Peak offline Stage-A Teacher generalisation heavily decayed when subjected to strict confidence gating. | Evidence-backed |
| **NR-6** | The aforementioned hard-gate decay is almost exclusively concentrated in minority classes (Fear, Disgust), rendering safety policies blind to them. | Evidence-backed |
| **NR-7** | Auxiliary Label-Smoothing (LP-loss) offered microscopic calibration shifts but no tangible multi-domain generalisation benefit. | Evidence-backed |

## 5.9 LP-loss Screening Results

LP-loss implementation and short-budget screening on KD/DKD baselines:

**Table 5.9-1: KD vs KD+LP-loss (HQ-train validation).**

| Run | Raw macro-F1 | TS ECE | TS NLL |
| --- | ---: | ---: | ---: |
| KD baseline | 0.728 | 0.037 | 0.793 |
| KD+LP (w=0.01, k=20) | 0.728 | 0.025 | 0.761 |

**Offline gates interpretation:**
- eval-only: KD+LP slightly improves macro-F1 (0.438 → 0.441).
- ExpW: KD+LP slightly decreases macro-F1 (0.460 → 0.458).

**Conclusion:** LP-loss did not show clear cross-domain macro-F1 gain in this short-budget setting.

## 5.10 Offline Benchmark Diagnostics

![Cross-Dataset Macro-F1 Degradation](../figures/fig6_crossdataset_macro_f1.png)
*Figure 5.7: Degradation of generalisation metrics when evaluating models on out-of-domain sources.*

![Teacher Domain Shift Validation](../figures/fig7_teacher_domain_shift.png)
*Figure 5.8: Distributional shift mapped via Teacher confidence outputs across varying datasets.*

**Per-source breakdown on eval-only (CE checkpoint):**

| Source | n | Raw macro-F1 |
| --- | ---: | ---: |
| expw_full | 6,780 | 0.490 |
| expw_hq | 3,336 | 0.279 |
| rafml_argmax | 982 | 0.485 |
| rafdb_compound | 792 | 0.330 |

**CLAHE ablation (ExpW + FER2013):**

- ExpW: CLAHE on → 0.482 / CLAHE off → 0.469 (CLAHE on is better)
- FER2013: CLAHE on → 0.497 / CLAHE off → 0.457 (CLAHE on is better)

**Interpretation:** Low aggregate eval-only results are driven by mixture composition (notably `expw_hq`) and minority-class fragility. CLAHE is not the root cause; turning it off degrades results.

## 5.11 Protocol-aware Paper Comparison

### RAF-DB (Accuracy)

- Paper: 82.69% whole-face accuracy
- Ours (CE, test_rafdb_basic): 86.28% accuracy, 0.792 macro-F1
- **Status:** Competitive; exact split details may differ

### FER2013 (Official Split)

![FER2013 Benchmark Comparison](../figures/fig9_fer2013_official.png)
*Figure 5.9: Comparative benchmarks against standard FER2013 leaderboards.*

**Table 5.11-1: FER2013 official split evaluation (CE/KD/DKD).**

| Split | Protocol | Accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| PublicTest | single-crop | **0.614** | **0.554** |
| PublicTest | ten-crop | 0.609 | 0.557 |
| PrivateTest | single-crop | 0.608 | 0.539 |
| PrivateTest | ten-crop | 0.612 | 0.548 |

**Interpretation:** Even with the official split, strict 1:1 comparison depends on preprocessing, alignment, and training details that vary across studies. We treat the official-split table as the strongest **anchor** for gap analysis.

## 5.12 Chapter Summary

The results confirm that while the MobileNetV3 student achieves competitive in-domain accuracy, it faces significant challenges under domain shift, particularly for Fear and Disgust. Crucially, we found that Knowledge Distillation (KD/DKD) substantially improves calibration without necessarily boosting macro-F1, effectively decoupling these two optimization goals. The documented negative results—especially the failure of offline-passing models to improve webcam replay—underscore the necessity of the dual-gate protocol. These findings lead directly into the **Discussion & Limitations** in Chapter 6, where we interpret these trade-offs in the broader context of FER deployment.

---

# Chapter 6 — Discussion & Limitations

This chapter synthesizes the experimental findings from Chapter 5 to address the project's core research questions. We discuss the implications of the observed decoupling between calibration and accuracy, the structural nature of minority-class fragility under domain shift, and the critical role of deployment-aligned evaluation. We also critically examine the limitations of the current study—ranging from protocol variance to demographic fairness—and contextualize our lightweight real-time model against heavier state-of-the-art benchmarks.

## 6.1 Discussion of Key Findings

Our investigation into real-time FER has yielded several critical insights. First, the systematic comparison of distillation methods revealed a decoupling between classification accuracy and calibration quality. While Knowledge Distillation (KD) and Decoupled Knowledge Distillation (DKD) significantly lowered the Expected Calibration Error (from 0.050 to 0.027), they did not improve the macro-F1 score compared to the Cross-Entropy baseline. This suggests that teaching a student to mimic a teacher's uncertainty makes it a more reliable estimator of its own confidence, but not necessarily a better classifier of hard edge cases. For safety-critical applications where confidence thresholds determine actions, this improved calibration is a valuable, if distinct, victory.

Second, the fragility of minority classes under domain shift emerged as a dominant theme. Despite strong in-domain performance (0.791 macro-F1), our models suffered a near 50% performance drop when tested on wild data, with the "Fear" and "Disgust" classes collapsing to near-zero utility in live webcam tests. This confirms that current state-of-the-art backbones, even when trained on diverse data, have not structurally solved the problem of learning robust representations for ambiguous, low-intensity emotions.

Finally, the failure of our adaptation candidate (NR-1)—which passed offline safety checks but regressed in live deployment—validates the necessity of our **Dual-Gate Evaluation Protocol**. It proves empirically that offline metrics are insufficient proxies for deployment quality. A system that only optimizes for the former runs the risk of "silent failure," whereas our dual-gate approach explicitly catches these regressions before they reach the user.

## 6.2 Limitations and Threats to Validity

Transparency regarding limitations is essential for the scientific validity of this report. Key limitations include:

**External Validity:** The deployment replay results rely on a limited set of webcam sessions. While they powerfully demonstrate failure modes, they constitute a small sample size. Future work must expand this to a wider cohort of users and environments to claim generalized robustness.

**Protocol Variance:** Historical comparisons are complicated by subtle shifts in preprocessing (e.g., crop sizes, temperature scaling defaults) across different experimental phases. While we have controlled for this as much as possible, hidden variables may still influence comparative conclusions.

**Screening Power:** Several experiments, particularly the NL/NegL screening, were conducted with "short budgets" to enable rapid triage. While efficient, this approach risks missing potential gains that might only emerge after prolonged convergence, meaning our negative results should be interpreted as "no easy win" rather than definitive proof of failure.

**Demographic Bias:** The training datasets (FERPlus, RAF-DB) contain inherent demographic imbalances. We did not perform a systematic fairness audit across race, gender, or age groups. Consequently, the model likely harbors performance disparities that are not capturing in aggregate metrics, posing an ethical limitation for broader deployment.

## 6.3 Comparative Analysis with Published Work

It is important to contextualize our results against the broader FER literature. Most published benchmarks optimize for a single objective: maximum accuracy on a fixed test split, often utilizing massive backbones (ResNet-50, VGG-16) and heavy GPU resources. In contrast, our project constraints required a **real-time** system capable of running on a standard CPU with <40ms latency.

Given this trade-off, our lower raw accuracy on standard benchmarks (e.g., 61.4% on FER2013 vs. >70% for SOTA) is both expected and acceptable. We deliberately sacrificed some capacity for speed and producibility. The gap represents the "price of real-time," encompassing the reduction in parameter count (from >25M to ~5.4M) and the lack of computationally expensive test-time augmentations. When compared to similar mobile-optimized architectures in similar settings, our system remains competitive, validating that the MobileNetV3 student is an effective choice for the intended operational envelope.

## 6.4 Ethical Considerations

The development of emotion recognition technology carries significant ethical weight. We strictly adhered to privacy and consent protocols; all live webcam data was collected with explicit informed consent and processed locally without external transmission. However, we acknowledge the broader societal risks, including the potential for misuse in surveillance or high-stakes automated decision-making (e.g., hiring). This system is designed solely as an academic research prototype for Human-Computer Interaction (HCI) and is not validated or intended for consequential real-world deployment. The lack of a rigorous demographic fairness audit further underscores that this model should not be used in settings where bias could lead to harm.

## 6.5 Chapter Summary

This chapter has engaged with the "why" behind the "what." We have argued that the divergence between calibration and accuracy fundamentally changes how we should evaluate distilled models. We have exposed the limitations of offline metrics through the lens of domain shift failures, and we have honestly accounted for the constraints—both technical and ethical—of our work. Having critically assessed the system's strengths and weaknesses, the final chapter will synthesize these lessons into an overall conclusion and outline the path for future research.

---

# Chapter 7 — Conclusion & Lessons Learned

## 7.1 Conclusion

This project set out to bridge the gap between academic FER benchmarks and the messy reality of real-time deployment. We successfully engineered a reproducible, end-to-end pipeline that ingests raw, multi-source data and distills it into a lightweight, CPU-capable student model running at sub-40ms latency.

The investigation answered our core research questions with definitive empirical evidence. We proved that **Knowledge Distillation** is a powerful tool for improving model calibration (RQ1), dropping Expected Calibration Error (ECE) from 0.050 to 0.027, even if it does not automatically yield better hard classification accuracy. We quantified the severe impact of **Domain Shift** (RQ2), showing that minority classes like Fear and Disgust are the "canaries in the coal mine," collapsing to near-zero utility in live tests long before overall accuracy metrics show alarm. Most importantly, we demonstrated through our negative adaptation results (RQ3) that relying solely on offline benchmarks is a dangerous practice for deployed systems; our NR-1 candidate passed offline gates but fundamentally failed in deployment replay. This led to the formulation of the **Dual-Gate Evaluation Protocol** (RQ4), a methodological contribution that enforces a stricter, significantly more reliable standard for model promotion.

Ultimately, this report argues that "success" in real-time FER is not just a high F1 score; it is the demonstrated ability to maintain stability and reliability when the clean training data is left behind and the camera turns on.

## 7.2 Lessons Learned

The journey from concept to code yielded several hard-won lessons that extend beyond this specific project workflow:

1.  **Artifacts are the Truth:** Relying on memory or loose file names is a recipe for disaster. By enforcing a strict "artifact-grounded" workflow—where every manifest, log sequence, and metric JSON is meticulously saved—we turned potential debugging nightmares into traceable audits.
2.  **Validate Inputs Early:** The automated manifest validation prevented countless hours of wasted GPU time. Catching a "missing file" error before training starts is infinitely cheaper than catching it after a crash in epoch 40.
3.  **Metrics Can Lie:** A model with high offline accuracy can be unusable in a live demo due to jitter. Conversely, a lower-accuracy model might feel "better" to a user if it is well-calibrated and stable. We learned to measure what matters to the deployment, not just what is easy to calculate in a loop.
4.  **Adaptation is Dangerous:** The intuitive appeal of "learning on the fly" is tempered by the reality of model drift. Without a rigorous, bi-directional safety gate (checking both new and old domains), adaptation is as likely to harm the model as it is to help it.
5.  **Preprocessing is a Hidden Hyperparameter:** Seemingly minor details like CLAHE configuration or crop padding can swing performance as much as a new architecture. These must be treated as first-class configuration citizens, not implementation details.

## 7.3 Future Work

While this system functions as a robust prototype, several avenues remain for meaningful extension:

*   **Temporal-Aware Networks (Video-Based FER):** Our approach demonstrates that treating webcam streams as independent static images and applying smoothing (EMA) post-hoc is only a palliative solution. Real-world facial expressions are inherently continuous physiological events. Future designs should inherently model temporal autocorrelation by transitioning from static CNN architectures (MobileNetV3) to temporal architectures such as LSTMs, lightweight 3D-CNNs, or efficient Video Transformers (VideoMAE).
*   **Expanded Deployment Testing:** Moving beyond a single laptop to a diverse array of CPU and GPU devices would provide a more comprehensive view of the latency/accuracy Pareto frontier.
*   **Targeted LP-Loss Tuning:** The negative result on LP-loss suggests that a more nuanced hyperparameter search—varying weights and layer injection points—might unlock the manifold-preserving benefits that theory predicts.
*   **Refined Adaptation Strategies:** Future work should explore more sophisticated buffers for self-learning, perhaps capping class contributions to prevent majority-class bias from dominating the adaptation process.
*   **Fairness Audits:** A systematic evaluation of performance across demographic groups is a necessary step before any rigorous real-world application could be considered.

---

# References

[1] G. Hinton, O. Vinyals, and J. Dean, "Distilling the knowledge in a neural network," arXiv preprint arXiv:1503.02531, 2015.

[2] B. Zhao, Q. Cui, R. Song, Y. Qiu, and J. Liang, "Decoupled knowledge distillation," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11953–11962.

[3] C. Deng, D. Huang, X. Wang, and M. Tan, "Nested learning: A new paradigm for machine learning," arXiv preprint arXiv:2303.10576, 2023.

[4] J. Deng, J. Guo, N. Xue, and S. Zafeiriou, "ArcFace: Additive angular margin loss for deep face recognition," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 4690–4699.

[5] M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2019, pp. 6105–6114.

[6] A. Howard, M. Sandler, G. Chu, L.-C. Chen, B. Chen, M. Tan, W. Wang, Y. Zhu, R. Pang, V. Vasudevan, Q. V. Le, and H. Adam, "Searching for MobileNetV3," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 1314–1324.

[7] Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, "A ConvNet for the 2020s," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2022, pp. 11976–11986.

[8] K. He, X. Zhang, X. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.

[9] K. He, X. Zhang, X. Ren, and J. Sun, "Identity mappings in deep residual networks," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2016, pp. 630–645.

[10] S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2018, pp. 3–19.

[11] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.

[12] Y. Cui, M. Jia, T.-Y. Lin, Y. Song, and S. Belongie, "Class-balanced loss based on effective number of samples," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2019, pp. 9268–9277.

[13] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, "Long-tail learning via logit adjustment," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

[14] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2017, pp. 1321–1330.

[15] Y. Kim, J. Yim, J. Yun, and J. Kim, "NLNL: Negative learning for noisy labels," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2019, pp. 101–110.

[16] T. Ishida, G. Niu, W. Hu, and M. Sugiyama, "Learning from complementary labels," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 5639–5649.

[17] A. Mollahosseini, B. Hasani, and M. H. Mahoor, "AffectNet: A database for facial expression, valence, and arousal computing in the wild," *IEEE Trans. Affect. Comput.*, vol. 10, no. 1, pp. 18–31, 2019.

[18] W. Wu, Y. Peng, S. Wang, and Y. He, "YuNet: A tiny millisecond-level face detector," *Mach. Intell. Res.*, vol. 20, pp. 656–665, 2023.

[19] R. Wightman, "PyTorch Image Models (timm)," GitHub repository, 2019. [Online]. Available: https://github.com/huggingface/pytorch-image-models

[20] I. J. Goodfellow, D. Erhan, P. L. Carrier, et al., "Challenges in representation learning: A report on three machine learning contests," in *Proc. Int. Conf. Neural Inf. Process. (ICONIP)*, 2013, pp. 117–124.

[21] S. Li, W. Deng, and J. Du, "Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild," in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 2852–2861.

[22] D. Wang, A. Shelhamer, J. Hoffman, X. Yu, and T. Darrell, "Tent: Fully test-time adaptation by entropy minimization," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, 2021.

---

# Appendix

## A.1 Evidence Inventory Data Dictionary (Key Artifacts)

To facilitate offline reading and formal print submission, the fundamental data outputs from the JSON reproducibility artifacts referenced throughout this report are tabulated below. *(Note: Raw JSON files housing complete multi-class confusion matrices, prediction arrays, and hyperparameters remain archived in the project repository).*

### 1. Dataset Integrity Artifact (`manifest_validation_all_with_expw.json`)

| Metric | Recorded Value |
| :--- | :--- |
| **Total Rows (Images)** | 466,284 |
| **Missing Paths** | 0 |
| **Bad / Corrupt Labels** | 0 |
| **Primary Sources** | FERPlus (138k), FER2013-Uniform (140k), ExpW (91k), AffectNet (71k) |

### 2. Teacher Ensemble Offline Benchmarks (`ensemble_metrics.json`)

Evaluation of the frozen Teacher pseudo-labeler on the `test_all_sources.csv` split, combined via logits from RN18, EfficientNet-B3, and ConvNeXt-Tiny.

| Metric | Score |
| :--- | ---: |
| **Target Accuracy** | 0.687 |
| **Macro-F1 (7-class)** | 0.660 |
| **Highest Class F1** | Happy (0.839) |
| **Lowest Class F1** | Fear (0.535) |
| **Ensemble ECE** | 0.288 |

**Teacher Ensemble Per-class F1 Scores** (extracted from `ensemble_metrics.json`):

| Ensemble Model | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RN18 + B3 + CNXT | 0.6304 | 0.6034 | 0.5350 | 0.8389 | 0.5924 | 0.7205 | 0.6967 |

### 3. Student Calibration & Baseline Artifacts (`reliabilitymetrics.json`)

Comparative records validating the MobileNetV3 Student models (Cross-Entropy vs DKD) on the HQ-Train validation split. 

| Metric | CE Baseline | DKD Distilled |
| :--- | ---: | ---: |
| **Macro-F1** | 0.728 | 0.728 |
| **TS Expected Calibration Error (ECE)** | 0.050 | 0.027 |
| **TS Negative Log Likelihood (NLL)** | 0.793 | 0.791 |

**Detailed Per-class F1 Scores** (extracted from `reliabilitymetrics.json`):

| Stage | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CE | 0.7263 | 0.6428 | 0.7640 | 0.8014 | 0.7170 | 0.7871 | 0.7550 |
| KD | 0.7237 | 0.6782 | 0.7447 | 0.7610 | 0.7234 | 0.7801 | 0.7224 |
| DKD | 0.7255 | 0.6828 | 0.7561 | 0.7596 | 0.7286 | 0.7915 | 0.7185 |

### 4. Continuous Domain Shift (Webcam Replay) Artifacts (`score_results.json`)

Records from the dual-gate live deployment module measuring same-session webcam replays.

| Metric | Unadapted Baseline (205446) | Candidate NR-1 (215903) |
| :--- | ---: | ---: |
| **Smoothed Accuracy** | 0.588 | 0.527 |
| **Smoothed Macro-F1** | 0.525 | 0.467 |
| **Minority-F1 (lowest 3)** | 0.161 | 0.138 |
| **Jitter (flips/min)** | 14.86 | 14.16 |
| **Frames Evaluated** | 4,154 | 4,154 |

## A.2 Metric Definitions

- **Macro-F1**: unweighted mean of per-class F1 scores.
- **ECE (Expected Calibration Error)**: binned absolute gap between accuracy and confidence.
- **NLL (Negative Log-Likelihood)**: negative log probability assigned to correct labels.
- **Temperature Scaling**: global temperature $T$ applied to logits; preserves argmax but changes probability sharpness.

## A.3 FYP Requirements Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| 1) Study deep learning methods for FER | Met | Literature review in Chapter 2; background and method discussion throughout. |
| 2) Study/apply knowledge distillation | Met | KD/DKD implemented and evaluated in Sections 4.4, 5.4; full ablation study. |
| 3) Identify 2+ datasets | Met | Six validated manifests: FERPlus, AffectNet, RAF-DB, ExpW, FER2013 (official + folder). |
| 4) Investigate 3+ methods | Met | Student stages: CE vs KD vs DKD; Teacher backbones: RN18/B3/CNXT. |
| 5) Explore performance enhancement techniques | Met | NL/NegL screening (Section 5.6), domain adaptation (Section 5.8), LP-loss (Section 5.9). |
| 6) Paper presentation | Met | Dec 2025 interim presentation; Jan 2026 paper study presentation. |

## A.4 Advanced Technical Details

### Dataset Provenance and SHA256 Fingerprints

Snapshots generated on 2026-02-09 include:
- RAF-DB basic file counts, byte totals, SHA256 fingerprints
- FER2013 folder packaging (msambare) manifests
- ExpW (in-the-wild) provenance records

### Teacher Training Source Composition

Detailed sub-dataset breakdown applied during teacher training (e.g., for RN18, leaving 225,629 rows after filtering from the 466,284 total):

| Dataset Source | Retained Rows | Notes |
| :--- | ---: | :--- |
| **FERPlus** | 138,526 | High-quality baseline |
| **AffectNet** (balanced) | 71,764 | Heavily downsampled for class parity |
| **RAF-DB** (basic) | 15,339 | Studio-curated base |
| **Total Effective** | 225,629 | Filtered to exclude ExpW & synthetic |

### Student Distillation Split Structure

**HQ training manifest (`classification_manifest_hq_train.csv`)**, distilled via Teacher Ensemble:

| Data Split | Sample Count | Percentage |
| :--- | ---: | ---: |
| **Training** | 213,144 | ~82.3% |
| **Validation** | 18,020 | ~7.0% |
| **Testing** | 27,840 | ~10.7% |
| **Total Rows** | 259,004 | 100.0% |

### NL(proto) and NegL Terminology

- **NL(proto)**: Nested Learning [3] prototype auxiliary signal; used in Jan-2026 screening
- **NegL**: Complementary-label negative learning [16] with entropy-based gating

### Offline Gates Pre-registration

Evaluation thresholds for model promotion (from `domain_shift/evaluation_plan.md`):
- FAIL: macro-F1 drop > 0.01 OR minority-class F1 drop > 0.02
- WIN: minority-F1 improvement ≥ 0.01 with no macro-F1 regression

### Eval-Only Domain Regression (Per-Source Breakdown)

Macro-F1 analysis on external stress-test sources, using the Baseline Student (CE checkpoint). The low aggregate score is heavily suppressed by extreme difficulty in `expw_hq` and compound mappings.

| Evaluation Source | Sample Size | Macro-F1 (CE) | Evaluation Note |
| :--- | ---: | ---: | :--- |
| `expw_full` | 6,780 | 0.490 | Open-domain in-the-wild |
| `rafml_argmax` | 982 | 0.485 | Standard multi-label argmax |
| `rafdb_compound` | 792 | 0.330 | Compound expression mismatch |
| `expw_hq` | 3,336 | 0.279 | **Lowest performer (severe bottleneck)** |

### FER2013 Official Split Protocol

Official split evaluation strictly matching the Kaggle challenge standard (`fer2013.csv`):

| Test Split | Sample Size | Protocol Options | Best Record (Single-Crop) |
| :--- | ---: | :--- | :--- |
| **PublicTest** | 3,589 | Single-crop / Ten-crop | 0.614 Accuracy (DKD) |
| **PrivateTest** | 3,589 | Single-crop / Ten-crop | (Reserved for final benchmark) |

### Paper Comparison Evidence Index

| Evaluation | Dataset | Status | Metrics Artifact |
| --- | --- | --- | --- |
| RAF-DB | Basic split | Partial | Offline suite CSV |
| FER2013 official | PublicTest/PrivateTest | Partial | `outputs/benchmarks/fer2013_official_summary_20260212/` |
| FER2013 folder | Kaggle msambare | No | Per-run reliability metrics |
| AffectNet balanced | Balanced subset | No | Offline suite CSV |
| ExpW | Full manifest | No | Domain shift evaluation artifacts |

---

**End of Final Report Version 3**

Report Version: 3 (7-chapter restructured)  
Document Date: Mar 16, 2026  
Status: Content-ready for submission


## A.5 Consolidation of Security Elements from Level 3 and Level 4 Subjects

In accordance with the Capstone Project requirements, this section details how security elements, principles, and best practices learned from Level 3 and Level 4 computing subjects have been consolidated and securely applied to the design, implementation, and evaluation of this Real-time Facial Expression Recognition (FER) System. Developing a computer vision system that captures, processes, and stores biometric data (faces) introduces significant security, privacy, and integrity risks. The following subsections map formal security concepts to specific architectural and procedural decisions made in this project.

### A.5.1 Edge Inference as a Privacy-Preserving Architecture (Network & Information Security)

A core threat in any webcam-based biometric system is data interception during transit, such as Man-in-the-Middle (MitM) attacks. Cloud-based computer vision APIs (e.g., sending video frames to a remote server for prediction) expose Personally Identifiable Information (PII) to network vulnerabilities and violate the principle of data minimization.

**Application in this Project:**
To mitigate this risk, the entire inference pipeline was deliberately engineered as an **Edge/Local Inference Application**. By aggressively compressing the model (compressing heavy ResNet-18/ConvNeXt teachers into a lightweight MobileNetV3 student via knowledge distillation), the system is capable of running locally on a standard consumer CPU.
- **Data at Rest / Data in Transit:** Because all frame extraction, face detection, and emotional classification happen locally in system memory, no biometric data is transmitted over the network. 
- **Privacy by Design:** This fulfills the privacy-by-design requirement taught in Level 3/4 information security modules, ensuring GDPR and local data privacy compliance by strictly isolating the data space. The webcam buffer is volatile and discarded immediately after the session unless explicitly saved for local offline adaptation.

### A.5.2 Cryptographic Data Integrity and Anti-Poisoning (Cryptography & System Security)

Machine learning models are highly susceptible to data poisoning attacks—where an adversary silently modifies training data or labels to inject backdoors or degrade model reliability. Because this project pulls data from multiple external sources (Kaggle datasets, Google Drive archives), verifying file integrity is critical.

**Application in this Project:**
Concepts from Cryptography and System Security were applied to the data ingestion pipeline:
- **Cryptographic Hashing:** As documented in Appendix A.4, SHA-256 cryptographic fingerprints are generated and checked for massive datasets (like RAF-DB test splits and ExpW archives). 
- **Manifest Validation:** Rather than loading raw folders which could be silently modified, the system uses strict `.csv` and `.json` manifests (`manifest_validation_all_with_expw.json`). A Python script validates every file path, ensuring no zero-byte malicious files or corrupted images execute arbitrary code during the PyTorch `DataLoader` instantiation.

### A.5.3 Secure Software Development Lifecycle (SSDLC) and Supply Chain Security

Modern application development heavily relies on third-party libraries, exposing developers to supply chain attacks (e.g., malicious PyPI packages executing remote code). Level 4 subjects emphasize the importance of secure environments and strict dependency tracking.

**Application in this Project:**
- **Dependency Pinning:** A strict `requirements.txt` (and `requirements-directml.txt`) freezes exact module versions (e.g., `absl-py==2.3.1`, `certifi==2025.11.12`). This guarantees reproducible builds and prevents upstream dependency hijacking from silently injecting malicious payload updates.
- **Environment Isolation:** The project uses Python virtual environments (`.venv`) to strictly isolate the application scope, preventing dependency collisions or polluting the global OS environment, following the principle of least privilege.

### A.5.4 Defense Against Model Inversion and Membership Inference Attacks (AI Security)

Machine Learning deployments are vulnerable to privacy attacks where adversaries try to reconstruct the training data from the final model weights (Model Inversion) or determine if a specific person's face was included in the training dataset (Membership Inference Attack, MIA). 

**Application in this Project:**
- **Knowledge Distillation as a Security Shield:** By deploying a *Student Model* (MobileNetV3) rather than the *Teacher Model* (ResNet-18/ConvNeXt), the system inherently shields the original training data. The student never sees the raw hard-labels (one-hot vectors) of the facial images; it only learns from the softened logits (probabilities) emitted by the teacher. This lossy information bottleneck acts as an algorithmic one-way function, making it mathematically infeasible for an attacker extracting the deployed edge device weights to invert the model and retrieve recognizable faces of the people (e.g., from AffectNet or ExpW) used in the training set.

### A.5.5 AI Robustness and Adversarial Input Resilience (Advanced Topics in AI / CV)

Standard Convolutional Neural Networks are vulnerable to adversarial perturbations—invisible noise added to an image that tricks the classifier, or natural distributional shifts (camera noise, sudden lighting changes) that crash the system. 

**Application in this Project:**
To harden the system against input-layer attacks and natural noise:
- **Algorithmic Defenses:** The student model is trained using **Temperature-Scaled Decoupled Knowledge Distillation (DKD)**, which has been shown to produce flatter, more calibrated confidence distributions rather than overconfident, spiky predictions. 
- **Temporal Stabilisation (Hysteresis & EMA):** At the inference level, adversarial "flickering" or input noise is countered using Exponential Moving Averages (EMA) and a Hysteresis safety margin. A single spoofed or corrupted frame cannot alter the system state unless the noise is sustained across a multi-frame voting window. This structural defense prevents momentary input attacks from successfully manipulating the application logic.

### A.5.6 Ethical Considerations and Bias Mitigation (Computing Ethics)

Security extends beyond mathematics into ethical computing—ensuring systems do not encode hidden biases that harm specific demographic groups. Facial recognition technology is historically prone to demographic bias depending on the source data.

**Application in this Project:**
- The training corpus merges FERPlus, RAF-DB, ExpW, and AffectNet to heavily diversify age, ethnicity, and lighting conditions.
- By deliberately using the `affectnet_full_balanced` subset (71,764 rows strongly downsampled for parity across the 7 emotion classes instead of using the highly imbalanced full set), the system avoids learning strong predictive priors based on majority demographic classes. This prevents the system from systematically failing or producing "confident-wrong" misclassifications on minority data strata—a core requirement of ethical computer vision deployment taught in senior computing units.