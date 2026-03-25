# Final Presentation Script (15 min)

Project: Real-time Facial Expression Recognition System  
Presenter: Donovan Ma  
Date: 2026-03-22

---

## Slide 1 - Title and One-line Thesis (0:45)

Good [morning/afternoon], Professor and everyone.  
My final-year project is a real-time facial expression recognition system designed for deployment, not only for offline benchmark performance.

The core thesis of this presentation is: in real-time FER, offline non-regression is not sufficient for checkpoint promotion. We need a dual-gate protocol that checks both offline safety and same-session replay behavior.

Today I will focus on technical design choices, key failure modes, and how those failures changed the evaluation protocol.

---

## Slide 2 - Outline (0:30)

This talk has six parts.

First, I define the deployment problem and constraints.  
Second, I present the end-to-end system design.  
Third, I explain teacher-student training and calibration behavior.  
Fourth, I introduce the dual-gate promotion protocol.  
Fifth, I show negative results and fix directions.  
Finally, I close with conclusions.

---

## Slide 3 - Problem Setting and Deployment Constraints (1:00)

FER in deployment is different from FER on static benchmarks.

In deployment, the model must be stable across frames, confidence-calibrated for threshold logic, robust to webcam domain shift, and fast enough on CPU.

In this project, the practical deployment target is sub-40 millisecond tail latency in isolated model runtime, with stable on-screen predictions for users.

The label space is the canonical 7 classes: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## Slide 4 - Research Questions and Contributions (1:00)

I organized the project around four research questions.

RQ1 asks whether KD and DKD improve lightweight student behavior and calibration.  
RQ2 asks how domain shift affects per-class behavior.  
RQ3 asks whether Self-Learning plus NegL can adapt safely.  
RQ4 asks what protocol is valid when offline and deployment metrics diverge.

The three main contributions are:

One, an artifact-grounded teacher-student FER pipeline.  
Two, a dual-gate promotion protocol.  
Three, a formal negative-result catalog from NR-1 to NR-7.

---

## Slide 5 - End-to-End System Architecture (1:30)

This slide shows the full technical pipeline.

We start from multi-source raw datasets, then clean and validate through manifests.  
Next, we train three teachers and build a weighted ensemble.  
Then we export soft labels and train a lightweight student through CE, KD, and DKD stages.  
Finally, we deploy with a real-time inference loop and evaluate candidates using a dual-gate process before any promotion.

An important engineering principle here is artifact traceability.

Each stage emits concrete outputs such as manifests, checkpoints, calibration files, reliability metrics, and replay scoring artifacts. This is what makes the pipeline reproducible and auditable.

---

## Slide 6 - Data Engineering and Provenance (1:30)

Data quality controls are foundational in this project.

Instead of relying on folder assumptions, we use a manifest-first data pipeline with automated path and label validation, plus provenance snapshots.

The validated dataset size is 466,284 rows, with zero missing paths and zero invalid labels.

The value of this design is practical: it prevents silent dataset corruption and avoids wasting training time on broken inputs. It also ensures that model comparisons are based on fixed and auditable data definitions.

---

## Slide 7 - Teacher Stack and Ensemble Design (1:30)

For teachers, I use three complementary backbones: ResNet-18, EfficientNet-B3, and ConvNeXt-Tiny.

Teachers are trained with ArcFace-style margin training to enforce stronger inter-class separation in representation space.

For ensemble construction, I apply weighted logit fusion and select weights on mixed-domain evaluation. The selected weights are 0.4, 0.4, and 0.2 for RN18, B3, and CNXT.

This ensemble achieves 0.687 accuracy and 0.660 macro-F1 on test_all_sources.

Its main role is to generate robust soft targets for student distillation.

---

## Slide 8 - Student Distillation and Calibration Behavior (1:45)

The deployment student is MobileNetV3-Large because it provides a useful accuracy-latency trade-off for edge-style CPU inference.

Training follows three stages:

First CE baseline.  
Then KD with teacher soft targets.  
Then DKD with decoupled distillation components.

The key result is a calibration-accuracy decoupling.

Temperature-scaled ECE improves from 0.050 in CE to 0.028 in KD and 0.027 in DKD, but macro-F1 does not exceed the CE baseline in this setup.

This means calibration quality and decision-boundary quality must be measured independently in deployment-oriented FER.

---

## Slide 9 - Real-time Inference and Stabilization Design (1:15)

The live pipeline is: YuNet detection, preprocessing, ONNX student inference, temperature scaling, EMA smoothing, and hysteresis decision.

Stabilization is controlled through parameters such as ema_alpha and hysteresis_delta.

I evaluate both semantic correctness and temporal stability using raw and smoothed metrics, plus jitter flips per minute.

This avoids optimizing only static-image behavior while ignoring user-facing temporal consistency.

---

## Slide 10 - Dual-Gate Promotion Protocol (1:30)

This slide is the methodological center of the project.

Single-gate offline promotion is unsafe because a model can pass offline and still regress in deployment replay.

Gate A is offline non-regression on eval-only and ExpW checks.  
Gate B is same-session replay quality under fixed labels and fixed stabilizer settings with A/B comparability.

Promotion happens only when both gates pass.

This AND condition is what prevents benchmark-only improvement from being mistaken as deployment improvement.

---

## Slide 11 - Negative Results That Changed the Method (2:15)

The strongest negative result is NR-1.

The adapted candidate passed offline gate checks but failed replay quality checks.

On same-session replay with smoothed predictions:

Baseline: accuracy 0.588, macro-F1 0.525, minority-F1 0.161, jitter 14.86.  
Adapted: accuracy 0.527, macro-F1 0.467, minority-F1 0.138, jitter 14.16.

So jitter slightly improved, but predictive quality dropped significantly. That is not a promotable checkpoint.

NR-2 further showed that restricted adaptation such as head-only and BN-only fine-tuning failed offline safety gates.

Other negative findings, NR-3 to NR-7, showed instability or weak gains in auxiliary methods, and hard-gate performance decay concentrated in minority classes.

These failures are not side notes. They are direct evidence that shaped the final protocol.

---

## Slide 12 - Limits Found and How We Fix Them (1:30)

Limit one: offline metrics can mislead deployment quality.  
Fix: enforce dual-gate promotion as mandatory.

Limit two: minority-class fragility, especially Fear and Disgust under domain shift.  
Fix direction: class-balanced replay buffers and per-class gate monitoring, not only global macro-F1.

Limit three: adaptation instability on small webcam buffers.  
Fix direction: conservative updates, stricter confidence and entropy gating, and replay-first acceptance logic.

Limit four: short-budget screening may miss long-horizon behavior.  
Fix direction: two-phase screening, fast triage then longer confirmatory runs for shortlisted candidates.

---

## Slide 13 - Runtime Evidence and Deployment Readiness (0:45)

For runtime evidence in isolated model benchmarking on CPU:

Mean inference is 23.01 milliseconds, P95 is 30.04, P99 is 34.97, throughput is 43.46 FPS, and ONNX model size is 16.05 MB.

These numbers support practical edge deployment, while still acknowledging known failure boundaries under domain shift.

---

## Slide 14 - Conclusion (0:45)

To conclude, this project delivers more than a model.

It delivers a deployment-valid evaluation workflow for real-time FER.

The key technical value is the combination of reproducible artifact-grounded engineering, calibration-aware distillation analysis, and negative-result-driven dual-gate promotion control.

Final message: in real-time FER, reliable promotion policy is as important as model architecture.

---

## Slide 15 - Q&A Backup (optional)

If asked, I can further explain:

Why calibration gains did not automatically increase macro-F1.  
Why same-session replay A/B is stricter and more trustworthy than cross-run live comparison.  
How negative results were used as first-class design evidence.

---

## Short Delivery Notes

If time is tight, compress these parts first:

1. Slide 6 by about 20 seconds.  
2. Slide 8 by about 25 seconds.  
3. Slide 11 by about 60 to 75 seconds.

This keeps the total delivery inside a 15-minute slot while preserving all core technical arguments.
