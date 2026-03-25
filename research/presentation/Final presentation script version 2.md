# Final Presentation Script (Version 2, 15 min)

Project: Real-time Facial Expression Recognition System  
Presenter: Donovan Ma  
Date: 2026-03-23  
Target deck: Final_presentation_version_2_2026_3_23.md

---

## Slide 1 - Title and One-line Thesis (0:40)

Good morning Professor and everyone.  
My project is a real-time facial expression recognition system that is designed for deployment, not only offline benchmarking.

The core thesis is this: passing offline non-regression alone is not enough for checkpoint promotion. In this project, a model is only promotable when it passes both offline safety gates and same-session replay quality gates.

Today I will focus on technical design, mathematical mechanisms inside the training and inference pipeline, and the negative results that changed our final protocol.

---

## Slide 2 - Outline (0:20)

I will go through six parts quickly:  
problem constraints, full system design, teacher-student training with calibration behavior, dual-gate promotion logic, key negative results and limits, then conclusion.

---

## Slide 3 - Problem Setting and Deployment Constraints (0:55)

The deployment problem is harder than standard FER leaderboard testing.

Offline metrics do not fully capture four deployment requirements: temporal stability, confidence calibration, domain-shift robustness, and CPU latency constraints.

In this project, our deployment target is CPU-friendly inference with sub-40ms tail latency on isolated runtime profiling, stable frame-to-frame output for users, and a safe checkpoint promotion policy.

We keep a fixed 7-class label space: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.

---

## Slide 4 - Research Questions and Contributions (0:55)

Our four research questions are:

RQ1, can KD and DKD improve lightweight student behavior and calibration?  
RQ2, how large is the domain-shift gap from curated data to webcam deployment?  
RQ3, can Self-Learning plus NegL adapt safely without harmful regression?  
RQ4, what promotion protocol is valid when offline and deployment signals disagree?

Our contributions are threefold: an artifact-grounded FER pipeline, a dual-gate promotion protocol, and a formal negative-result catalog from NR-1 to NR-7.

---

## Slide 5 - End-to-End System Architecture (1:15)

This slide is the full engineering chain.

We start from raw datasets, then clean and validate manifests, train teachers, build weighted ensemble soft labels, train CE/KD/DKD students, run real-time inference, evaluate through dual gates, and then decide promote or reject.

The key principle is auditability: each stage emits persistent artifacts, including manifests, checkpoints, calibration files, reliability metrics, and replay score outputs.

This artifact-first design is important because every claim in the report and presentation can be traced back to concrete files.

---

## Slide 6 - Data Engineering and Provenance (1:10)

Before model design, we solved data reliability.

Our strategy uses multi-source training and stress evaluation across FERPlus, AffectNet, RAF-DB, ExpW, and FER2013 variants.

The integrity controls are manifest-first ingestion, path and label validation, and SHA256-based provenance snapshots.

In this project we validated 466,284 rows with zero missing paths and zero invalid labels.

Why does this matter? Because if data integrity is unstable, then teacher-student comparisons and replay A/B conclusions are not trustworthy.

---

## Slide 7 - Teacher Stack and Ensemble Design (Deep Technical) (1:20)

Here we explain the first math concept.

We use three teacher backbones: ResNet-18, EfficientNet-B3, and ConvNeXt-Tiny.

For teacher training, we apply ArcFace-style angular margin supervision. The core term is:

$$
\cos(\theta + m)
$$

What is this for? It increases angular separation between classes, so class boundaries are sharper in feature space.

How did we use it in this project? We applied margin-based teacher training before distillation, then fused teacher logits with weights 0.4, 0.4, and 0.2 for RN18, B3, and CNXT.

The selected ensemble achieved mixed-domain evidence on test_all_sources with accuracy 0.687 and macro-F1 0.660, and those fused soft targets became the supervision source for student KD and DKD.

---

## Slide 8 - Student Distillation and Calibration Behavior (Most Technical) (1:30)

This slide contains the central distillation math.

Our deployment student is MobileNetV3-Large. We train in three stages: CE baseline, then KD, then DKD.

First equation, KD softening:

$$
p_T = \text{softmax}(z/T)
$$

What is this for? Temperature $T$ softens teacher logits into dark-knowledge probabilities, so the student learns inter-class similarity, not only one-hot labels.

How we used it: KD used hard labels plus teacher soft targets generated from the weighted ensemble.

Second equation, DKD decomposition:

$$
L_{DKD} = \alpha L_{TCKD} + \beta L_{NCKD}
$$

What is this for? It separates target-class transfer and non-target transfer so gradient behavior is more controllable.

How we used it: in the DKD student branch, we tuned this decoupled objective to improve confidence behavior while preserving deployment efficiency.

Third equation, calibration metric ECE:

$$
ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} \left|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)\right|
$$

What is this for? ECE measures the gap between predicted confidence and actual correctness.

How we used it: we tracked TS ECE for CE, KD, and DKD; results were 0.050, 0.028, and 0.027. This shows calibration improved strongly, even though macro-F1 did not always increase together.

---

## Slide 9 - Real-time Inference and Stabilization Design (1:10)

Now we move from training math to deployment math.

Our inference chain is YuNet detection, preprocessing with optional CLAHE, ONNX student inference, temperature scaling, EMA smoothing, and hysteresis decision.

The EMA equation is:

$$
p_t = \alpha p_t^{raw} + (1-\alpha)p_{t-1}
$$

What is this for? It is a temporal low-pass filter over class probabilities, reducing frame-level jitter.

How we used it: we fixed EMA and hysteresis settings during replay A/B so model comparisons are fair and reproducible.

For deployment we report both raw and smoothed metrics, plus jitter flips per minute, because stability is a first-class objective.

---

## Slide 10 - Dual-Gate Promotion Protocol (1:20)

This is the decision logic of the whole project.

Single-gate promotion is unsafe, because a model can pass offline gates but still fail replay-domain quality.

Gate A is offline non-regression on eval-only and ExpW with pre-registered thresholds.  
Gate B is same-session replay quality under fixed labels and fixed stabilizer settings.

The promotion rule is strict: promote only if both gates pass.

This rule came directly from our negative-result evidence, not from assumption.

---

## Slide 11 - Negative Results That Changed the Method (1:55)

NR-1 is the key result.

An adapted candidate passed offline gate but failed replay quality gate.

Replay A/B smoothed results were:
baseline acc 0.588, macro-F1 0.525, minority-F1 0.161, jitter 14.86;  
adapted acc 0.527, macro-F1 0.467, minority-F1 0.138, jitter 14.16.

Interpretation: small jitter reduction did not compensate for clear quality regression.

NR-2 showed head-only and BN-only adaptation failing offline safety gate.  
NR-3 to NR-7 showed unstable or weak gains in NL/NegL screening, calibration gains without macro-F1 gains, minority-class hard-gate decay, and no clear LP-loss cross-domain macro-F1 win in short-budget screening.

So negative results were not side notes; they were used as design evidence for protocol changes.

---

## Slide 12 - Limits Found and How We Fix Them (1:15)

We found four practical limits and each has a concrete fix direction.

Limit one: offline-only metrics can mislead deployment, so dual-gate is mandatory.  
Limit two: minority-class fragility under domain shift, so we plan class-balanced replay buffers and per-class gate tracking.  
Limit three: adaptation instability on small webcam buffers, so we use conservative updates with stricter entropy and confidence gating plus replay-first acceptance.  
Limit four: short-budget screening misses long-horizon behavior, so we split screening into fast triage and longer confirmatory runs.

---

## Slide 13 - Runtime Evidence and Deployment Readiness (0:40)

On isolated CPU benchmarking, runtime is deployment-practical: mean 23.01 ms, P95 30.04 ms, P99 34.97 ms, throughput 43.46 FPS, ONNX size 16.05 MB.

So the model is lightweight enough for edge-style use, while known failure boundaries are explicitly documented.

---

## Slide 14 - Conclusion (0:35)

Final message: this project contributes both a FER model pipeline and a deployment-valid promotion method.

The key practical novelty is artifact-grounded reproducibility, calibration-aware distillation analysis, and negative-result-driven dual-gate checkpoint selection.

In real-time FER, promotion policy is as important as model architecture.

---

## Slide 15 - Q&A Backup (optional)

If asked, I will expand on:

1. Why calibration improvement did not automatically increase macro-F1.  
2. Why same-session replay A/B is stricter than cross-run live comparison.  
3. Why negative results were treated as first-class evidence.  
4. Next-step plan for minority-class robustness and fairness audit.

---

## Quick Delivery Notes

- Keep Slide 8 equations concise: explain purpose first, formula second.
- Keep Slide 11 focused on NR-1 numbers; compress NR-3 to NR-7 into one sentence if time is tight.
- If over time, shorten Slide 6 and Slide 12 by one sentence each.
