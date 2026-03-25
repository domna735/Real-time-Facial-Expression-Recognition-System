# Final Presentation (15 min)

Project: Real-time Facial Expression Recognition System  
Author: Donovan Ma  
Date: 2026-03-22  
Scope: Technical methods, full system design, limitations, and negative-result-driven fixes

---

## Slide 1 - Title and One-line Thesis (0:45)

### Title
Real-time Facial Expression Recognition via Teacher-Student Distillation and Dual-Gate Deployment Evaluation

### One-line thesis
In real-time FER, offline non-regression alone is not enough; checkpoint promotion must pass both offline gates and same-session replay gates.

### Talk roadmap
1. Why this problem is hard in deployment
2. System architecture and technical design
3. What worked, what failed, and how failures changed the protocol

---

## Slide 2 - Outline (0:30)

### Outline
1. Problem and constraints in real-time FER
2. End-to-end technical system design
3. Teacher-student training and calibration behavior
4. Dual-gate evaluation protocol
5. Negative results, limitations, and fix directions
6. Conclusion and takeaways

---

## Slide 3 - Problem Setting and Deployment Constraints (1:00)

### Practical problem
Offline FER benchmarks do not capture deployment requirements:
- temporal stability
- confidence calibration
- domain shift robustness
- CPU-latency constraints

### Deployment target in this project
- CPU-friendly inference with sub-40 ms tail latency (isolated model benchmark)
- stable frame-to-frame behavior for user-facing output
- safe checkpoint promotion policy

### Canonical label space
Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

---

## Slide 4 - Research Questions and Contributions (1:00)

### Research questions
- RQ1: Can KD/DKD improve lightweight student behavior and calibration?
- RQ2: How severe is domain shift from curated data to webcam deployment?
- RQ3: Can Self-Learning + NegL adapt safely without regressions?
- RQ4: What promotion protocol is valid when offline and deployment signals diverge?

### Core contributions
- Artifact-grounded teacher-student FER pipeline
- Dual-gate promotion protocol (offline non-regression + replay performance)
- Formal negative result catalog (NR-1 to NR-7)

---

## Slide 5 - End-to-End System Architecture (1:15)

### Pipeline
Raw datasets -> clean/validate manifests -> teacher training -> weighted teacher ensemble -> softlabel export -> student CE/KD/DKD training -> real-time inference -> dual-gate evaluation -> promote/reject

### Design principle
Every stage emits auditable artifacts:
- manifests
- checkpoints
- calibration files
- reliability metrics
- replay score outputs

### Visual
Use architecture figure: figures/fig0_pipeline_architecture.png

---

## Slide 6 - Data Engineering and Provenance (1:10)

### Data strategy
Multi-source training and stress evaluation across FERPlus, AffectNet, RAF-DB, ExpW, and FER2013 variants.

### Integrity controls
- CSV-manifest-first ingestion
- path and label validation
- SHA256 provenance snapshots

### Key numbers
- 466,284 validated rows
- 0 missing paths
- 0 invalid labels

### Why this matters
This prevents silent data faults and keeps all downstream comparisons reproducible.

### Visuals to place
- research/presentation/figures/figP1_data_provenance_flow.png
- figures/fig0_data_imbalance.png

---

## Slide 7 - Teacher Stack and Ensemble Design (Deep Technical) (1:20)

### Teacher backbones
- ResNet-18
- EfficientNet-B3
- ConvNeXt-Tiny

### Margin-based supervision (ArcFace-style)
- Teachers are trained with additive angular margin to improve class separation.
- Core term:

\[
\cos(\theta + m)
\]

### Ensemble fusion
- Weighted logit fusion: 0.4 / 0.4 / 0.2 (RN18 / B3 / CNXT)
- Mixed-domain selection evidence: test_all_sources accuracy 0.687, macro-F1 0.660

### Role in system
Teachers provide robust soft targets and reduce variance before student distillation.

### Visuals to place
- research/presentation/figures/figP2_teacher_ensemble_arcface.png

---

## Slide 8 - Student Distillation and Calibration Behavior (Most Technical) (1:30)

### Student backbone
MobileNetV3-Large (deployment-oriented efficiency).

### Distillation setup
1. CE baseline
2. KD (hard labels + soft teacher targets)
3. DKD (decoupled target and non-target distillation)

### KD softening

\[
p_T = \text{softmax}(z/T)
\]

### DKD decomposition

\[
L_{DKD} = \alpha L_{TCKD} + \beta L_{NCKD}
\]

### ECE metric

\[
ECE = \sum_{m=1}^{M} \frac{|B_m|}{n} \left|\operatorname{acc}(B_m) - \operatorname{conf}(B_m)\right|
\]

### Key calibration result
- CE TS ECE: 0.050
- KD TS ECE: 0.028
- DKD TS ECE: 0.027

### Insight
Calibration quality and macro-F1 are separable objectives; both must be tracked.

### Visuals to place
- research/presentation/figures/figP3_kd_dkd_decomposition.png
- figures/fig3_calibration_comparison.png

---

## Slide 9 - Real-time Inference and Stabilization Design (Technical) (1:10)

### EMA smoothing

\[
p_t = \alpha p_t^{raw} + (1-\alpha)p_{t-1}
\]

where \(p_t^{raw}\) is the current-frame probability vector.

### Hysteresis decision rule
- Prevent rapid class switching unless challenger confidence exceeds current class by a margin.
- Reduces flicker under detector jitter and motion noise.

### Deployment stability metric
- Jitter flips per minute
- Compare raw vs smoothed macro-F1 and accuracy

### Visuals to place
- figures/fig11_hysteresis_jitter.png
- figures/fig8_webcam_raw_vs_smoothed.png

---

## Slide 10 - Dual-Gate Promotion Protocol (1:20)

### Why single-gate is unsafe
Offline pass can still fail under replay-domain deployment behavior.

### Gate A: Offline non-regression
- eval-only and ExpW gates
- fail thresholds pre-registered

### Gate B: Same-session replay quality
- fixed labels
- fixed stabilizer settings
- A/B comparability against baseline

### Promotion rule
Promote only when both gates pass.

### Visual
Use figures/fig12_dual_gate_decision_flow.png

---

## Slide 11 - Negative Results That Changed the Method (1:55)

### NR-1 (most important)
Adapted candidate passed offline gate but failed replay quality gate.

Replay A/B (smoothed):
- Baseline: acc 0.588, macro-F1 0.525, minority-F1 0.161, jitter 14.86
- Adapted: acc 0.527, macro-F1 0.467, minority-F1 0.138, jitter 14.16

Meaning: slight jitter reduction did not compensate for major quality regression.

### NR-2
Head-only and BN-only adaptation failed offline safety gate (macro-F1 regression).

### NR-3 to NR-7 (compressed)
- NL/NegL screening unstable or no robust gain
- KD/DKD calibration gains without macro-F1 gains
- hard-gate decay concentrated in minority classes
- LP-loss showed no clear cross-domain macro-F1 win in short-budget screening

### Visual
Use figures/fig10_adaptation_ab.png and figures/fig13_replay_ab_delta.png

---

## Slide 12 - Limits Found and How We Fix Them (1:15)

### Limit 1: Offline metrics can mislead deployment
Fix: dual-gate protocol is now mandatory for promotion.

### Limit 2: Minority-class fragility (Fear/Disgust) under domain shift
Fix direction:
- targeted class-balanced replay buffers
- per-class gate tracking (not only global macro-F1)

### Limit 3: Adaptation instability on small webcam buffers
Fix direction:
- conservative update policies
- stricter entropy/confidence gating
- replay-first acceptance criteria

### Limit 4: Short-budget screening can miss long-horizon behavior
Fix direction:
- staged screening: quick triage then longer confirmatory runs for survivors

---

## Slide 13 - Runtime Evidence and Deployment Readiness (0:40)

### CPU benchmark evidence (isolated model runtime)
- Mean 23.01 ms
- P95 30.04 ms
- P99 34.97 ms
- Throughput 43.46 FPS
- ONNX size 16.05 MB

### Positioning
The model is lightweight and practical for edge-style deployment, with clear known failure boundaries.

---

## Slide 14 - Conclusion (0:35)

### Final message
This project delivers not only a FER model but a deployment-valid evaluation method.

### What is technically new in practice
- reproducible artifact-grounded pipeline
- calibration-aware student distillation analysis
- negative-result-driven dual-gate promotion logic

### Closing line
For real-time FER, reliable promotion policy is as important as model architecture.

---

## Slide 15 - Q&A Backup (optional)

### Backup topics if asked
- Why DKD calibration gain did not improve macro-F1
- Why replay A/B is stricter than live cross-run comparison
- Why negative results are treated as first-class evidence
- Next-step plan for minority-class robustness and fairness audit

---

## Presenter Notes (Quick Timing Guide)

- Slide 1: 0:40
- Slide 2: 0:20
- Slide 3: 0:55
- Slide 4: 0:55
- Slide 5: 1:15
- Slide 6: 1:10
- Slide 7: 1:20
- Slide 8: 1:30
- Slide 9: 1:10
- Slide 10: 1:20
- Slide 11: 1:55
- Slide 12: 1:15
- Slide 13: 0:40
- Slide 14: 0:35

Total (Slides 1-14): 15:00

Q&A strategy:
- Keep Slide 15 as backup only.
- If interrupted by questions before Slide 12, compress NR-3 to NR-7 into one sentence.
