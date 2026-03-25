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

## Slide 5 - End-to-End System Architecture (1:30)

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

## Slide 6 - Data Engineering and Provenance (1:30)

### Data strategy
Multi-source training and stress evaluation across FERPlus, AffectNet, RAF-DB, ExpW, FER2013 variants.

### Integrity controls
- CSV-manifest-first ingestion
- path and label validation
- SHA256 provenance snapshots

### Key numbers
- 466,284 validated rows
- 0 missing paths
- 0 invalid labels

### Why this matters
It prevents silent data faults and makes all downstream comparisons reproducible.

---

## Slide 7 - Teacher Stack and Ensemble Design (1:30)

### Teacher backbones
- ResNet-18
- EfficientNet-B3
- ConvNeXt-Tiny

### Teacher training method
ArcFace-style margin training for stronger class separation.

### Ensemble fusion
Weighted logit fusion selected on mixed-domain benchmark:
- RN18/B3/CNXT = 0.4 / 0.4 / 0.2
- test_all_sources accuracy = 0.687
- test_all_sources macro-F1 = 0.660

### Role in system
Teachers provide robust soft targets for student distillation.

---

## Slide 8 - Student Distillation and Calibration Behavior (1:45)

### Student backbone
MobileNetV3-Large (deployment-oriented efficiency).

### Training stages
1. CE baseline
2. KD (hard labels + soft teacher targets)
3. DKD (decoupled target and non-target distillation)

### Key result
Calibration improved strongly while macro-F1 did not exceed CE:
- CE TS ECE: 0.050
- KD TS ECE: 0.028
- DKD TS ECE: 0.027

### Interpretation
Calibration quality and decision-boundary quality are separable objectives and must both be evaluated.

---

## Slide 9 - Real-time Inference and Stabilization Design (1:15)

### Inference path
YuNet detection -> preprocessing (optional CLAHE) -> ONNX student inference -> temperature scaling -> EMA smoothing -> hysteresis decision

### Stabilization controls
- ema_alpha
- hysteresis_delta
- optional voting policy

### Deployment metrics used
- raw vs smoothed accuracy/macro-F1
- jitter flips per minute

### Visual
Use figures/fig11_hysteresis_jitter.png and figures/fig8_webcam_raw_vs_smoothed.png

---

## Slide 10 - Dual-Gate Promotion Protocol (1:30)

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

## Slide 11 - Negative Results That Changed the Method (2:15)

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

## Slide 12 - Limits Found and How We Fix Them (1:30)

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

## Slide 13 - Runtime Evidence and Deployment Readiness (0:45)

### CPU benchmark evidence (isolated model runtime)
- Mean 23.01 ms
- P95 30.04 ms
- P99 34.97 ms
- Throughput 43.46 FPS
- ONNX size 16.05 MB

### Positioning
The model is lightweight and practical for edge-style deployment, with clear known failure boundaries.

---

## Slide 14 - Conclusion (0:45)

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

- Slides 1-4: 3:15
- Slides 5-10: 8:30
- Slides 11-12: 3:45
- Slides 13-14: 1:30
- Total: ~17:00 if full detail

To fit strict 15:00:
- shorten Slide 6 by 20s
- shorten Slide 8 by 25s
- shorten Slide 11 by 75s

Target delivery: 14:45 to 15:00 plus Q&A
