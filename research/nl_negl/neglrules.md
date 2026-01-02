# NegL Rules (Scaffold)

Purpose: define how Negative Learning (NegL) is applied so it improves calibration (ECE/NLL) without harming minority recall.

## Core idea
- NegL uses *complementary labels*: instead of saying "this is class y", it says "this is **not** class k".
- We apply NegL selectively (gating) to avoid over-penalizing ambiguous or minority samples.

## Proposed rules
- **Uncertainty gate**
  - Apply NegL only when the student prediction entropy is above a threshold OR teacher/student consistency is low.
- **Class-aware ratio**
  - Minority classes use lower NegL ratio to protect recall.
  - Majority classes can use higher NegL ratio to reduce overconfidence.
- **Teacher-guided negatives (later)**
  - Sample negative class from a teacher confusion matrix so the repulsion focuses on realistic confusions.

## Metrics to watch
- Calibration: ECE, NLL
- Reliability: Brier score
- Performance: Macro-F1, Minority-F1
- Deployment: flip-rate / jitter and unknown/unstable rate

## Status
- This file is a scaffold. Training integration will come next (NL/NegL combined training script).