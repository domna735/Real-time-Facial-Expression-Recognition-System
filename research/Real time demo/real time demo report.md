# Real-time Demo Report

This document records **deployment-facing observations** from the webcam demo (`demo/realtime_demo.py`).

Important note (scope):

- Items marked **subjective** are based on human observation during live use.
- Numeric claims should still be treated as **artifact-backed only** (e.g., `demo/outputs/*/score_results.json`, `outputs/evals/students/*/reliabilitymetrics.json`).

---

## Feb 2026 update: subjective checkpoint preference (webcam)

### Compared checkpoints

The following student checkpoints were tested in the real-time demo:

- CE (Dec-2025 main run): `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`
- KD + LP-loss (Feb-2026 screening): `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt`
- DKD (Jan-2026 best-of-screening candidate): `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602/best.pt`

### Subjective finding (2026-02-06)

- **CE** feels **most stable** (fewer noticeable label flickers) and has the **best perceived accuracy** in live webcam use, compared to the KD+LP and DKD checkpoints.

This is a deployment-facing observation and should be validated with a controlled labeled session + scoring artifacts.

---

## Why this can happen (detailed, evidence-aware hypotheses)

Real-time “stability” is not the same objective as offline macro-F1. In this repo’s demo, stability is shaped by:

1) **What the demo uses to decide when to switch labels**

- `demo/realtime_demo.py` performs temporal smoothing and switch suppression using *probabilities*:
  - EMA smoothing over probability vectors (`ema_alpha`)
  - Hysteresis (`hysteresis_delta`): only switch if the new top probability exceeds the current label by `delta`
  - Optional voting window

Even when two checkpoints have similar offline macro-F1, they can produce different **probability margins** frame-to-frame, which directly changes how often hysteresis allows switching.

2) **Temperature scaling is applied during the demo**

- For student checkpoints, the demo applies `logits / T` before softmax.
- If `--temperature` is not explicitly provided, the demo attempts to read `calibration.json` next to the checkpoint.

Implication:

- Temperature scaling does not change argmax labels, but it *does* change probability sharpness.
- That, in turn, changes EMA + hysteresis behavior (how quickly the displayed label reacts vs. sticks).

So two checkpoints can feel different in “stability” even if their top-1 predictions are similar, because their calibrated probability geometry differs.

3) **KD/DKD are optimized to match teacher distributions, not human-perceived stability**

In KD/DKD, the student is trained to match teacher logits/probabilities. If the teacher ensemble is uncertain or biased for webcam-like frames (lighting/pose/motion), the student can inherit:

- smaller margins between the top-2 classes (more “near-ties”)
- stronger confusion between common webcam emotions (often Neutral/Happy/Sad)

Near-ties are exactly what produces visible flicker in a real-time UI if the face detector crop shifts slightly across frames.

4) **Face detection + crop jitter interacts with model sensitivity**

On webcam, per-frame face boxes can shift (even slightly). A checkpoint that is more sensitive to crop/illumination changes will show more probability variance, even if its offline metrics are strong.

CE can feel more stable if its learned decision boundary is less sensitive to these small perturbations on the user’s camera domain.

5) **Domain shift can flip which checkpoint is “best”**

Offline tables (HQ-train, eval-only, ExpW) are different distributions than the user’s live webcam stream. It is plausible for CE to be strongest on the user’s webcam domain even if KD/DKD show better calibration after temperature scaling on offline splits.

---

## How to validate objectively (recommended next measurement)

To turn the subjective observation into an artifact-backed claim:

1) Record a **single labeled webcam session** (same person, lighting, background) and replay/score it with each checkpoint.
2) Keep demo parameters fixed across checkpoints:
	- detector type
	- `ema_alpha`, `hysteresis_delta`, vote settings
	- CLAHE on/off
3) Force temperature to be comparable:
	- either set the same `--temperature 1.0` for all (no calibration), or the same fixed `T` for all.
4) Save outputs under `demo/outputs/<run_id>/` and compute:
	- raw vs smoothed accuracy/macro-F1
	- jitter flips/min

This produces the repo-standard artifacts (`per_frame.csv`, `score_results.json`) so the stability claim becomes traceable.

