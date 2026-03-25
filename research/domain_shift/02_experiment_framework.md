# 02 — Experiment Framework

## Goal

Improve webcam-domain performance while keeping offline regression within tolerance.

## Primary hypotheses

H1) Small, safe adaptation (BN-only or head-only) improves webcam-mini metrics.

H2) Offline regression mainly comes from: small correlated buffer + ambiguous frames + style overfitting.

H3) NegL on medium-confidence frames reduces pseudo-label collapse and improves stability/calibration.

## Experiment units

Each experiment is a tuple:
- Base checkpoint
- Buffer manifest (how it was built + sampling params)
- Update policy (`--tune head` / `--tune bn` / ...)
- LR/epochs and any NegL config

## Minimal ablation grid (keep small)

1) Update policy: `head` vs `bn`
0) Base checkpoint: KD baseline vs KD+LP (optional pre-step; measure before any adaptation)
2) Update size: LR {`5e-6`, `1e-5`} × epochs {1}
3) Buffer sampling: stable-only on/off; min-frame-gap {10, 20}; cap-per-class {50, 100, 200}
4) NegL: off vs on (medium-confidence only)

## Evidence checklist (must log)

- Webcam-mini:
  - `metrics.smoothed.macro_f1_present`, `metrics.smoothed.accuracy`
  - flip-rate/jitter measures (from scorer)
- Offline eval-only:
  - accuracy, macro-F1, per-class F1
  - calibration (ECE/NLL + TS variants)

## Decision rule

Only accept a checkpoint if webcam improves AND offline does not regress beyond tolerance AND stability is non-worse.
