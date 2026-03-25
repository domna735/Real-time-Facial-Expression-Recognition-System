# NegL Rules (Domain Shift)

Purpose: make NegL safe for webcam adaptation.

## When NegL is allowed

- Only on **medium-confidence** samples (confidence band), not on low-confidence noise.
- Prefer applying NegL only when the prediction is stable (or stable-by-smoothing).

## What NegL should do

- Encourage the model to **not** commit probability mass to obviously wrong classes.
- Avoid acting like “anti-learning” for minority classes.

## Default negative class selection

Start simple:
- pick the 1–2 classes with **lowest predicted probability** as negatives.

Avoid initially:
- choosing the top-2 confusable classes as negatives (can hurt recall and destabilize).

## Safety knobs

- Keep `negl.weight` small (e.g., 0.02–0.05).
- Track `applied_frac` (how often NegL is used).

## Acceptance gates (must pass)

- Webcam-mini improves (smoothed macro-F1_present and/or stability)
- Offline eval-only does not regress beyond tolerance
- Flip-rate/jitter non-worse
