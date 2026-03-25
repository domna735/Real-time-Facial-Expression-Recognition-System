# NL / NegL Study Notes (Domain Shift)

This file links the older NL/NegL offline study to the new domain-shift adaptation plan.

## Context split

- Offline NL/NegL experiments (KD/DKD pipeline): see existing folder `research/nl_negl_plan/`.
- Domain shift adaptation (webcam): this folder.

## Key lesson carried forward

- Gating matters more than the loss name.
- If a signal is applied too rarely, it cannot help.
- If applied too often, it can destabilize and harm offline generalization.

## What is new here

- “Self-learning + NegL” is used for **target-domain adaptation** (webcam), not as a general KD/DKD improvement.
- We add strict acceptance gates (webcam-mini + offline eval-only + stability) before promoting.
