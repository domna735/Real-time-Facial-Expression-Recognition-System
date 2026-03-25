# Part 1 — KD/DKD Weaknesses vs NL/NegL Improvement Opportunities

This file turns your notes into a clean mapping from observed KD/DKD issues → what NL/NegL can realistically improve → what signals we should measure.

## Mapping table (condensed)

| Aspect | KD weakness | DKD weakness | NL improvement opportunity | NegL improvement opportunity | Real metrics / deployment signal |
|---|---|---|---|---|---|
| Non-target knowledge retention | Coupled loss suppresses non-target signals; high-confidence samples downweighted | Improved but still logit-only | Memory retains inter-class structure; adaptive momentum prevents drowning | Add repulsion around wrong classes for sharper boundaries | Minority-F1 ↑, Macro-F1 stable |
| Signal balance / hyperparameter sensitivity | Cannot independently tune target/non-target weights | α/β sensitive; easily unstable | Difficulty/consistency gates dynamically adjust weighting | Control negative-sample ratio to avoid over-penalty | Gradient spikes ↓, training stability ↑ |
| Catastrophic forgetting | Minority classes drift after long training | Same drift risk | Memory retains early minority signals; reduces drift | Works with NL to protect minority classes | Minority-F1 no degradation after ~epoch 10 |
| Calibration issues | Teacher overconfident; student follows mistakes | Teacher bias may amplify | Consistency check dampens overconfident teacher guidance | Penalize confident mistakes; reduce ECE/NLL | ECE ↓ ≥20%, NLL ↓ ≥10% |
| Real-time stability | Output jitter; frequent label flips | Same lack of temporal consistency | Temporal memory smooths outputs; reduces flip rate | Safer thresholds reduce wrong triggers | Flip-rate ↓, user-perceived stability ↑ |
| Domain shift / noise | Sensitive to webcam noise/lighting | Same sensitivity | Adaptive momentum improves robustness | Synthetic noise / complementary labels improve robustness | More stable under occlusion/lighting |

Notes:
- We keep the claims measurable: if a row is not measurable, it becomes “TBD / future work”.
- For this repo, KD/DKD already improved calibrated ECE after temperature scaling, while Macro-F1 uplift was limited.
