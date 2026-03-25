# Part 2 — Research Framework (Hypotheses → Methods → Evidence)

This matches your “Step / Hypothesis / Method / Evidence / Reflection / Conclusion” structure.

## Baseline context (already done)
- Student: MobileNetV3-Large (`mobilenetv3_large_100`)
- Baseline pipeline: CE → KD → DKD
- Evidence source: `outputs/students/*/reliabilitymetrics.json`, `calibration.json`, `history.json`

## Step plan

### Step 1 — Verify NL (catastrophic forgetting)
- Hypothesis: NL prevents long-run minority drift / catastrophic forgetting.
- Method:
  - Implement NL gating *only* in student training loop (no NegL yet).
  - Compare short vs long schedules (e.g., CE 10 epochs vs CE 30 epochs; and KD/DKD long runs if needed).
- Evidence:
  - Minority-F1 over epochs (per-class F1 curves)
  - Macro-F1 stability over time
  - Train/val loss curves
- Pass condition (initial): long-run minority-F1 does not degrade after epoch ~10.

### Step 2 — Verify NegL (calibration)
- Hypothesis: NegL reduces overconfident wrong predictions, improving ECE/NLL.
- Method:
  - Add NegL as an auxiliary loss with complementary labels.
  - Use uncertainty gating (only apply NegL when entropy/high uncertainty or when teacher-student inconsistency is high).
- Evidence:
  - ECE, NLL (raw + temperature scaling)
  - Confident error rate
- Pass condition (initial): ECE ↓ ≥20% or NLL ↓ ≥10% vs baseline under same eval protocol.

### Step 3 — Expand data (domain gap)
- Hypothesis: webcam-style augmentation + minority expansion improves robustness.
- Method:
  - Add webcam-like aug policy (motion blur, brightness/contrast, occlusion cutout, compression artifacts).
  - Keep manifest-driven reproducibility.
- Evidence:
  - Macro-F1, Minority-F1 on mixed-source test and (optionally) webcam logs.

### Step 4 — Keep teacher ensemble (stable softlabels)
- Hypothesis: ensemble provides best stable KD targets.
- Method:
  - Keep current winning 3-teacher softlabels run as fixed target.

### Step 5 — Student choice
- Hypothesis: MobileNetV3 is best tradeoff (calibration, latency).
- Method:
  - Keep MobileNetV3 unless there’s clear need to change.

### Step 6 — Combined test
- Hypothesis: NL + NegL + KD + data aug improves both F1 + calibration.
- Method:
  - Run combined pipeline only after isolated ablations pass.

### Step 7 — Observe results
- If ECE improves but F1 is flat, still count as success for deployment reliability.

### Step 8 — Next: hard-sample mining / per-class calibration
- Park as optional; only do after Step 1–6.
