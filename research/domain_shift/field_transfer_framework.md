# Field Transfer (Domain Shift) Research Framework — Real-time FER

Last updated: 2026-02-05

Goal: improve **real-time** FER quality under domain shift (ExpW + webcam) with an evidence-first workflow:

- every result backed by artifacts (JSON/CSV/MD)
- every risky adaptation guarded by an offline **eval-only** regression gate + rollback

This framework is intentionally broken into **4 experiments** (easy → research-y), each with clear deliverables.

---

## Experiment 1 — Domain shift dashboard (no training)

### Question
What is our current baseline behavior under domain shift (offline + live), and which failure modes dominate?

### Protocol
- Offline baselines:
  - Evaluate 3 student baselines (CE / KD / DKD or best-available) on:
    - ExpW test manifest
    - eval-only manifest (regression gate baseline)
- Live baselines:
  - Summarize all labeled webcam runs (`demo/outputs/*/score_results.json`).

### Metrics
- Offline: raw macro-F1, minority-F1 (lowest-3), raw/TS ECE and NLL
- Live: raw vs smoothed `macro_f1_present`, accuracy, minority-F1 (present), jitter flips/min

### Deliverables
- `outputs/domain_shift/webcam_summary.md`
- `outputs/domain_shift/domain_shift_score_v0.md` (fixed formula + weights)
- `outputs/evals/_compare_*_domainshift_expw_*.md` (existing pattern)

Why this matters: it prevents “improving” something without knowing the baseline.

Current KD baselines (2026-02-05; evidence-backed):

- KD baseline: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/best.pt`
- KD + LP-loss: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt`

---

## Experiment 2 — Temporal stabilization ablation (engineering, no training)

### Question
How much of the real-time failure is *model* vs *temporal smoothing policy*?

### Protocol
- Pick one labeled webcam session as the reference.
- Re-score the same session across stabilizer settings:
  - raw (no smoothing)
  - EMA sweep (`ema_alpha`)
  - vote window sweep (`vote_window`, `vote_min_count`)
  - hysteresis sweep (`hysteresis_delta`)

### Metrics
- Smoothed `macro_f1_present` / `minority_f1_present`
- Jitter flips/min
- Also track the raw metrics to ensure smoothing isn’t hiding a weak classifier.

### Deliverables
- `outputs/temporal/temporal_ablation_<run_stamp>.md`
- Decision: pick one “default stabilizer setting” used for all later live experiments.

---

## Experiment 3 — Streaming-safe test-time adaptation (TENT → SAR-lite)

### Question
Can we adapt online to the target stream **without collapse** and without regressing broad generalization?

### Methods to compare
- No adaptation (baseline)
- TENT-style entropy minimization (adapt normalization affine params only)
- SAR-lite (from Paper #2):
  - reliable-sample filter via entropy threshold $E_0$
  - sharpness-aware update (optional; SAM-style) on the entropy objective
  - model recovery/reset on collapse signal

### Target streams
- ExpW simulated streaming batches (deterministic; faster debugging)
- Webcam buffer adaptation (real deployment-like stream)

### Streaming constraints (design requirements)
- Update only a **small parameter subset** first (normalization affine)
- Update in **micro-batches** (e.g., 16–64 frames) rather than per-frame
- Always run offline gate + rollback after each adaptation chunk

### Metrics
- Offline gate (must-not-regress): eval-only macro-F1 (and minority-F1)
- Offline target: ExpW macro-F1 / minority-F1
- Live: smoothed `macro_f1_present` and flips/min (using default stabilizer from Exp 2)
- Stability: collapse detection rate, recovery/reset events

### Deliverables
- Per-run metadata (base ckpt, params unfrozen, lr/steps/batch, entropy threshold, recovery settings)
- `outputs/evals/students/<run>/reliabilitymetrics.json` for eval-only and ExpW
- `demo/outputs/<run_stamp>/score_results.json` for the live session

---

## Experiment 4 — Safe self-learning + NegL (research-y, 1 iteration)

### Question
Can we improve target performance using unlabeled target data while controlling confirmation bias?

### Method
- Buffer construction (webcam frames or ExpW train stream)
- Update rule:
  - high confidence → pseudo-label update (self-training)
  - medium confidence → NegL (discourage likely-wrong classes)
  - low confidence → skip
- Safety rails:
  - eval-only regression gate + rollback
  - logging of applied fractions (how often each rule triggers)

### Metrics
- Same as Experiment 3 + additional logging:
  - pseudo-label rate
  - NegL applied fraction

### Deliverables
- A single compare table vs baseline and vs Exp 3 best method
- Clear “pass/fail” statement based on the offline gate

---

## What counts as a “win”

A change is acceptable only if:

- Offline eval-only gate: macro-F1 does **not** regress meaningfully (define a threshold before the run)
- ExpW: minority-F1 improves, or macro-F1 improves without minority collapse
- Live: smoothed `macro_f1_present` improves **and** flips/min does not get worse beyond an agreed cap

(Exact thresholds are specified in the evaluation plan.)
