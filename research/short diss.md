# Discussion Plan (next step) — Supervisor Meeting

Date: 2026-01-21

## 1) Meeting goal (what I want to decide today)

1) Confirm the primary objective for the next 1–2 weeks:
	 - Deployment readiness (CPU real-time correctness + FPS) vs
	 - Domain-shift robustness (ExpW/webcam-like) vs
	 - Balanced (do minimum to unblock both)

2) Agree on a **fair live evaluation protocol** to compare with offline metrics.

3) Choose the next single best intervention track:
	 - Stabilization tuning
	 - Pipeline parity fixes
	 - Domain-shift training (augmentations / long-tail)

## 2) Current status (quick snapshot)

- CPU/device forcing is supported in the real-time demo via `--device {auto,cpu,cuda,dml}`.
- Baseline domain-shift (ExpW) compare table exists and shows a clear robustness gap.
- Live scoring is upgraded to compute both `metrics.raw` and `metrics.smoothed` and also `macro_f1_present`.

Key artifacts to show in meeting:

- Plan: `research/plan of next step/plan.md`
- ExpW domain-shift baseline table: `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md`
- Week log with decisions: `research/process_log/Jan process log/Jan_week4_process_log.md`

## 3) Key problem to discuss (why live differs from offline)

Observation:

- “Real-time macro-F1” can look much worse than offline evaluation.

Root causes (most likely):

1) **Metric artifact (live sessions)**
	 - Many demo runs have no manual labels → F1 cannot be computed.
	 - Live sessions often contain only 1–2 emotions → “macro-F1 over all 7” becomes artificially low.
	 - Use `macro_f1_present` for live sessions.

2) **Stabilization changes the output**
	 - Demo `pred_label` is a stabilized label (EMA/vote/hysteresis).
	 - Offline evaluation is raw logits → not directly comparable.

3) **Pipeline parity / domain shift**
	 - Face detector crop policy, resize, normalization, CLAHE, RGB/BGR handling can shift distributions.
	 - Webcam/ExpW images have lighting/blur/compression differences.

4) **Regression issue (offline distillation vs CE)**

	 - Interim report (Dec 25, 2025) shows: CE achieves macro-F1 0.741952, while KD/DKD do not surpass CE in the first run.
	 - KD/DKD improve calibration substantially (temperature-scaled ECE ~0.027 vs CE ~0.050), but raw accuracy/macro-F1 regress.
	 - This implies our current KD/DKD settings and/or training length may be optimizing calibration at the cost of raw macro-F1.

## 4) Decision rule (agreed next step selection)

We should decide based on 1 labeled live run scored with `--pred-source both`:

- If `metrics.raw` >> `metrics.smoothed`:
	- Tune stabilization first (no retraining, high ROI).

- If both are low but offline evaluation is strong:
	- Fix pipeline parity (crop/normalize/detector policy) before retraining.

- If both are low and ExpW is low:
	- Proceed with domain-shift training steps (augmentations → long-tail → target-aware fine-tune).

## 5) Proposed actions for next 3–7 days (concrete checklist)

### A) Collect a fair live baseline (required)

Goal: one 2–3 minute CPU-forced session with deliberate manual labels across 4–5 emotions.

Commands:

```powershell
python demo/realtime_demo.py --model-kind student --device cpu
python scripts/score_live_results.py --per-frame demo/outputs/<run_stamp>/per_frame.csv --out demo/outputs/<run_stamp>/score_results.json --pred-source both
```

Deliverable:

- One `demo/outputs/<run_stamp>/score_results.json` with `macro_f1_present` + raw/smoothed comparison.

### B) Stabilization tuning (only if raw > smoothed)

Goal: reduce over-smoothing while keeping the demo stable.

Plan:

- Sweep 2–3 settings (small grid): EMA alpha / vote window / hysteresis thresholds.
- Re-score each run and compare:
	- `metrics.smoothed.macro_f1_present` vs baseline
	- flip-rate / unstable segments (if reported)

Deliverable:

- A small table of “stability vs correctness” trade-offs.

### C) Pipeline parity checks (only if live raw is unexpectedly low)

Checklist:

- Confirm face crop size and preprocessing match training/eval assumptions.
- Verify color space handling and normalization.
- Check CLAHE usage consistency (on/off and parameters).
- If needed: save a few demo crops and run offline evaluator on them to isolate pipeline vs model.

Deliverable:

- A short note: “pipeline mismatch found / not found” + what changed.

### D) Domain shift improvements (ExpW-first)

If ExpW remains weak after A/B/C:

1) Robustness augmentations (photometric + blur + compression)
2) Long-tail improvement (pick ONE): class-balanced loss OR focal OR logit adjustment
3) Target-aware fine-tuning (ExpW-heavy mix, optional KD/DKD)

Deliverable:

- New compare table under `outputs/evals/` showing minority-F1 improvement vs baseline.

## 6) Questions to ask supervisor (to unblock decisions)

1) What matters more for the next milestone: demo correctness on CPU vs ExpW robustness?
2) What live KPI should we report (accuracy vs macro-F1-present vs minority-F1-present)?
3) What is an acceptable stability vs responsiveness trade-off for the demo?
4) Any constraint on training budget (GPU hours) and acceptable complexity (new loss functions vs aug only)?

5) For deployment: do we prefer **higher raw macro-F1** (CE-like) or **better calibrated confidence** (KD/DKD-like), or do we need both?

6) What are the deployment acceptance thresholds (example):
	- CPU FPS target (e.g., >= 15/20/25 FPS)?
	- Max end-to-end latency (ms) and max “flip-rate” allowed?
	- Do we report stability metrics (flip-rate, dwell time) as KPIs alongside macro-F1-present?

7) Should the demo decision output be:
	- raw logits argmax, or stabilized `pred_label`, or both with a UI toggle?
	- add an “unknown/uncertain” state using calibrated confidence thresholding?

8) For domain shift: should we prioritize ExpW-full vs ExpW-HQ as the main target, and what is the allowed trade-off between in-domain accuracy and robustness?

9) Reproducibility / artifact policy: do we freeze one “deployment checkpoint” and one “research checkpoint”, and track them with provenance (seed, manifests, preprocessing flags) to avoid silent regressions?

## 7) Risks + mitigation

- Risk: live labeling is inconsistent / too short → metrics unstable.
	- Mitigation: standard 2–3 minute protocol, cover 4–5 emotions.

- Risk: improvements help ExpW but not real webcam.
	- Mitigation: build a small “webcam-mini” labeled set and track it alongside ExpW.

- Risk: spending time retraining before fixing pipeline parity.
	- Mitigation: enforce the decision rule (raw vs smoothed vs offline).

## 8) What I need (resources)

- 30–60 minutes for one labeled live session + 2–3 stabilization sweeps.
- If training is approved: GPU time for 1–2 controlled runs (augmentations first, then one long-tail variant).

