# 6) Basic Demo Result Report（基本 Demo 結果報告）
Date: 2025-12-24

## Feb 2026 addendum pointers (new evidence)

For the Feb-2026 evidence-backed benchmark suite exports, dataset diagnostics, and paper-style comparison notes, see:

- `research/final report/final report.md` → Sections **9.3.5** and **9.3.6**
- Analytical comparison (trade-off framing; supervisor intent): `research/final report/final report.md` → Section **6.1**
- One-page paper table: `research/paper_vs_us__20260208.md`
- Paper protocol/metric notes: `research/paper_metrics_extraction__20260208.md`

## Demo objective
Demonstrate real-time facial expression recognition (FER) from webcam/video using the trained model pipeline.

## Demo entrypoint
- Main demo script: `demo/realtime_demo.py`
- Related inference helper: `scripts/realtime_infer_arcface.py`

## What the demo does
- Captures frames from a camera/video.
- Detects a face (supported detectors in code).
- Applies preprocessing consistent with training (includes CLAHE when enabled).
- Runs FER model inference.
- Applies temporal smoothing to stabilize predictions.
- Writes a CSV log of results.

## Inputs/outputs
- Input: webcam feed or video path (configured by args in demo script)
- Outputs:
  - Per-frame predictions and timestamps saved into the demo output folder (`demo/outputs/`)
  - Optional screenshots / debug visualizations depending on args

## Current results snapshot
- This mini-report documents the **demo pipeline availability and logging outputs**.
- Quantitative demo metrics (FPS, latency, flip-rate) are **measurable** from:
  - runtime logs
  - generated per-frame CSV
  - and should be reported after a dedicated demo run on the target machine.

TBD note (requires a real demo run):
- This report intentionally does not claim FPS/latency/flip-rate numbers because no demo CSV from a timed run is attached here.

## How to validate (quick checklist)
- Demo runs without crashing on Windows.
- Face box is detected and displayed.
- Emotions update smoothly (not flickering every frame).
- CSV log file is created and contains timestamp + probabilities/label.

## Next steps
- Run the demo for 2–3 minutes, then compute:
  - average FPS
  - mean/median inference latency
  - label flip-rate per minute
- Add a small script to summarize the CSV into these KPIs.

---

## Feb 2026 addendum (domain shift + quantitative demo scoring)

Jan–Feb 2026 work added a reproducible *labeled* webcam scoring protocol (raw vs smoothed metrics, plus jitter flip-rate).

Evidence artifacts:

- Demo scoring outputs: `demo/outputs/*/score_results.json`
- Summary note: `research/Real time demo/real time demo report.md`

Interpretation note:

- Offline “best” checkpoints do not always feel best in real-time; probability margins interact with smoothing/hysteresis. Therefore real-time evaluation must be reported separately from offline benchmarks.
