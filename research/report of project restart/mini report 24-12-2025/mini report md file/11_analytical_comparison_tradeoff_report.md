# 11) Analytical Comparison (Trade-off) Report（分析性對比／取捨分析）

Date: 2026-02-09

## Goal
This mini-report clarifies what “comparison” means for this FYP: an **analytical comparison** of trade-offs and constraints, not a win/lose race against SOTA papers.

## Core message (what the supervisor wants)
- Explain **where and why** our system underperforms some paper-reported numbers.
- Explain whether the performance gap is **reasonable** under our real-time and deployment constraints.
- Use papers as **reference points** to interpret differences, not as a benchmark we must “beat”.

## Our system vs typical paper setting (high-level)
Our system is deployment-oriented:
- Student: MobileNetV3-Large (lightweight)
- CPU real-time target (typical goal: 25–30 FPS)
- Multi-threaded pipeline
- Temporal smoothing + hysteresis to reduce flicker
- Multi-source noisy dataset (466k validated rows)
- Domain shift (webcam) is explicitly in scope
- Reproducible artifacts (manifests + checkpoints + metrics JSON)

Many papers are benchmark-oriented:
- Heavy backbones (ResNet-50 / ViT / Swin, etc.)
- GPU inference
- Single curated dataset protocols
- No latency/stability/deployment constraints

## Why direct 1:1 comparison is often invalid
A fair “paper-comparable” claim requires matching:
- Official dataset split/protocol (and any test-time augmentation like 10-crop)
- Label mapping (7-class vs 8-class vs compound)
- Metric definition (accuracy vs macro-F1, averaging method)

Repo evidence that enforces this:
- One-page comparability table: `research/paper_vs_us__20260208.md`
- Protocol extraction notes: `research/paper_metrics_extraction__20260208.md`
- Fair-compare rules: `research/fair_compare_protocol.md`

## Where we lose (and why it is reasonable)
1) Model capacity trade-off
- Heavy models usually win raw accuracy/macro-F1.
- We choose MobileNetV3 for CPU real-time; some accuracy loss is expected.

2) Dataset difference (mixture + noise)
- Many papers use a single curated dataset and its official split.
- Our evaluation includes mixed-source and domain-shift gates; macro-F1 can be lower due to label noise and distribution mismatch.

3) Deployment constraints change the objective
- We explicitly care about stability (flicker), calibration, and webcam behavior.
- These are often not optimized/reported in offline-only papers.

4) Domain shift is in-scope
- Webcam shift creates predictable drops; our work treats it as a first-class target.

## What we optimize that papers often do not
- Calibration (ECE/NLL; temperature scaling) stored per run.
- Real-time stability (raw vs smoothed metrics; jitter/flip-rate) stored per demo run.
- Reproducibility (manifests and metrics artifacts for every claim).

## One-page “overall sanity” snapshot (evidence-backed)
To avoid over-claiming comparability while still answering “is the model overall OK?”, we provide a consolidated table across four hard gates:
- `outputs/benchmarks/overall_summary__20260208/overall_summary.md`

## Where this is integrated in the final report
- `research/final report/final report.md` → Section **6.1** (Analytical comparison vs papers)
