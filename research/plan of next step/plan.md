# Plan (next step): CPU readiness + domain-shift robustness

Date: 2026-01-18

## Status (as of 2026-02-13)

Done (evidence-backed):

- FC2 (FER2013 official split + ten-crop): protocol-aware single-crop + ten-crop evaluation exists and is summarized in `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`.
- Paper-side comparability support: one-page table `research/paper_vs_us__20260208.md` + protocol extraction notes `research/paper_metrics_extraction__20260208.md` + training/protocol checklist `research/paper_training_recipe_checklist__20260212.md`.
- Offline benchmark suite + Week-2 diagnostics: `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv` and `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`.
- Report integration: main professor-facing report is maintained in `research/final report/final report version 2.md` (protocol-aware FER2013 section + appendices for evidence inventories).
- Domain-shift webcam scoring artifacts exist under `demo/outputs/*/score_results.json`.

In progress / still missing for deployment readiness:

- Timed CPU KPI run (FPS/latency) with a labeled session and stored summary.

Next (highest ROI):

1) Run one timed CPU demo session + scoring (Phase 1, Section 2.1).
2) Keep using eval-only + ExpW as safety gates before promoting any adaptation.


---

## Feb 2026 addendum — Fair paper comparison + anti-forgery evidence rules

Date: 2026-02-11

This addendum updates the next-step plan based on supervisor guidance:

1) Continue improving **fair comparison** by using **official** datasets/splits whenever papers use them.
2) Even if results are “close”, analyze **why there is a gap** (protocol + model + training settings).
3) Consider that gaps may come from **parameters / hyperparameters / evaluation protocol** differences.

## A0) Anti-forgery evidence policy (must-follow)

Goal: make it impossible to accidentally (or ambiguously) report numbers.

Rule: every reported metric must be backed by a **single on-disk artifact** and must declare:

- **What is being evaluated:** model checkpoint path.
- **What data:** manifest path.
- **Which split:** `train` / `val` / `test` (or `PublicTest` / `PrivateTest` for official FER2013), plus **n**.
- **What protocol:** single-crop vs ten-crop (and any other TTA).
- **Where the metrics live:** the exact `reliabilitymetrics.json` path (and any summary MD/CSV generated from it).

Hard rule:

- Never describe a validation metric as a “test” metric.
- Never compare to a paper number unless split + protocol are explicitly marked (`Comparable: Yes/Partial/No`).

Recommended reporting template (copy/paste into logs and report):

```text
Model: <checkpoint path>
Manifest: <manifest path>
Split: <split> (n=<rows>)
Protocol: <single-crop|ten-crop|...>
Metrics: <path/to/reliabilitymetrics.json>
Comparable: <Yes|Partial|No> (reason: ...)
```

## A1) Make offline train/val/test definitions explicit

This repo uses multiple evaluation regimes that are not interchangeable:

- **Training-time validation (in-distribution):** the `val` split used during training (e.g., Stage-A teacher val; HQ-train student val). This measures *in-domain* behavior and is useful for model selection, not for paper comparison.
- **Offline gates (deployment-aligned stress tests):** eval-only, ExpW, FER2013-uniform-7, mixed-source gates. These are designed to expose domain shift and label-noise fragility.
- **Paper-comparison anchors:** official splits when available (e.g., FER2013 PublicTest/PrivateTest from `fer2013.csv`).

When writing results:

- Always separate blocks by regime (train-val vs gates vs official-split).
- Always include manifest + split + n + protocol + artifact path.

## 0) Context

- Target dev machine CPU: 13th Gen Intel(R) Core(TM) i9-13900HX (2.20 GHz)
- Deployment goal: real-time FER demo that runs reliably on CPU (and optionally GPU), with stable predictions.
- Research goal: improve robustness under domain shift (e.g., ExpW / webcam-like conditions), with minority-F1 as the primary metric.

## 1) What success looks like (acceptance criteria)

### A. CPU usability (real-time demo)

Minimum:

- Demo runs end-to-end on CPU with student checkpoint (no crashes).
- Logs are produced in `demo/outputs/<run_stamp>/`.
- “Correct result” is validated using demo logs:
  - Run `scripts/score_live_results.py` on `per_frame.csv` with some manual labels.
  - Protocol-lite accuracy is not degenerate (e.g., not near 0 with stable manual segments).
  - Also compare `metrics.raw` vs `metrics.smoothed` (raw argmax(probs) vs stabilized label) to detect whether smoothing is hurting accuracy/F1.

Target:

- Stable real-time experience at 224×224 face crops on CPU:
  - End-to-end throughput: >= 20 FPS sustained
  - Median per-frame latency: <= 50 ms (measured from demo logs)
  - Protocol-lite accuracy (from `score_results.json`) target >= 0.80 on a short labeled session

### B. Domain shift (robustness)

Minimum:

- A reproducible domain-shift evaluation table exists for:
  - at least 3 student checkpoints (CE / KD / DKD baselines)
  - evaluated on ExpW (or another target manifest)

Target:

- Improve minority-F1 on the target evaluation by a meaningful margin while keeping overall macro-F1 reasonable.
- Do not regress calibration badly after temperature scaling (track `temperature_scaled.nll` and `temperature_scaled.ece`).

## 2) Phase 1 — Measure before changing anything (1–2 days)

### 2.1 CPU demo benchmark (student)

Goal: measure real-world CPU performance using the real-time pipeline (camera + face detection + preprocessing + inference + stabilization + logging).

Actions:

1. Run the demo with the student model (auto-pick best student) on CPU:

```powershell
python demo/realtime_demo.py --model-kind student --device cpu
```

1. During the run, add **deliberate manual labels** for a fair live baseline:

    - Aim for a 2–3 minute session.
    - Try to cover 4–5 emotions (short segments is OK).
    - Avoid runs where only 1 emotion appears (macro metrics become misleading).

1. Record the output folder created under `demo/outputs/` and keep the CSV logs.
1. Score stability + protocol-lite accuracy from the produced logs (manual labels required):

```powershell
python scripts/score_live_results.py --per-frame demo/outputs/<run_stamp>/per_frame.csv --out demo/outputs/<run_stamp>/score_results.json --pred-source both
```

Interpretation note:

- If your live session only contains 1–2 emotions, “macro-F1 across all 7 classes” will look artificially low (because classes with zero support contribute F1=0).
- Use `metrics.*.macro_f1_present` for a fairer live comparison (macro-F1 over classes that actually appear in manual labels).

Decision rule (live metric gap triage):

- If `metrics.raw` > `metrics.smoothed`: tune stabilization (EMA/vote/hysteresis) to reduce over-smoothing.
- If both are low but offline evaluation is strong: suspect **pipeline parity issues** (face crop/alignment/normalization mismatch).
- If both are low and ExpW domain-shift is low: prioritize domain-shift interventions (Phase 3).

1. Summarize FPS / latency from the produced logs (derived from `per_frame.csv` time span).

Important note:

- The demo supports `--device {auto,cpu,cuda,dml}`. Use `--device cpu` for a true CPU benchmark.

Deliverables:

- One short summary table (CPU benchmark) stored in the process log.
- The demo output folder preserved for evidence.

Optional (recommended) deliverable:

- A small labeled “webcam-mini” eval set (for deployment KPI): export ~200–500 labeled frames from one or more demo sessions and track `macro_f1_present` over time.

### 2.2 Domain-shift evaluation (offline)

Goal: establish a baseline table for domain shift using existing checkpoints.

Recommended target dataset (already in repo):

- `Training_data_cleaned/expw_full_manifest.csv` (use `--eval-split test`)

Actions:

1. Run the one-click evaluation script (batch evaluate multiple run dirs):

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_domain_shift_eval_oneclick.ps1
```

1. Confirm the compare table exists under `outputs/evals/`.

Notes on the current evaluation implementation:

- `scripts/eval_student_checkpoint.py` uses `build_splits(...)` to create a deterministic split from the provided manifest.
- If the manifest is already a fixed test-only list, we should consider enhancing the evaluator with a `--no-split` option so it evaluates all rows directly (optional improvement in Phase 1.3).

Deliverables:

- `outputs/evals/_compare_*.md` for ExpW domain shift.
- The evaluation run folders in `outputs/evals/students/` (each contains `reliabilitymetrics.json`, `calibration.json`, and `eval_meta.json`).

Evidence (current baseline):

- `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md`


---

## Phase FC — Fair paper comparison (do this before new “paper vs us” claims)

## FC1) Lock down official-split anchors (data acquisition / provenance)

Goal: use the same split definitions as papers whenever possible.

Actions:

1. Identify the paper target dataset(s) you will compare (FER2013 already covered).
2. For each dataset:

    - Record **how it was obtained** (Kaggle/official request/drive) and keep a provenance snapshot JSON.
    - If license-restricted (like `fer2013.csv`), do not redistribute; store derived manifests + metrics artifacts only.

Acceptance:

- There is a manifest that corresponds to the official split (or the paper’s split definition) and an evaluation folder containing `reliabilitymetrics.json`.

## FC2) Implement FER2013 ten-crop evaluation (highest ROI)

Reason: many FER2013 papers report **ten-crop** accuracy; our current official-split evaluation is **single-crop**.

Goal: add a second metric block (ten-crop) while keeping single-crop unchanged.

Implementation sketch (minimal scope):

- Add an evaluation option (e.g., `--tta ten-crop`) to the student evaluator.
- Use deterministic cropping order and average logits/probabilities across crops.
- Ensure the output folder name encodes the protocol (e.g., `__singlecrop__` vs `__tencrop__`).

Run plan:

1. Evaluate the **same** checkpoint(s) on official PublicTest/PrivateTest with ten-crop.
2. Export a new summary table that contains **both** protocols.
3. Update the “Comparable” status in the paper table accordingly.

Acceptance:

- Ten-crop metrics exist on disk as `outputs/evals/students/.../reliabilitymetrics.json` and a summary MD/CSV/JSON is generated.
- Report contains single-crop and ten-crop as separate labeled blocks.

## FC3) Controlled gap analysis checklist (per paper target)

Goal: turn “we have a gap” into an explainable list of causes.

For each paper number you reference, fill this checklist:

- Split definition matched? (Y/N)
- Crop/TTA matched? (single/ten-crop/other)
- Preprocessing matched? (alignment/crop, CLAHE, normalization)
- Input resolution matched? (e.g., 224 vs 112/128/256)
- Label mapping matched? (7/8/compound)
- Backbone capacity matched? (MobileNet vs ResNet50/ViT/Swin)
- Training settings comparable? (batch size, optimizer, LR schedule, augmentation, class balancing, KD/DKD temperatures/weights)

Output:

- One short “gap reason” paragraph per paper target, with the checklist results attached.

### 2.3 Pipeline parity checks (live vs offline) (recommended)

Rationale: even with a good student checkpoint, live performance can drop if the demo pipeline differs from offline evaluation (face detection/crop, resize, normalization, color space, CLAHE, etc.).

Checks:

- Confirm face crop size and preprocessing used in the demo matches training/eval assumptions (e.g., 224×224, RGB/BGR handling, normalization, CLAHE on/off).
- Confirm the face detector/cropping policy is stable (tight vs loose boxes can change expression cues).
- If needed, run the offline evaluator on a small set of demo crops to isolate whether the model or the live pipeline is the bottleneck.

### 2.4 Small code improvements to enable clean CPU measurement (optional but recommended)

Rationale: without a forced CPU mode, performance benchmarks can be accidentally run on CUDA/DirectML.

Proposed changes:

- (DONE) Added `--device {auto,cpu,cuda,dml}` to:
  - `demo/realtime_demo.py`
  - `scripts/realtime_infer_arcface.py` (forwarding wrapper)
- (TODO) Add `--device {auto,cpu,cuda,dml}` to:
  - `scripts/eval_student_checkpoint.py`
- Wire the argument to the existing `src/fer/utils/device.py:get_best_device(prefer=...)`.

Deliverable:

- A reproducible CPU benchmark command that always uses CPU.

### 2.4 Backup pack (reproducible ZIP) (completed, keep verifying)

Goal: produce a “restore-anytime” ZIP that contains runnable demo + essential tools + best model artifacts.

Status:

- Created by: `scripts/make_realtime_fer_backup_zip.ps1`
- Output: `outputs/realtime_fer_backup.zip`
- Includes ONNX export via: `scripts/export_student_onnx.py`

Verification (recommended):

1. Unzip into a fresh folder (not inside the repo).
1. Create venv and install deps with `requirements.txt`.
1. Run CPU demo:

```powershell
python demo/realtime_demo.py --model-kind student --model-ckpt outputs/students/<best_run>/best.pt --device cpu
```

## 3) Phase 2 — Pick the best next intervention (based on Phase 1 tables)

Decision rule:

- If CPU demo performance is not acceptable:

  - Focus on deployment optimization first (ONNX export / faster runtime / quantization), because domain-shift improvements won’t matter if the model can’t run.

- If CPU performance is acceptable but domain-shift performance is weak:
- If CPU performance is acceptable but live correctness is confusing:

  - First resolve evaluation correctness: ensure manual labels exist and use `macro_f1_present`.
  - If `metrics.raw` >> `metrics.smoothed`: tune stabilization before retraining.
  - If live is weak but offline is strong: fix pipeline parity before retraining.

- If CPU performance is acceptable and domain-shift performance is weak:

  - Focus on robustness training changes that improve minority-F1 under shift (Phase 3).

## 4) Phase 3 — Improve domain shift with minimal-risk steps (1–3 weeks)

Start from the smallest changes with the best expected ROI:

### 4.1 Stronger robustness augmentations (first choice)

Hypothesis: much of ExpW/webcam domain shift is photometric (lighting, contrast, blur, compression).

Actions:

- Add/strengthen training-time augmentations:

  - brightness/contrast/gamma
  - motion blur / gaussian blur
  - random grayscale
  - mild noise / jpeg compression

Evaluation:

- Re-run the domain-shift evaluator and compare minority-F1 and macro-F1.

### 4.2 Explicit long-tail improvement (minority-F1)

Actions (choose one first):

- Class-balanced reweighting OR focal loss OR logit adjustment.

Evaluation:

- Track minority-F1 on the target evaluation manifest.

### 4.3 Target-aware fine-tuning (if 4.1/4.2 are insufficient)

Actions:

- Fine-tune student using a target-heavy manifest (e.g., ExpW train portion) with KD/DKD using your teacher ensemble softlabels when available.

Evaluation:

- Domain-shift table should show clear improvement on ExpW test.

## 5) Phase 4 — Deployment packaging (after CPU + shift are acceptable)

Actions:

- Use the USB pack workflow to package the demo with the best student checkpoint:

  - `scripts/make_realtime_demo_zip.ps1`
- Verify the packed demo runs on a CPU-only target environment.

Deliverables:

- `outputs/realtime_demo_usb.zip`
- A demo run log folder proving the packaged version runs.

## 6) What we will discuss next (decision points)

Decisions (confirmed):

CPU KPI:

- Performance: FPS >= 20 sustained on CPU
- Latency: keep median per-frame latency low (target <= 50 ms)
- Correctness: predictions should be “correct” when manually labeled; use `scripts/score_live_results.py` to verify

Domain-shift target priority:

- Primary next month: ExpW
- Secondary: webcam logs collected from the real demo environment

Training run budget:

- No fixed limit for now (early stage)

Suggestion (to keep progress efficient even without a limit):

- Keep a light guardrail: only change 1–2 variables per run, and always write a short compare markdown for each new run.

## Plan of next step | 29 Jan 2026

This plan is designed to be executed **one step at a time**, from “easy wins” to “research-y”, while preserving the project rule: **every claim must be backed by artifacts** (JSON/CSV/compare tables).

## Step order (easy → research-y)

### Step 1 — Domain-shift summary table (easy, fast, no training)

Goal:

- Consolidate existing webcam runs into a single comparable table.

Inputs (already in repo):

- `demo/outputs/*/score_results.json`

Action:

- Gather the existing run folders (at minimum the two runs already cited in the final report).
- Produce a single markdown table in a stable location (so it can be referenced by report/process logs).

Deliverables (artifacts):

- (DONE) `outputs/domain_shift/webcam_summary.md` (table of: raw vs smoothed, jitter flips/min, macro_f1_present, minority_f1_lowest3, scored frames; evidence links to each run’s `score_results.json`)

Notes:

- This step is intentionally **non-research**, but it creates a “ground truth dashboard” for the next steps.

### Step 2 — Define “Domain Shift Score” v0 (easy, fast, no training)

Goal:

- Define a single composite score for deployment behavior that is **reproducible** (fixed formula, fixed weights).

Proposed ingredients (all already available or can be derived from `score_results.json`):

- Smoothed accuracy (higher is better)
- Smoothed `macro_f1_present` (higher is better)
- Jitter flips/min (lower is better)
- Optional: mean confidence drift (if present in scoring output)

Definition rule:

- Normalize each metric into a bounded [0, 1] component (with explicit caps), then compute a weighted sum.
- Keep the formula constant across runs to avoid cherry-picking.

Deliverables (artifacts):

- `outputs/domain_shift/domain_shift_score_v0.md` (the exact formula + weights)
- Update `outputs/domain_shift/webcam_summary.md` to include the composite score column

### Step 3 — Temporal stabilization ablation (medium, still engineering)

Goal:

- Show a **flip-rate vs quality** trade-off across stabilizers.

Inputs:

- A single labeled `demo/outputs/<run>/per_frame.csv` and its labels (the same run re-scored multiple ways).

Plan:

- Re-score the same run under multiple settings:
  - raw (no smoothing)
  - EMA (vary `ema_alpha`)
  - vote window (vary `vote_window`)
  - hysteresis (vary `hysteresis_delta`)

Deliverables (artifacts):

- `outputs/temporal/temporal_ablation_<run>.md` (table + a small interpretation paragraph)
- Optional: `outputs/temporal/temporal_tradeoff_<run>.png` (flip-rate vs smoothed macro_f1_present)

Decision:

- Pick a **default stabilizer setting** for the demo (for the rest of the project) based on the trade-off.

### Step 4 — 1–2 domain shift case studies (medium, data capture)

Goal:

- Add 1–2 controlled deployment scenarios to strengthen the “domain shift” story.

Suggested cases (choose any 2):

- Low light / warm indoor light
- Side face / partial profile
- Occlusion (mask / hand / hair)

Protocol:

- Keep the pipeline settings fixed (same checkpoint, same stabilizer settings from Step 3).
- Keep labeling simple: short segments with clear intent (even 2–3 minutes is enough).

Deliverables (artifacts):

- New folders under `demo/outputs/<run_stamp>/` with:
  - `per_frame.csv`
  - `score_results.json`
- Update `outputs/domain_shift/webcam_summary.md` to include these new runs

### Step 5 — Calibration visuals + calibration drift (medium, research vibe)

Goal:

- Add 1–2 reliability diagrams and a short “calibration drift under domain shift” analysis.

Targets:

- Offline: student `reliabilitymetrics.json` (already produced)
- Webcam: labeled runs (from Step 4)

Deliverables (artifacts):

- `outputs/calibration/reliability_<source>.png` (offline)
- `outputs/calibration/reliability_webcam_<run>.png` (webcam)
- `outputs/calibration/calibration_drift.md` (a tiny table: offline vs webcam calibration metrics)

### Step 6 — Safe self-learning + NegL prototype (research-y, 1 iteration is enough)

Goal:

- Demonstrate a safe update loop with explicit rollback and safety gating.

Prototype rule:

- High confidence → pseudo-label update
- Medium confidence → NegL
- Low confidence → skip
- Every update must pass the offline eval-only gate; otherwise rollback.


Deliverables (artifacts):

- A run log entry documenting:
  - what data buffer was used
  - what update was applied
  - whether the eval-only gate passed
  - rollback evidence if it failed

## Next action (do this first)

Start with **Step 1** so we have a single summary table that becomes the “dashboard” for all later steps.

## Field transfer research framework (reference)

The experiment breakdown + evaluation rules + results table templates are maintained here:

- `research/domain_shift/field_transfer_framework.md`
- `research/domain_shift/evaluation_plan.md`
- `research/domain_shift/results_table_templates.md`
