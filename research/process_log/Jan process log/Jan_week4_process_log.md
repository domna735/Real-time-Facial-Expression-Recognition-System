# Process Log - Week 4 of January 2026

This document captures the daily activities, decisions, and reflections during the fourth week of January 2026, focusing on reconstructing the facial expression recognition system as per the established plan.

---

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## YYYY-MM-DD | Short Title

Intent:
Action:
Result:
Decision / Interpretation:
Next:

---

## 2026-01-18 | Next-step planning: CPU readiness + domain shift

Intent:

- Confirm the target CPU specification for deployment planning.
- Create a reproducible summary of `Training_data/` class counts and imbalance.
- Define the immediate next steps to validate CPU usability and domain-shift performance before new training.

Action:

- Confirmed the target dev machine CPU: 13th Gen Intel(R) Core(TM) i9-13900HX (2.20 GHz).
- Generated dataset usage summaries for `Training_data/` (class counts + imbalance) using `scripts/summarize_training_data_counts.py`.

Result:

- Created `table.md` (Markdown summary table).
- Created `outputs/training_data_counts.json` (JSON dump of counts).

Decision / Interpretation:

- Before starting new training, two measurements are required to reduce uncertainty:
  1) CPU real-time feasibility (FPS / latency) using the student model in the demo pipeline.
  2) Domain-shift evaluation using existing evaluation scripts on ExpW (and/or other target manifests).

Next:

- Run a timed student demo session and summarize FPS / latency / flip-rate from `demo/outputs/` logs.
- Run domain-shift evaluation on ExpW and generate a compare table under `outputs/evals/`.

---

## 2026-01-19 | Backup packaging + CPU-forced demo option

Intent:

- Create a reproducible backup ZIP for the real-time FER system (code + tools + model artifacts), including an ONNX export.
- Add a clean way to force CPU (and other devices) for true benchmarking and repeatable runs.

Action:

- Implemented `--device {auto,cpu,cuda,dml}` in the real-time demo and wired it into both student and teacher checkpoint loading.
- Updated the ArcFace wrapper to forward the same `--device` option to the demo.
- Created a backup pack script that stages the minimal runnable repo subset and exports the best student checkpoint to ONNX.
- Ran the backup pack script end-to-end to generate the ZIP and validate the ONNX export.

Result:

- Demo device forcing:
  - `demo/realtime_demo.py` now supports `--device`.
  - `scripts/realtime_infer_arcface.py` now supports `--device` and forwards it.
- Backup pack:
  - Created `outputs/realtime_fer_backup.zip`.
  - Included best student checkpoint and exported `models/student_best.onnx` inside the staged package.
  - ONNX export self-check succeeded (`onnx_check_ok: true`).

Decision / Interpretation:

- CPU benchmarking can now be run reliably (no accidental CUDA/DirectML use).
- The project is now recoverable from a single ZIP that contains the runnable demo + essential artifacts.

Next:

- Run a short CPU-only demo session and record the output folder under `demo/outputs/`.
- Use `scripts/score_live_results.py` on the generated `per_frame.csv` to summarize stability/jitter and protocol-lite accuracy.
- (Optional) Add `--device` to `scripts/eval_student_checkpoint.py` to keep offline evaluation consistent with the demo device selection.

---

## 2026-01-19 | Publish minimal GitHub demo repo (safe + LFS)

Intent:

- Publish a public GitHub demo repository without accidentally uploading the full research repo or any datasets.
- Include the runnable demo plus required runtime artifacts (`student_best.pt`, ONNX) using Git LFS.

Action:

- Identified the backup-stage folder as the best publish base:
  - `outputs/_realtime_fer_backup_stage/Real-time-Facial-Expression-Recognition-System_v2_restart/`
- Created a clean publish folder `github_demo_repo/` and copied the backup-stage contents into it.
- Copied root `.gitignore` and `.gitattributes` into `github_demo_repo/`.
- Promoted `BACKUP_README.md` to `README.md`.
- Copied the best student artifacts into `github_demo_repo/models/`:
  - `student_best.pt`
  - `student_best.onnx` (+ `student_best_onnx_export_meta.json`)
  - `student_best_calibration.json` + `student_best_reliabilitymetrics.json`
- Initialized a new git repo inside `github_demo_repo/`, enabled Git LFS, committed, and pushed to GitHub.

Result:

- A public, minimal, runnable demo repo exists (separate from the main research workspace).
- Large artifacts are tracked via Git LFS:
  - `models/student_best.pt`
  - `models/student_best.onnx`
  - `demo/models/face_detection_yunet_2023mar.onnx`
- Datasets remain excluded from the published repo.

Decision / Interpretation:

- Keeping a dedicated publish folder avoids accidental leakage of private/large research assets.

Next:

- Follow the Phase 1 plan: run CPU demo benchmark + run ExpW domain-shift evaluation table.

---

## 2026-01-19 | Baseline demo log summary (existing run)

Intent:

- Get a quick baseline of demo throughput/latency from existing `demo/outputs/` logs.

Action:

- Parsed the latest available demo log:
  - `demo/outputs/20260109_095433/per_frame.csv`
- Computed throughput and per-frame wall-time from `time_sec` deltas.

Result:

- Baseline (from logs):
  - Estimated FPS: 21.86
  - Median per-frame time: 41.1 ms
  - P95 per-frame time: 83.85 ms

Decision / Interpretation:

- This meets the target FPS/median thresholds, but the device used for that run is not guaranteed to be CPU.

Next:

- Re-run the demo with `--device cpu` (webcam or video file source) and re-compute the same summary for a true CPU benchmark.

---

## 2026-01-19 | ExpW domain-shift evaluation table (baseline)

Intent:

- Establish a reproducible baseline domain-shift table on ExpW before making any new training changes.

Action:

- Ran the one-click domain-shift evaluator:
  - `scripts/run_domain_shift_eval_oneclick.ps1`
- Evaluated 5 existing DKD student run dirs on ExpW (test split) using:
  - `Training_data_cleaned/expw_full_manifest.csv`

Result:

- Generated compare table:
  - `outputs/evals/_compare_20260119_170620_domainshift_expw_full_manifest_test.md`
- Observed baseline performance (ExpW shift) is substantially lower than in-domain metrics:
  - Raw accuracy roughly ~0.58–0.62 across runs.
  - Minority-F1 (lowest-3) roughly ~0.25 across runs.

Decision / Interpretation:

- Domain shift to ExpW is confirmed as a key weakness (supports the plan priority: ExpW first).

Next:

- Use this compare table as the baseline reference for Phase 3 interventions (augmentations / long-tail tweaks / target-aware fine-tuning).

---

## 2026-01-21 | Live vs offline metric gap: define a fair live baseline + next actions

Intent:

- Investigate why real-time “macro-F1” appears very different from offline evaluation.
- Make the live evaluation protocol comparable and actionable (raw vs smoothed, fair macro definition).

Action:

- Identified that the demo output label is stabilized (EMA/vote/hysteresis), while offline evaluation is on raw logits.
- Upgraded the live scoring workflow to compute metrics from both:
  - `metrics.raw` = argmax of per-frame probabilities
  - `metrics.smoothed` = stabilized `pred_label`
- Added a live-session-safe macro metric:
  - `macro_f1_present` = macro-F1 computed only over classes that appear in manual labels
- Re-scored an existing manually-labeled demo run to validate the scorer and quantify smoothing impact.

Result:

- Many demo runs contain **no manual labels**, so F1 cannot be computed for those runs (must collect labeled live sessions).
- On a labeled run, `metrics.raw` outperformed `metrics.smoothed` (smoothing reduced correctness), and `macro_f1_present` was far more meaningful than “macro across all 7” for short live sessions.

Decision / Interpretation:

- The live-vs-offline gap is partly a **metric artifact** (missing labels + limited class coverage) and partly a **pipeline effect** (stabilization can reduce raw correctness).
- Next improvements should be chosen using a strict decision rule:
  1) If `raw` >> `smoothed`: tune stabilization before any retraining.
  2) If both are low but offline is good: suspect pipeline parity (face crop/normalization) and fix that first.
  3) If both are low and ExpW is low: proceed with domain-shift interventions (augmentations → long-tail → target-aware finetune).

Next:

- Run a 2–3 minute CPU-forced demo session with deliberate manual labels (try to cover 4–5 emotions).
- Score with `--pred-source both` and report `macro_f1_present` + raw-vs-smoothed deltas.
- If smoothing hurts: tune EMA/vote/hysteresis settings.
- Perform pipeline parity checks (crop size, normalization, CLAHE, BGR/RGB, detector crop policy).
