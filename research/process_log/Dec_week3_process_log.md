# Process Log - Week 3 of December
This document captures the daily activities, decisions, and reflections during the third week of December 2025, focusing on reconstructing the facial expression recognition system as per the established plan.

## Decisions locked (as of 2025-12-17)
- Use curated training manifest: `Training_data_cleaned/classification_manifest_hq_train.csv`.
- Use eval-only manifest (de-dup): `Training_data_cleaned/classification_manifest_eval_only.csv`.
- Reduce validation frequency during long runs: `--eval-every 10`.
- Keep resume safe: always write `checkpoint_last.pt` every epoch; stop/restart reuses it.
- Handle FERPlus 48×48 at 384×384 via two-stage training: include `ferplus` at 224×224, exclude it for the 384×384 finetune.

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## YYYY-MM-DD | Short Title
Intent:
Action:
Result:
Decision / Interpretation:
Next:

## 2025-12-16 | Single venv + GPU training backend
Intent:
Stabilize one `.venv` and ensure GPU is usable for training on the RTX 5070 Ti.

Action:
- Repaired the consolidated `.venv` after a rename caused broken `pip` launchers by using `python -m pip` and force-reinstalling `pip`.
- Initially confirmed the `sm_120` incompatibility with older CUDA PyTorch builds, and used `torch-directml` as a temporary GPU workaround.
- Backed up the working DirectML environment snapshot to `requirements-directml.txt` for rollback.
- Switched the same `.venv` to CUDA by installing PyTorch **nightly `cu128`** wheels (CUDA 12.8) that include `sm_120` support.
- Regenerated `requirements.txt` after the CUDA nightly install.
- Organized GPU diagnostic scripts under `tools/diagnostics/` while keeping root wrappers (`check_gpu.py`, `check_gpu_dml.py`) for convenience.
- Added minimal repo structure + utilities: `src/fer/utils/device.py` and `scripts/check_device.py`.

Result:
- CUDA is working on the RTX 5070 Ti: `torch==2.11.0.dev20251215+cu128`, `torch.version.cuda==12.8`, `torch.cuda.is_available()==True`, and `torch.cuda.get_arch_list()` includes `sm_120`.
- `check_gpu.py` passes a CUDA tensor smoke test and reports the GPU is functional.
- `requirements.txt` now reflects the CUDA nightly stack; `requirements-directml.txt` keeps the previous DirectML snapshot.

Decision / Interpretation:
Use CUDA nightly `cu128` PyTorch as the primary training backend (it ships `sm_120` kernels). Keep the DirectML snapshot as a fallback option.

Next:
Start implementing the reconstruction plan training pipeline (teacher models → ensembles → student → demo → NL/NegL), and update the device helper to prefer `cuda` during training.

---

## 2025-12-16 | Dataset cleaning (7-emotion standardization)
Intent:
Prepare training data for reconstruction by standardizing labels to 7 emotions only: `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`.

Action:
- Created a non-destructive dataset cleaner `tools/data/clean_7emotions.py`.
- Generated a cleaned view under `Training_data_cleaned/` (does **not** modify originals under `Training_data/`).
- Standardized common folder-name variants (e.g., `Anger→Angry`, `Happiness→Happy`, `Sadness→Sad`, `suprise→Surprise`) and excluded other classes.
- For the YOLO-format dataset, rewrote `data.yaml` to `nc=7` + canonical `names`, and filtered/remapped label indices to the 7 emotions.

Result:
- Output created: `Training_data_cleaned/affectnet_full_balanced`, `Training_data_cleaned/fer2013_uniform_7`, `Training_data_cleaned/ferplus`, `Training_data_cleaned/affectnet_yolo_format`.
- Report generated: `Training_data_cleaned/clean_report.json` (counts + excluded-class summary + YOLO remap details).

Decision / Interpretation:
Keep raw datasets unchanged to prevent accidental data loss; treat `Training_data_cleaned/` as the canonical input for the reconstruction pipeline.

Next:
- Build a unified training manifest/splits from `Training_data_cleaned/` for Teacher #1 (RN18).
- Handle RAF datasets (RAFDB/RAF-ML/RAF-AU/compound) via their annotation files in a separate import step (not folder renames).

---

## 2025-12-16 | RAF import + unified training CSV
Intent:
Extend the 7-emotion cleaning pipeline to include RAF datasets and produce a single CSV manifest for training.

Action:
- Extended `tools/data/clean_7emotions.py` to parse RAF annotation files and write folder-structured outputs under `Training_data_cleaned/`:
	- `rafdb_basic/`: used `basic/EmoLabel/list_patition_label.txt` (IDs 1–7 mapped to the 7 canonical emotions).
	- `rafdb_compound_mapped/`: used `compound/EmoLabel/list_patition_label.txt` (compound labels 1–11 mapped to the *primary/base* emotion: Happy/Sad/Fear/Angry/Disgust).
	- `rafml_argmax/`: used `RAF-ML/EmoLabel/distribution.txt` + `partition_label.txt` and assigned a single label via argmax over the 6 emotion probabilities.
- Added CSV manifest generation: `Training_data_cleaned/classification_manifest.csv`.

Result:
- RAF outputs created:
	- `Training_data_cleaned/rafdb_basic/{train,test}/{Angry,Disgust,Fear,Happy,Sad,Surprise,Neutral}/...`
	- `Training_data_cleaned/rafdb_compound_mapped/{train,test}/{Happy,Sad,Fear,Angry,Disgust}/...` (no Neutral/Surprise by construction)
	- `Training_data_cleaned/rafml_argmax/{train,test}/{Angry,Disgust,Fear,Happy,Sad,Surprise}/...` (no Neutral)
- Unified manifest generated: `Training_data_cleaned/classification_manifest.csv` (374,491 rows total).
- JSON report updated: `Training_data_cleaned/clean_report.json` includes RAF counts and manifest summary.

Decision / Interpretation:
- RAF-AU is excluded from 7-emotion classification because it provides AUs rather than emotion labels.
- For RAF-DB-compound, mapping to the *primary/base* emotion is a pragmatic way to reuse the data for single-label classification; if this harms accuracy, revisit by excluding compound data or adopting a different mapping strategy.

Next:
- Use `Training_data_cleaned/classification_manifest.csv` as the training index for Teacher #1 (RN18) and wire it into the training dataloader.
- Confirm train/val split policy (currently uses each dataset’s provided splits where available).

---

## 2025-12-16 | Manifest validation + RN18 smoke test
Intent:
Confirm the unified CSV manifest is usable for training and validate that labels/paths are correct.

Action:
- Added manifest-backed dataset utilities: `src/fer/data/manifest_dataset.py` (reads `classification_manifest.csv`, maps the 7 labels to indices, and applies a standard RN18 transform).
- Added validator: `tools/data/validate_manifest.py` (checks canonical labels, missing paths, and decodes a sample of images; writes `outputs/manifest_validation.json`).
- Added training smoke test: `scripts/smoke_train_rn18.py` (runs a short RN18 training loop on CUDA; writes `outputs/smoke_rn18/smoke_results.json`).

Result:
- Manifest validation passed:
	- `rows_total=374,491`, `rows_valid=374,491`, `bad_labels=0`, `missing_paths=0`, decode sample `200/200` succeeded.
	- Validation output: `outputs/manifest_validation.json`.
- RN18 smoke training succeeded on CUDA (`NVIDIA GeForce RTX 5070 Ti Laptop GPU`):
	- Train/val sizes used by the loader: `train=301,694`, `val=32,519`.
	- Smoke run (30 steps, batch 64): train loss ≈ `1.95`, partial val acc ≈ `0.225`.
	- Output: `outputs/smoke_rn18/smoke_results.json`.

Decision / Interpretation:
- The CSV manifest is valid and usable as the training index.
- Train/val split policy for training loaders:
	- Use each dataset’s explicit `val` split when present (e.g., `fer2013_uniform_7`, `ferplus`).
	- For sources without an explicit `val` split, carve a deterministic per-source stratified subset from `train` (default `5%`, seeded) to form validation.

Next:
- Promote the smoke script into a real Teacher #1 training entrypoint (epochs, checkpoints, metrics, calibration outputs) and lock down the final split policy for reproducibility.

---

## 2025-12-16 | ExpW high-quality (HQ) dataset + dedicated manifest
Intent:
Add ExpW as a separate “high-quality training/testing” track (not mixed into `Training_data_cleaned/classification_manifest.csv`).

Action:
- Extracted the ExpW image archive (multi-part 7z). The first part was named `origin.7z.001.001`, so it was renamed to `origin.7z.001` to make standard split extraction work.
- Extracted all images under `Training_data/Expression in-the-Wild (ExpW) Dataset/origin/` and handled the nested layout (`origin/origin/*.jpg`).
- Implemented `tools/data/import_expw_hq.py` to:
	- Parse `label.lst` (bbox + bbox confidence + expression id 0–6).
	- Filter by bbox confidence (default `>=60`).
	- Create deterministic stratified splits (train/val/test = `0.8/0.1/0.1`, seed `1337`).
	- Write cropped face images into `Training_data_cleaned/expw_hq/<split>/<label>/...` and generate `Training_data_cleaned/expw_hq_manifest.csv`.
	- Emit an import report JSON under `outputs/expw_hq_import_report.json`.
- Validated trainability by running `scripts/smoke_train_rn18.py` against `Training_data_cleaned/expw_hq_manifest.csv`.

Result:
- ExpW extraction succeeded: `106,962` images extracted.
- HQ subset created from annotations: `kept=33,375 / 91,793` label rows (confidence `>=60`), with `missing_images=0` and `bad_label=0`.
- Split sizes (from loader): `train=26,701`, `val=3,338`, `test=3,336`.
- Outputs created:
	- Cropped HQ dataset view: `Training_data_cleaned/expw_hq/`
	- Dedicated HQ manifest: `Training_data_cleaned/expw_hq_manifest.csv`
	- Import summary: `outputs/expw_hq_import_report.json`
	- Smoke training result: `outputs/smoke_rn18_expw_hq_manifest/smoke_results.json`

Decision / Interpretation:
- Keep ExpW separate from the unified `classification_manifest.csv` to preserve baseline comparisons and to treat ExpW as an HQ track (bbox/face quality filtering + distinct evaluation split).
- Defaulting to cropping faces from ExpW bboxes is aligned with the “high-quality” intent; if we need a non-cropped variant later, the importer can be re-run with cropping disabled.

Next:
- Decide how ExpW HQ is used in the reconstruction plan (e.g., HQ-only fine-tuning and/or held-out HQ evaluation) without contaminating baseline datasets/splits.

---

## 2025-12-16 | ExpW full manifest + unified manifest rebuild (all data)
Intent:
Create a CSV for *all* ExpW labeled faces (91,793) and include ExpW into the unified `classification_manifest.csv`, while keeping the HQ track separate.

Action:
- Generated a full ExpW manifest (no confidence filter) from `Training_data/Expression in-the-Wild (ExpW) Dataset/label.lst`:
	- Wrote `Training_data_cleaned/expw_full_manifest.csv` with `rows_total=91,793` and deterministic stratified splits (train/val/test = `0.8/0.1/0.1`, seed `1337`).
	- Stored `image_path` as absolute paths to the extracted ExpW images and kept bbox metadata columns (`bbox_top/left/right/bottom`, `confidence`, `face_id`, `orig_image`) so cropping can be done on-the-fly.
	- Wrote `outputs/expw_full_import_report.json`.
- Rebuilt the unified manifest to include ExpW-full using `tools/data/build_classification_manifest.py`:
	- `base_rows=374,491` (existing cleaned datasets)
	- `expw_rows=91,793`
	- `total_rows=466,284` written to `Training_data_cleaned/classification_manifest.csv`.
- Validated both manifests with `tools/data/validate_manifest.py` (paths exist + decode sample + bbox crop check):
	- Unified: `outputs/manifest_validation_all_with_expw.json`
	- ExpW-full: `outputs/manifest_validation_expw_full.json`

Result:
- ExpW-full manifest created: `Training_data_cleaned/expw_full_manifest.csv` (91,793 rows).
- Unified manifest updated: `Training_data_cleaned/classification_manifest.csv` now includes ExpW and totals 466,284 rows.
- Validation passed for both:
	- `bad_labels=0`, `missing_paths=0`, decode sample `300/300` ok, bbox crop check ok.

Decision / Interpretation:
- Keep **two** ExpW options:
	- `expw_hq_manifest.csv` + `Training_data_cleaned/expw_hq/` for HQ training/testing (cropped + confidence filtered).
	- `expw_full_manifest.csv` for full ExpW coverage (absolute paths + bbox metadata, no crops written).
- Keep **two** training indices overall:
	- `classification_manifest.csv` = all datasets (including ExpW-full).
	- ExpW-only manifests (`expw_full_manifest.csv` or `expw_hq_manifest.csv`) for isolated evaluation/fine-tuning.

Next:
- Start Step 0 of the reconstruction plan using the now-frozen unified index: lock seeds, record validation JSONs, then proceed to training Teacher #1 with the manifest-driven pipeline.

---

## 2025-12-16 | Higher-res input + HQ crop quality upgrade (choose 384 next)
Intent:
Reduce the “vague” appearance in face crops/training inputs and confirm higher-resolution training works before starting teacher training.

Action:
- Extended `scripts/smoke_train_rn18.py` to support `--image-size` (so we can train at 224/384/448 etc.).
- Enhanced `tools/data/import_expw_hq.py` to support higher crop export quality and better face framing:
	- `--jpeg-quality` + `--jpeg-subsampling`
	- `--bbox-pad-ratio` to add context around the face box before cropping.
- Re-exported ExpW HQ with improved quality settings and padding.
- Ran a CUDA smoke train at 448×448 on ExpW HQ to confirm the pipeline works end-to-end at higher input size.

Result:
- ExpW HQ re-export completed: `kept=33,375 / 91,793` and updated `Training_data_cleaned/expw_hq/` + `Training_data_cleaned/expw_hq_manifest.csv`.
- Updated report: `outputs/expw_hq_import_report.json`.
- 448×448 smoke training succeeded on CUDA; results written to `outputs/smoke_rn18_expw_hq_manifest/smoke_results.json`.

Decision / Interpretation:
- Use `384×384` as the first “higher-res” default for rebuilding teacher models (better detail than 224 with lower compute/memory cost than 448).
- Keep 448 available as a follow-up experiment once the teacher pipeline is stable.

Next:
- Implement Teacher training entrypoints (ArcFace head, AdamW + warmup + cosine, balanced sampling + effective-number weighting, augmentations + CLAHE) and do a 1-epoch smoke run at 384.

---

## 2025-12-16 | Teacher training entrypoint (ArcFace protocol) + checkpoints + ONNX
Intent:
Implement the “Rebuild minimum teacher models” training entrypoint matching the interim report protocol, and add reproducible artifacts + robust resume for overnight runs.

Action:
- Added `scripts/train_teacher.py`:
	- Backbones via `timm` (supports: `resnet18`, `resnet50`, `tf_efficientnet_b3`, `convnext_tiny`, `vit_base_patch16_384`).
	- ArcFace head (m=0.35, s=30) with stabilization: plain-logits warmup (5 epochs) and margin ramp (epochs 5→15).
	- Optimizer: AdamW (lr=3e-4, wd=0.05) + cosine schedule + 2-epoch LR warmup.
	- Imbalance handling: balanced mini-batches (>=2 per class) + effective-number weighting.
	- Augmentations: random crop/flip/color jitter + optional CLAHE.
	- Required artifacts written per run: `alignmentreport.json`, `calibration.json`, `reliabilitymetrics.json`, `history.json`.
- Added robustness features for overnight training:
	- Auto-resume from `<output-dir>/checkpoint_last.pt`.
	- Checkpoint cadence: `--checkpoint-every 10` saves `checkpoint_epochXYZ.pt` plus `checkpoint_last.pt` every epoch.
	- Best model tracking: writes `best.pt`.
	- Inference export: writes `best.onnx` (when best updates) and `last.onnx`.
	- Timing: prints and records `epoch_sec` and `total_sec`.
- Updated `scripts/run_teachers_overnight.ps1` to run all 5 teachers with stable `--output-dir` locations so resume works after crashes.

Result:
- 1-epoch smoke at 384 succeeded (RN18), and the run folder contained the required JSON artifacts plus checkpoints.

Decision / Interpretation:
- Only intentional deviation from the interim report protocol: using `384×384` inputs for teacher rebuilding (interim report defaulted to 224). Keep 224 as a fallback if memory/time becomes a blocker.
- ViT note (from interim report): ViTs can be less stable in the <300k data regime; still trained here for completeness, but expect CNN teachers to be the main KD sources.

---

## 2025-12-16 | Teacher pipeline fixes (ViT warnings + ONNX export verification)
Intent:
Make the overnight teacher runner robust to non-fatal stderr warnings, and ensure ONNX export reliably produces `best.onnx/last.onnx` artifacts.

Action:
- Updated `scripts/run_teachers_overnight.ps1` to avoid treating stderr warnings as fatal; rely on process exit code.
- Adjusted ViT backbone creation in `scripts/train_teacher.py` to avoid changing ViT pooling config (pretrained weight mismatch warnings previously surfaced on stderr).
- Fixed ONNX export blockers in `scripts/train_teacher.py`:
	- Prefer legacy TorchScript-based export path (avoid requiring `onnxscript`).
	- Added `onnx==1.20.0` to `requirements.txt` and `requirements-directml.txt`.
	- Added `--export-onnx-only` to export ONNX from an existing checkpoint without retraining.
	- Patched checkpoint loading for PyTorch 2.6+ default `weights_only=True` by forcing `torch.load(..., weights_only=False)` when supported.

Result:
- Targeted smoke runs for ViT + ConvNeXt complete (training no longer aborts due to the previous ViT-related stderr behavior).
- ONNX export verified after installing `onnx` and patching checkpoint load:
	- `outputs/teachers/_smoke_convnext_fix/last.onnx` successfully created via `--export-onnx-only`.

Decision / Interpretation:
Treat ONNX export as a first-class artifact and keep a zero-training `--export-onnx-only` path to recover exports from already-finished checkpoints.

Next:
Re-run the overnight teachers once confirmed (optionally start with ConvNeXt + ViT only) and ensure each output dir ends with `best.onnx` and `last.onnx`.

Next:
- Run `run_teachers_overnight.ps1` in `-Smoke` mode to sanity-check all 5 backbones, then run the full 60-epoch overnight training.

---

## 2025-12-17 | Teacher training runtime + resume robustness
Intent:
Continue teacher training (RN18 + ConvNeXt first), confirm ArcFace is really active, and reduce wall-clock time without breaking reproducibility artifacts or resume.

Action:
- Confirmed resume behavior: training state is restored from `checkpoint_last.pt` in each `--output-dir` (not from JSON metric files).
- Confirmed ArcFace is active after warmup/ramp (e.g., later epochs report `warmup_plain_logits=False` and `arcface_margin=0.35`).
- Reduced overhead for long runs by adding speed controls to `scripts/train_teacher.py`:
	- `--eval-every` to validate less frequently.
	- `--skip-onnx-during-train` to avoid repeated ONNX exports mid-run (export still happens at the end, and `--export-onnx-only` remains available).
	- Enabled CUDA-side speedups where safe (TF32 + cuDNN benchmark) for faster matmul/conv.
- Exposed the new speed flags in `scripts/run_teachers_overnight.ps1` so overnight runs can be tuned without editing Python.
- Ran a smoke execution of the updated PowerShell runner to confirm it resumes properly and does not fail on non-fatal stderr warnings.

Result:
- RN18 and ConvNeXt runs resume correctly from their `checkpoint_last.pt` and continue training as expected.
- Teacher runs remain reproducible and still produce the required artifacts (`alignmentreport.json`, `reliabilitymetrics.json`, `calibration.json`, checkpoints, `best.pt`, ONNX outputs).
- Clear next lever identified: training time is dominated by dataset size + input resolution; reducing eval/export frequency helps but dataset scale is still the main driver.

Decision / Interpretation:
- Keep ArcFace protocol and resume logic as-is (they are working).
- Treat dataset scale as the main knob if 384×384 training is too slow; do not compromise resolution first unless needed.

Next:
- Propose a “quality-first subset” strategy to cut down the 466k-row unified manifest while keeping higher input resolution.

---

## 2025-12-17 | Two-stage 224→384 teacher schedule (FERPlus-safe) + RN18/B3 smoke verification
Intent:
Make the two-stage policy (Stage A @ 224 includes FERPlus, Stage B @ 384 excludes FERPlus) runnable end-to-end, with a quick smoke check for both RN18 and EfficientNet-B3.

Action:
- Verified Stage A→B works for RN18 via direct CLI:
	- Stage A (224, include FERPlus):
		- `--image-size 224 --include-sources ferplus,rafdb_basic,affectnet_full_balanced,expw_hq`
	- Stage B (384, exclude FERPlus):
		- `--image-size 384 --exclude-sources ferplus --init-from <stageA>/checkpoint_last.pt`
		- Important: Stage B uses `--init-from` (new run: epoch resets, fresh optimizer/scaler), NOT `--resume`.
- Added a dedicated smoke runner for RN18 + B3:
	- `scripts/smoke_teachers_rn18_b3_2stage.ps1`
	- Runs Stage A→B for `resnet18` and `tf_efficientnet_b3`.
	- Uses small fixed batch limits (`--max-train-batches`, `--max-val-batches`) so it finishes quickly.
	- Validates required artifacts exist for each stage: `alignmentreport.json`, `history.json`, `reliabilitymetrics.json`, `calibration.json`, `checkpoint_last.pt`, `last.onnx`.
- Fixed a smoke-setting pitfall discovered during B3 Stage B:
	- `BalancedBatchSampler` requires `batch_size >= 7 * min_per_class(2) = 14`.
	- The smoke script now clamps B3 batch sizes to at least 14.
- Reduced unnecessary pretrained downloads for Stage B / resume cases:
	- Updated `scripts/train_teacher.py` so when `--init-from`/`--resume`/auto-resume is used, `timm` pretrained weights are automatically disabled (avoids HF Hub rate-limit warnings).
	- Optional manual override: `--no-pretrained`.

Result:
- RN18 Stage A→B smoke confirmed working and produces full artifacts.
- RN18 + B3 smoke script is available for quick regression checks before starting overnight runs.

How to run (smoke):
- From repo root (with `.venv` activated):
	- `powershell -File .\scripts\smoke_teachers_rn18_b3_2stage.ps1 -Clean`
	- Optional knobs: `-NumWorkers 0`, `-MaxTrainBatches 10`, `-MaxValBatches 5`, `-NoClahe`.

Decision / Interpretation:
- Keep the two-stage policy as the default for teacher rebuilding:
	- Stage A 224 gives FERPlus signal without 384 upsampling artifacts.
	- Stage B 384 restores high-resolution performance using HQ sources and excludes FERPlus.

Notes / Pitfalls:
- If you see “sending unauthenticated requests to the HF Hub”, set `HF_TOKEN` or use `--no-pretrained`.
- ONNX exporter warning (TorchScript legacy exporter) is expected and not fatal.

Next:
- Use `scripts/run_teachers_overnight_rn18_b3_2stage.ps1` for the real 60-epoch overnight runs (RN18 + B3 only), with `--eval-every 10` and resume-safe output dirs.

---

## 2025-12-17 | Overnight RN18+B3 2-stage runner (resume-safe + lock-safe)
Intent:
Run the full 60-epoch two-stage schedule for RN18 and EfficientNet-B3 without risking accidental deletion of an output directory during checkpoint/ONNX writes, and ensure Ctrl+C interruptions can be resumed.

Action:
- Updated `scripts/train_teacher.py` to create a per-run lock file `.<run>.lock` (implemented as `output_dir/.run.lock`) for the duration of training/export and remove it on normal exit.
- Updated `scripts/run_teachers_overnight_rn18_b3_2stage.ps1`:
	- Added lock-aware cleaning: `-Clean`, `-CleanStageA`, `-CleanStageB` refuse to delete if `output_dir/.run.lock` exists.
	- Resume semantics:
		- Re-running the script with the same `Seed` and default output dirs auto-resumes from `checkpoint_last.pt`.
		- Stage B only uses `--init-from` when Stage B has no checkpoint yet; if Stage B already has `checkpoint_last.pt`, it will auto-resume Stage B (no re-init).
	- Added stale-lock recovery flags:
		- `-UnlockStale` removes `output_dir/.run.lock` only if the PID recorded in the lock is not running.
		- `-ForceUnlock` (only with `-UnlockStale`) forces lock removal (use only if you are sure nothing is running).
	- Enforced sampler constraint: `BatchSize >= 14` (7 classes × `min_per_class=2`).

Result:
- Overnight command (single terminal is enough):
	- `powershell -ExecutionPolicy Bypass -File scripts\run_teachers_overnight_rn18_b3_2stage.ps1`
- Ctrl+C / KeyboardInterrupt behavior:
	- To resume: rerun the same command (DO NOT use `-Clean`).
	- `-Clean` intentionally restarts from scratch by deleting the output dirs.
- If a stale lock ever blocks reruns:
	- `powershell -ExecutionPolicy Bypass -File scripts\run_teachers_overnight_rn18_b3_2stage.ps1 -UnlockStale`
	- (Last resort) `... -UnlockStale -ForceUnlock`

Decision / Interpretation:
- Use one terminal to run the overnight script. A second terminal is optional only for monitoring (e.g., `Get-Content -Wait <logfile>` or GPU monitoring); it is not required for correctness.
- “Resume” means rerun with the same output dirs; “Clean” means reset/restart.

Next:
- Start RN18+B3 60-epoch overnight run using the two-stage runner; keep `-Clean` for intentional restarts only.

---

## 2025-12-17 | Verify warmup+cosine + temperature scaling (old vs current)
Intent:
Confirm whether the rebuilt teacher trainer currently implements (1) warmup+cosine schedule and (2) temperature scaling calibration, and record the verified differences between the old protocol and the current two-stage protocol.

Action:
- Checked `scripts/train_teacher.py` implementation:
	- LR schedule function `lr_for_step(...)` implements linear warmup then cosine decay.
	- Temperature scaling function `fit_temperature(...)` fits a single scalar T on validation NLL (LBFGS) and writes `calibration.json`.
	- Validation + temperature scaling frequency is controlled by `--eval-every`.
- Checked the old interim report `research/Real-time-Facial-Expression-Recognition-System Interim Report/version 2 Real-time-Facial-Expression-Recognition-System Interim Report old.md` for old teacher protocol details.
- Updated the comparison report `two stage teacher model training vs old teacher model training.md` with the verified details and an “Suggestions” section.

Result:
- Current pipeline status:
	- Warmup+cosine LR schedule: YES (2-epoch warmup by default).
	- Temperature scaling: YES (global single-T fitted on val NLL; metrics logged and `calibration.json` saved).
- Old protocol details are preserved in the old interim report (important items: ArcFace m=0.35/s=30, plain-logits warmup and margin schedule, cosine+warmup, CLAHE preprocessing, and calibration temperatures like T*≈1.2).

Decision / Interpretation:
- Keep the old interim report markdown and the PDFs as the canonical “training protocol record” even if you delete large raw datasets. If disk space is needed, prefer deleting raw images first, but keep:
	- `Training_data_cleaned/*.csv` manifests
	- `Training_data_cleaned/clean_report.json`
	- the protocol reports (old.md + PDFs)

Next:
- Optional improvements to consider next (small, high impact): add `--min-lr`, add optional per-class temperature scaling, and log manifest hash + environment snapshot per run.

---

## 2025-12-19 | Two-stage run timing anomaly (B3 Stage B very slow)
Intent:
Explain why EfficientNet-B3 Stage B (384) takes extremely long per epoch, confirm whether configuration is correct, and record the concrete run results/timings.

Action:

Result:
	- Stage A (224, include FERPlus): `train_rows=209,661`, `steps/epoch=3,275`.
	- Stage B (384, exclude FERPlus): `train_rows=92,903`, `steps/epoch=1,451`.
- Observed end-to-end timings (from `history.json`):
	- RN18 Stage A (224): `total_sec=15414` (~4.28h), best `macro_f1=0.7312` (@epoch 20).
	- RN18 Stage B (384): `total_sec=13114` (~3.64h), best `macro_f1=0.6730` (@epoch 59).
	- B3 Stage A (224): `total_sec=27567` (~7.66h), best `macro_f1=0.7262` (@epoch 40).
	- B3 Stage B (384): epoch 0 completed with `epoch_sec=13856` (~3.85h), `val_macro_f1=0.6620`.
	- `outputs/teachers/B3_tf_efficientnet_b3_seed1337_stageB_img384/.run.lock` contains `--init-from .../B3_tf_efficientnet_b3_seed1337_stageA_img224/checkpoint_last.pt`.
	- `nvidia-smi` shows GPU utilization ~99% with power draw ~50W (P0 state), indicating this run is compute-bound and likely power-capped on the laptop.

Decision / Interpretation:

Next:
	- Lower Stage B image size for B3 (e.g., 320/352 instead of 384), or reduce Stage B max epochs.
	- Disable `--clahe` for Stage B (CPU-side cost + may not be needed after Stage A).
	- Ensure Windows/NVIDIA power mode is “Best performance” and the laptop is plugged in to avoid low-power caps.

## Note: why Stage B can look worse than Stage A (and what to change)
Observation:

Key reason #1 (most important): Stage A and Stage B validation sets are NOT the same
	- Stage A: `val_rows=21,503` (includes FERPlus).
	- Stage B: `val_rows=6,821` (no FERPlus).
So “Stage B is worse” is not a fair apples-to-apples comparison; it is a different (often harder) distribution.

Key reason #2: Stage B currently initializes from Stage A `checkpoint_last.pt`, not `best.pt`

Key reason #3: Stage B is a fine-tune regime (smaller + different data), LR schedule may be too aggressive

Planned follow-up experiments (small, controlled):

---

Intent:
- Reproduce the old teacher report protocol more closely (RAF-DB test), and understand why RN18 Stage B metrics look much worse than Stage A.

- Switched the two-stage runner to support:
	- Running only RN18 (fast iteration) and selecting the manifest preset.
	- Stage B init-from now prefers Stage A `best.pt` (fallback: `checkpoint_last.pt`).
- Added an evaluation helper to match old-report conventions:
- Ran RN18 with the unified manifest (`Training_data_cleaned/classification_manifest.csv`) so RAF-DB sources are present.

Result:
- RN18 Stage A (224, include-sources) and Stage B (384, exclude FERPlus only) are NOT comparable because Stage B uses a much broader and noisier training/validation mixture.
		- Stage A include-sources: `ferplus, rafdb_basic, affectnet_full_balanced, expw_hq`
		- Rows after filter: `225,629`
		- Train/Val: `182,960 / 18,165`
		- Pretrained: `true`
	- From `outputs/teachers/RN18_resnet18_seed1337_stageB_img384/alignmentreport.json`:
		- Stage B exclude-sources: `ferplus` only (no include filter)
		- Train/Val: `258,370 / 27,017`
		- Pretrained: `false` (expected when loading from checkpoint)
- Observed metrics trend (from `history.json`):
	- Stage A: starts high and quickly reaches strong macro-F1 because it validates on the Stage A distribution (includes FERPlus and excludes several harder/noisier sources).
	- Stage B: starts lower and improves more slowly; early macro-F1 is ~0.63 vs ~0.69 for Stage A because the Stage B validation set is different and includes harder/noisier sources.

Loss-function sanity check:
- The loss is standard multi-class cross-entropy on logits:
	- During warmup epochs it uses `linear_head` logits.
	- After warmup it uses ArcFace logits (`ArcMarginProduct`) with a margin ramp.
- This is consistent with the intended ArcFace protocol; nothing indicates the loss is “wrong” given the current design.
- Note: train loss values are not directly comparable across Stage A/B because:
	- The data mixture differs strongly.
	- The head changes after warmup (plain logits → ArcFace logits).
	- The class-balancing strategy affects loss scale.

Decision / Interpretation:
- Stage B “worse macro-F1 than Stage A” is expected under this setup because Stage B validates on a different distribution.
- To reproduce the old report, prioritize evaluation on the same target (RAF-DB test) rather than comparing Stage A/B val macro-F1.

Operational note (Ctrl+C resume safety):
- Interrupting training with Ctrl+C is safe:
	- The trainer writes `checkpoint_last.pt` at the end of each completed epoch.
	- If you Ctrl+C mid-epoch, you will resume from the last completed epoch (not losing earlier epochs).
- If a stale `.run.lock` remains, use the runner’s `-UnlockStale` flag.

Next:
- Evaluate RN18 checkpoints on RAF-DB test with fixed `T=1.2` to compare apples-to-apples with the old teacher table.
- Only after RN18 matches expectations, repeat for B3 (prefer remote GPU for Stage B).

---

## 2025-12-19 | B3 Stage A “true best” rescan + multi-GPU server plan
Intent:
- Decide whether to re-pick the best Stage A checkpoint (because Stage A used `--eval-every 10`) before restarting B3 Stage B.
- Record the plan for running multiple teacher trainings in parallel on a 4-GPU server.

Action:
- Inspected `outputs/teachers/B3_tf_efficientnet_b3_seed1337_stageA_img224/` and confirmed it has multiple saved checkpoints:
	- `checkpoint_epoch009.pt, 019, 029, 039, 049, 059` plus `checkpoint_last.pt` and `best.pt`.
- Confirmed the current B3 Stage B lock is stale:
	- `outputs/teachers/B3_tf_efficientnet_b3_seed1337_stageB_img384/.run.lock` PID is not running.
- Decided to use the checkpoint-sweep approach to re-evaluate all *saved* checkpoints and select the best by the desired metric.

Result:
- “Screening the folder manually” is not enough: because Stage A evaluated only every 10 epochs, `best.pt` can be wrong (or at least “best among evaluated points”, not necessarily best among all epochs).
- A rescan is possible, but only among the checkpoints that exist on disk. If we did not save every epoch, we cannot recover a truly-best epoch that was never checkpointed.

Decision / Interpretation:
- Recommended workflow before restarting B3 Stage B:
	1) Run a sweep over Stage A checkpoints (same eval protocol) to pick the best checkpoint among saved ones.
	2) Restart Stage B using `--init-from <StageA/best.pt>` (or the sweep-selected checkpoint) so Stage B starts from the strongest Stage A point.
	3) When restarting Stage B, clear stale locks using the runner’s `-UnlockStale` (safe if PID is not running).

Server plan (4 GPUs):
- Yes — if the server has 4 GPUs, you can train different teacher models at the same time.
- Use **one PowerShell terminal per GPU** and pin each run with the runner’s `-CudaDevice` argument (sets `CUDA_VISIBLE_DEVICES`).
	- Example: run B3 on GPU0, ConvNeXt on GPU1, ViT on GPU2, and leave GPU3 for RN18 / evaluation / demos.
	- Use the per-model scripts:
		- `scripts/run_teachers_overnight_b3_2stage.ps1`
		- `scripts/run_teachers_overnight_convnext_2stage.ps1`
		- `scripts/run_teachers_overnight_vit_2stage.ps1`

Next:
- After RN18 Stage B finishes locally, sweep B3 Stage A checkpoints to confirm the best candidate, then restart B3 Stage B.
- When the server is available, launch the 3 teacher runs in parallel (one per GPU) using `-CudaDevice`.

---

## 2025-12-19 | Pipeline scripts + realtime demo audit + NL/NegL scaffolds
Intent:
- Bring the repo into a “reconstruction-plan complete” state for the KD/DKD student pipeline and for the realtime demo artifacts.
- Create the NL/NegL *scaffolding files* requested by the checklist (not full training integration yet).

Action:
- Added/verified end-to-end pipeline scripts:
	- `scripts/export_softlabels.py` (teacher logits export aligned to manifest rows)
	- `scripts/train_student.py` (CE/KD/DKD with shuffle-safe mapping via `softlabels_index.jsonl`)
	- `scripts/diagnose_alignment.py` (softlabel artifact checks)
	- `scripts/compute_reliability.py` (ECE/NLL/Brier and related metrics from logits)
	- `scripts/realtime_infer_arcface.py` (wrapper around the realtime demo)
	- `scripts/score_live_results.py` (Protocol-Lite scoring on `per_frame.csv`)
- Realtime demo audit + correctness fix:
	- Confirmed realtime demo supports manual labeling via keyboard (1..7) and clickable label bar.
	- Confirmed artifacts are written: `per_frame.csv`, `events.csv`, `demoresultssummary.csv`, `thresholds.json`.
	- Fixed a label-order mismatch by importing the canonical label order from the dataset module.
	- Fixed a startup bug introduced by the above change: `demo/realtime_demo.py` now imports `sys` (required for `sys.path.insert`).
- NL/NegL scaffolds added:
	- `src/fer/nl/memory.py` (minimal NL memory module scaffold)
	- `src/fer/negl/losses.py` (minimal NegL loss scaffold)
	- `src/fer/utils/grad_accum.py` (gradient accumulation helper)
	- `scripts/smoke_nl.py` (small smoke test)
	- `research/nl_negl/neglconfig.json` and `research/nl_negl/neglrules.md`

Result (Done / Not done against reconstruction plan):
- Realtime demo (Step 4):
	- Manual labeling + events logging: DONE.
	- Video input support (`--source <video>`): DONE.
	- Real-time parameter adjustment hotkeys: DONE.
	- Detectors (YuNet/DNN/Haar): DONE.
	- `logit_bias.json` support: NOT DONE (optional in plan; currently not read/applied).
	- Applying `calibration.json` temperature scaling during inference: NOT DONE (demo uses raw logits→softmax).
- NL + NegL (Step 5):
	- `neglconfig.json`, `neglrules.md`: DONE (scaffold).
	- NL smoke logs (saved run outputs): NOT DONE.
	- NL/NegL integration into student training: NOT DONE.

Decision / Interpretation:
- Treat the demo as “protocol-lite ready” now (labeling + artifacts + scoring script exist), but defer calibration/logit-bias inference knobs until after student/teacher checkpoints are finalized.
- Treat NL/NegL work as scaffolding only; do not claim training integration is complete until it is wired into the training scripts and produces logs/metrics.

Next:
- If needed, extend `demo/realtime_demo.py` to optionally load and apply:
	- `calibration.json` (global temperature scaling at inference), and
	- `logit_bias.json` (per-class additive logit bias before softmax).
- Decide where “NL smoke logs” should live (e.g., `outputs/nl_smoke/...`) and standardize the artifact naming.
```
---

## 2025-12-20 | Realtime demo validation + YuNet model download fix
Intent:
Validate the real-time webcam demo end-to-end (face detection → FER prediction → manual labeling UI → CSV artifacts) using a teacher checkpoint.

Action:
- Ran realtime demo with teacher checkpoint (RN18 Stage A best): `scripts/realtime_infer_arcface.py --source webcam --detector yunet`.
- Fixed a Python 3.11 dynamic-import crash caused by `@dataclass` evaluation during `importlib` loading of `scripts/train_teacher.py` (registered module in `sys.modules` before `exec_module`).
- Fixed YuNet ONNX download problem: GitHub was returning a **Git LFS pointer file** (tiny text file) instead of the real model; added download validation + retry URLs so the model is a real binary.
- Verified the demo produces output artifacts under `demo/outputs/test_RN18_stageA/`.
- Added an additional summary CSV for easy comparison: `per_class_correctness.csv` (per-expression accuracy using frames that have manual labels).

Result:
- Realtime webcam demo runs with YuNet after the model download fix.
- Outputs are generated:
	- `per_frame.csv` (frame-by-frame logs)
	- `events.csv` (manual label segments)
	- `demoresultssummary.csv` (run metadata)
	- `thresholds.json` (smoothing parameters)
	- `per_class_correctness.csv` (7-expression correctness summary)

Decision / Interpretation:
- When using GitHub-hosted ONNX models, validate downloads (avoid Git LFS pointer/HTML responses) to prevent OpenCV parse errors.
- Keep realtime evaluation simple and comparable by reporting per-class correctness from manually labeled frames.

Next:
- Run the same realtime validation using B3 pretrained teacher (`outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/best.pt`) and compare `per_class_correctness.csv` across models.

---

## 2025-12-20 | Teacher Ensemble (RN18+B3 Stage A) quick check + “bad metrics” diagnosis
Intent:
Create weighted teacher ensembles (RN18/B3) and sanity-check why the first ensemble metrics on `eval_only` looked unexpectedly poor.

Action:
- Exported ensemble softlabels on `Training_data_cleaned/classification_manifest_eval_only.csv` with `--split test --image-size 224`:
	- RN18 0.5 / B3 0.5 output dir: `outputs/softlabels/_smoke_ens_rn18_0p5_b3_0p5_test`
	- Alignment check: `scripts/diagnose_alignment.py ... --require-classorder` (OK)
- Ran two “single-teacher” baselines using the same exporter by setting weights to 1/0:
	- RN18-only: `outputs/softlabels/_smoke_single_rn18_test`
	- B3-only: `outputs/softlabels/_smoke_single_b3_test`
- Investigated the manifest split composition and file existence on disk.

Result:
- The ensemble is actually **better than both single teachers** on this exact evaluation slice:
	- RN18-only (test): accuracy ≈ 0.497, macro-F1 ≈ 0.387
	- B3-only (test): accuracy ≈ 0.527, macro-F1 ≈ 0.397
	- Ensemble 0.5/0.5 (test): accuracy ≈ 0.545, macro-F1 ≈ 0.420
- The “why so bad?” root cause is **evaluation mismatch + missing files**:
	- `classification_manifest_eval_only.csv` has `test_rows_total=11890`, but `missing=5110` images on disk.
	- Missing sources are dominated by: `expw_hq`, `rafml_argmax`, `rafdb_compound_mapped`.
	- After the loader drops missing paths, the effective test set becomes only `expw_full` (6780 rows).
	- This is a harder/noisier domain than the Stage A teacher’s in-domain validation (so metrics are not comparable).

Decision / Interpretation:
- The ensemble code path is behaving correctly (alignment OK; ensemble improves accuracy/macro-F1 vs each teacher).
- “Bad” numbers here are primarily because we are evaluating on a narrowed subset (only `expw_full`) and not the same distribution used for the Stage A best/val metrics.

Next:
- For fair teacher/ensemble comparison, evaluate on a manifest/sources that match what the teacher was trained on (or re-create the missing sources so `eval_only` test is complete).
- Add more visibility in tooling: patched `scripts/export_ensemble_softlabels.py` to warn when many rows in a chosen split are missing on disk (prevents silent subset evaluation).

---

## 2025-12-20 | Ensemble evaluation on an alternative (complete) test set: ExpW-full
Intent:
Get a cleaner “real” ensemble comparison without the missing-file bias present in `classification_manifest_eval_only.csv`.

Action:
- Verified `Training_data_cleaned/expw_full_manifest.csv` has `test` rows with **0 missing files** on disk.
- Ran the 3 ensemble weights (RN18/B3 Stage A) on `expw_full_manifest.csv --split test`:
	- RN18 0.3 / B3 0.7
	- RN18 0.5 / B3 0.5
	- RN18 0.7 / B3 0.3
- Used `--num-workers 0` to avoid Windows multiprocessing import overhead.

Result:
- ExpW-full test rows evaluated: `9179` (missing `0`).
- Metrics (ExpW-full test):
	- RN18 0.3 / B3 0.7: accuracy ≈ 0.566, macro-F1 ≈ 0.400
	- RN18 0.5 / B3 0.5: accuracy ≈ 0.577, macro-F1 ≈ 0.417  (best among these 3)
	- RN18 0.7 / B3 0.3: accuracy ≈ 0.530, macro-F1 ≈ 0.387

Decision / Interpretation:
- On ExpW-full test, the balanced ensemble (0.5/0.5) performs best among the tested weights.
- The earlier “bad result” impression was mainly due to evaluating a reduced subset caused by missing-source files in the eval-only manifest.

Next:
- If we want a multi-source test (RAF + ExpW HQ + etc.), restore/regenerate the missing datasets so `classification_manifest_eval_only.csv` is complete, then re-run the same ensemble sweep.

---

## 2025-12-20 | Ensemble evaluation on the full unified test split (49,457 rows)
Intent:
Evaluate the ensemble teacher using **more data** (multi-source test split) to get a stronger, more representative comparison.

Action:
- Fixed a data-path resolution issue in `scripts/export_ensemble_softlabels.py`:
	- Many manifests store `image_path` relative to `Training_data_cleaned/`, but the exporter previously resolved relative paths from repo root.
	- Updated default `--data-root` to `Training_data_cleaned/` so unified-manifest paths resolve correctly.
- Ran ensembles on `Training_data_cleaned/classification_manifest.csv --split test` (multi-source test, 49,457 rows).
- Also ran single-teacher baselines on the same split for comparison.

Result:
- Full unified test evaluated: `49,457` rows.
- Single teachers (full unified test):
	- RN18-only: accuracy ≈ 0.622, macro-F1 ≈ 0.587
	- B3-only: accuracy ≈ 0.644, macro-F1 ≈ 0.609
- Ensembles (full unified test):
	- RN18 0.3 / B3 0.7: accuracy ≈ 0.645, macro-F1 ≈ 0.611
	- RN18 0.5 / B3 0.5: accuracy ≈ 0.656, macro-F1 ≈ 0.622  (best among tested)
	- RN18 0.7 / B3 0.3: accuracy ≈ 0.625, macro-F1 ≈ 0.591

Decision / Interpretation:
- On the larger multi-source test split, the ensemble is **better than each single teacher**, and the best weight among these three is `0.5/0.5`.
- The old report’s very high numbers (e.g., macro-F1 ≈ 0.79) likely correspond to a different evaluation target/setup (dataset mix + preprocessing + temperature scaling T*), so matching it requires using the same benchmark split and calibration protocol.

Next:
- If the goal is to reproduce the old report table, define the exact benchmark (e.g., RAF-DB test) and apply the same temperature scaling setting (T*=1.2) during evaluation.

---

## 2025-12-20 | RAF-DB benchmark (CLAHE + logit fusion) + uncleaned RAFDB-basic fix
Intent:
- Get an apples-to-apples RAF-DB benchmark under the “legacy-like” protocol (CLAHE + logit-space fusion).
- Start the requested “uncleaned Training_data” evaluations and resolve any path/layout issues.

Action:
- Ran RAF-DB-only test evaluations (3068 rows) for RN18+B3 weights {0.3/0.7, 0.5/0.5, 0.7/0.3} using:
	- `--use-clahe`
	- `--ensemble-space logit`
- Measured teacher softlabel sharpness using `scripts/inspect_softlabels.py` to guide KD temperature.
- Fixed uncleaned RAFDB-basic manifest generation:
	- Discovered images are in `Training_data/RAFDB-basic/basic/Image/aligned/aligned/` (nested folder).
	- Patched `scripts/build_uncleaned_manifests.py` to auto-detect `aligned/aligned`.
	- Regenerated `Training_data/uncleaned_manifests/rafdb_basic_manifest.csv`.
- Re-ran uncleaned RAFDB-basic test ensemble export/eval to confirm the missing-file issue is gone.

Result:
- RAF-DB test (cleaned) metrics (CLAHE + logit fusion):
	- RN18 0.3 / B3 0.7: accuracy ≈ 0.8514, macro-F1 ≈ 0.7708
	- RN18 0.5 / B3 0.5: accuracy ≈ 0.8563, macro-F1 ≈ 0.7775 (best among these three)
	- RN18 0.7 / B3 0.3: accuracy ≈ 0.8347, macro-F1 ≈ 0.7484
- Softlabel sharpness (RAF-DB, 0.5/0.5): targets are extremely sharp at T=1 (mean max-prob ≈ 0.983; p99 max-prob = 1.0); noticeably softer at T≈4–6.
- Uncleaned RAFDB-basic manifest rebuild:
	- Rows: 15,339; missing aligned images: 0
	- Uncleaned RAFDB-basic test ensemble (0.5/0.5, CLAHE + logit): accuracy ≈ 0.8563, macro-F1 ≈ 0.7775
	- Output folder: `outputs/softlabels/_uncleaned_rafdb_basic_test_rn18_0p5_b3_0p5_logit_clahe_20251220_v2`

Decision / Interpretation:
- For RAF-DB, 0.5/0.5 is best overall among the tested weights; moving toward B3-heavy improves Disgust/Fear.
- For KD/DKD, start temperature sweeps at higher T (suggest T ∈ {4, 5, 6}) because T=1 is too sharp to carry much “dark knowledge”.
- The uncleaned RAFDB-basic pipeline is now viable; missing-file failures were due to folder nesting, not model/code.

Next:
- Continue “one by one” uncleaned evaluations by generating/fixing manifests for the next dataset(s) (RAF-ML, FERPlus, FER2013-uniform-7, ExpW, AffectNet, etc.).
- Complete the remaining full unified test weight runs (0.3/0.7 and 0.7/0.3) under the same protocol.

---

## 2025-12-20 | Pick “best overall” ensemble softlabels folder
Intent:
- Decide which exported ensemble softlabels folder should be used as the default teacher targets for student KD/DKD.

Action:
- Scanned all `outputs/softlabels/**/ensemble_metrics.json` (49 runs) and ranked by macro-F1 (tie-break: accuracy).
- Selected the best run on the most representative benchmark (`fulltest`, i.e., unified multi-source test split).
- Updated the ensemble report to include an explicit recommendation section.

Result:
- Best overall (unified multi-source `fulltest`, logit+CLAHE):
	- `outputs/softlabels/_ens_rn18_0p3_b3_0p7_fulltest_logit_clahe_20251220_161909`
	- fulltest: accuracy ≈ 0.6824, macro-F1 ≈ 0.6534
- Best RAF-DB-specific teacher (RAFDB test, logit+CLAHE):
	- `outputs/softlabels/_ens_rn18_0p5_b3_0p5_rafdb_test_logit_clahe_20251220_154146`
	- RAFDB test: accuracy ≈ 0.8563, macro-F1 ≈ 0.7775
- AffectNet-full-balanced best among current runs:
	- `outputs/softlabels/_ens_affectnet_full_balanced_rn18_0p7_b3_0p3_logit_clahe_20251220_145024`

Decision / Interpretation:
- If the student is meant to generalize across datasets, default to the best `fulltest` ensemble softlabels.
- If the student is meant to maximize RAF-DB performance, prefer the RAF-DB-best softlabels folder instead.

Next:
- Start student KD/DKD using the chosen softlabels folder (unified vs RAF-DB target), with temperature sweep `T ∈ {4,5,6}`.
