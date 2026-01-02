# Two-stage teacher model training vs old teacher model training

## Why this document exists
We rebuilt the teacher-model training pipeline so it is **reproducible, manifest-driven, and safe to run overnight** on Windows, while also fixing a key quality issue we observed when mixing **FERPlus (48×48)** with high-resolution training.

This file summarizes:
- What we have built/changed in the current codebase
- What the **old report** teacher training looked like
- The practical differences between **old single-stage teacher training** and the **current two-stage teacher training**


## What we have done (current rebuild)

### 1) Data + labels are now standardized and audit-able
- Canonical **7-class label set**: Angry / Disgust / Fear / Happy / Sad / Surprise / Neutral.
- Datasets are standardized into folder datasets and **CSV manifests** (unified + curated HQ + eval-only + ExpW variants).
- Training is **manifest-driven** with source filtering (include/exclude by `source`) so Stage A/B can be expressed cleanly.

### 2) Teacher training script is now “pipeline-grade”
Implemented/extended the teacher trainer to be stable and operational:
- ArcFace head support with a stability schedule (warmup / margin ramp).
- Balanced sampling (per-class minimum) + optional class-balanced weighting.
- Checkpointing (`checkpoint_last.pt`, `best.pt`) with `global_step` saved/restored.
- Calibration + reliability artifacts (e.g., ECE/Brier/NLL + reliability curve data).
- ONNX export outputs (and options like export-only / skip-during-train).

### 3) Long-run safety: output directory locking + lock-aware cleaning
To avoid “output dir deleted while training/export writes”:
- Each run writes a lock file `output_dir/.run.lock` while training/export is active.
- PowerShell runners refuse to delete locked output directories.
- Stale-lock recovery flags exist for cases like crash/power loss.


## Old teacher training (from old report PDFs + interim report)
The old teacher baseline (as described in the old teacher training report) can be summarized as:

### Model selection + final teacher
- Explored backbones: ResNet18/50, EfficientNet‑B3, ConvNeXt‑Tiny, ViT Tiny/Small.
- Final selected teacher ensemble: **ResNet18 + EfficientNet‑B3** with **weighted probability fusion**.
  - Typical chosen weight: **0.7 (RN18) : 0.3 (B3)**.
  - Note: the old teacher report uses **T=1.0** in the ensemble selection line (fusion temperature), and also applies **post-hoc calibration temperatures** (T*) later.

### Dataset + label spec
- Unified index included multiple FER sources (FERPlus, RAF-DB, AffectNet, etc.).
- Explicit 7-way labels: angry, disgust, fear, happy, neutral, sad, surprise.
- Strong emphasis on **alignment + dedup + provenance** (e.g., `_ixNextAffFull` in report text).

### Training objective + optimization (teacher)
- Loss: **ArcFace** chosen as primary objective.
  - ArcFace head: margin **m=0.35**, scale **s=30**.
  - Stabilization: **plain-logits warmup 5 epochs**, and **margin scheduling** (m: 0.0→0.35 over epochs 5–15).
- Optimizer: **AdamW**, typical values:
  - `lr=3e-4`, `weight_decay=0.05`
- Scheduler: **warmup → cosine decay**
  - Warmup: **2 epochs**
  - Total: **60 epochs** (teacher baseline)
- Input size + augmentation (teacher training): **224×224 random resize/crop**, flip, color jitter, and **CLAHE**.
- AMP: enabled in the report.

### Preprocessing (important for offline→online parity)
- Deployment preprocessing described in the old interim report includes: eye alignment, square crop (≈30% margin), **CLAHE (clip=2.0, tile=8×8)**, ImageNet normalization, resize to **224×224**.

### Calibration
- Post-hoc **temperature scaling** is explicitly part of the old pipeline:
  - Global teacher calibration: **T*≈1.2** (selected by validation NLL / ECE).
  - The old interim report also mentions **per-class temperature refinement** for minority classes (used after global scaling).


## Current teacher training (two-stage)

### Confirmed in the current code: warmup+cosine and temperature scaling
This repo’s current teacher trainer implements both items:
- **Warmup + cosine LR schedule**: LR is linearly warmed up for `--lr-warmup-epochs` (default 2), then cosine-decayed to ~0 by the end of training (step-based schedule).
- **Temperature scaling**: Every `--eval-every N` epochs (default 1), we fit a single scalar **T** on the validation split by minimizing **NLL** (LBFGS), then report metrics both **raw** and **temperature-scaled**, and write `calibration.json`.

ArcFace stabilization is also implemented:
- `--plain-logits-warmup-epochs` default 5
- `--margin-ramp-start` default 5, `--margin-ramp-end` default 15
- `--arcface-m` default 0.35, `--arcface-s` default 30

### Why two-stage exists (root cause)
FERPlus images are **48×48** (or near that). When we trained at **384×384** with FERPlus included, we observed quality degradation/instability because upsampling low-res data to high-res creates mismatched statistics and can dominate gradients in unhelpful ways.

So the final rule became:
- **Stage A @ 224**: include FERPlus (and other sources)
- **Stage B @ 384**: exclude FERPlus; fine-tune on higher-quality sources

### Stage A (224)
- Goal: learn robust general expression features while still benefiting from FERPlus signal.
- Uses the manifest/unified index with source filtering that includes FERPlus.

### Stage B (384)
- Goal: upgrade representation on high-resolution/HQ data without FERPlus resolution mismatch.
- Starts from Stage A weights using **new-run init** (not “resume the same run”).
- Uses source filtering to exclude FERPlus and focus on HQ sources.


## Observed results (this repo, Dec 2025 runs)

These numbers are taken directly from the saved run artifacts:
- `alignmentreport.json` (dataset counts + source filters)
- `history.json` (timing, accuracy, macro-F1)

### RN18 Stage A vs Stage B (actual artifacts)

#### Run folders (where the artifacts are)
- Stage A (224): `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/`
- Stage B (384): `outputs/teachers/test and other data/stage B/RN18_resnet18_seed1337_stageB_img384/`
  - Note: the `alignmentreport.json` inside this run still records its original `output_dir` as `outputs/teachers/RN18_resnet18_seed1337_stageB_img384/`, but this workspace currently stores the run under the archived `test and other data/stage B/` path.

#### Dataset + filters
- Stage A (224): `classification_manifest.csv` with `include_sources=[ferplus, rafdb_basic, affectnet_full_balanced, expw_hq]`
  - `train_rows=182,960`, `val_rows=18,165`
- Stage B (384): `classification_manifest.csv` with `exclude_sources=[ferplus]`
  - `train_rows=258,370`, `val_rows=27,017`

#### Timing + best validation macro-F1

| Run | Runtime (wall clock) | Best val macro-F1 |
|---|---:|---:|
| RN18 Stage A (224) | `last_total_sec=14781.35` (~4.11h) | `0.7902` (@epoch 22) |
| RN18 Stage B (384) *(partial; epochs logged = 20)* | `last_total_sec=7129.59` (~1.98h) | `0.6826` (@epoch 53) |

### Why we stop Stage B for now (Dec 2025 decision)

We are pausing Stage B runs because it is currently **low ROI** relative to student training:
- **Quality signal is negative for RN18**: Stage A already reaches a higher best val macro-F1 (`0.7902`) than Stage B (`0.6826`) on this pipeline’s validation.
- **Student training uses 224 + softlabels**: our distillation exports and realtime pipeline are currently built around the 224/CLAHE preprocessing; Stage B (384) teachers are not required to start KD/DKD.
- **Compute cost increases sharply for heavier backbones** (especially B3/ViT at 384), reducing iteration speed. That time is better spent on student KD/DKD runs and realtime validation.
- **We already have a strong Stage A teacher ensemble** and have completed HQ-train softlabel export for student training; further Stage B work can be revisited only if student results plateau.

Revisit criteria (when Stage B becomes worth it again):
- Student KD/DKD improvements stall, or realtime metrics lag despite strong offline metrics.
- We can run Stage B on a faster GPU server (or reduce Stage B resolution like 320/352) to control cost.


## Runtime issue found: EfficientNet‑B3 Stage B (384) can become multi-hour epochs

### What is happening
- B3 Stage B is correctly configured to exclude FERPlus (so it is not a “stuck” run caused by FERPlus 48×48 mismatch).
- The wall-clock cost is the problem: at 384×384, EfficientNet‑B3 is much heavier than RN18.
- On the laptop, the GPU appears compute-bound and power-capped during B3 Stage B.

Concrete evidence:
- The B3 Stage B epoch 0 timing is recorded as `epoch_sec≈13,856s` in `history.json`.
- A live `nvidia-smi` snapshot during the run shows ~99% GPU utilization with ~50W power draw (P0 state), consistent with a power-limited mobile GPU scenario.

### Why RN18 timings look “normal” but B3 does not
- RN18 is a small model; 384×384 at batch 64 is still fast.
- EfficientNet‑B3 at 384×384 increases compute sharply; if the laptop limits GPU power (battery / balanced mode / thermal limits), the slowdown can be extreme.

### What to do (practical options)
- Best option: run B3 Stage B on the remote GPU server.
- If staying local:
  - Ensure the laptop is plugged in and Windows/NVIDIA power mode is set to performance.
  - Consider reducing B3 Stage B image size (e.g., 320/352) or reducing Stage B epochs.
  - Consider disabling `--clahe` for Stage B if runtime remains unacceptable.


## Key differences (old vs current)

| Topic | Old teacher training (single-stage) | Current teacher training (two-stage) |
|---|---|---|
| Core schedule | Single-stage at 224 | Stage A 224 → Stage B 384 |
| FERPlus handling | Included in unified training (224) | Included only in Stage A; **excluded in Stage B** |
| Data control | Index/provenance emphasized, but runs were not fully “manifest-first” for every workflow | Manifest-driven datasets + explicit `source` filtering for Stage definition |
| Resume semantics | Standard checkpoint resume in a single run | Split into **resume** (continue same stage) vs **init-from** (start Stage B fresh from Stage A weights) |
| Operational safety | No explicit runtime lock against accidental deletion | `output_dir/.run.lock` + lock-aware PS1 cleaning |
| Outputs | Metrics and experiment folders referenced by report | Standardized artifacts per run (checkpoints + JSON metrics + ONNX exports + `calibration.json`) |
| Rationale | Best teacher selection + calibration for KD | Same goal, plus resolves FERPlus@384 mismatch + improves reproducibility |


## Practical “how to run” (current)

### Smoke verification (fast)
- Run the two-stage smoke script to verify both RN18 and B3 can complete Stage A→B and produce artifacts.

### Full run (60 epochs style)
- Use the overnight two-stage PowerShell runner.
- Resume behavior:
  - Re-run without cleaning: continues from `checkpoint_last.pt`.
  - Stage B initialization only happens if Stage B has no checkpoint yet.
  - Avoid deleting outputs while a run is active; the lock file prevents accidental cleanup.


## Notes / assumptions

## Reproducibility features implemented in the rebuild
These items are now present in the current pipeline (compared to the old report workflow):
- `--min-lr` floor support.
- `alignmentreport.json` records manifest SHA256 + filtered row counts + source filter policy.
- `environment.json` records Python/Torch/CUDA info and `pip freeze` snapshot.
- `--evaluate-only` and `--eval-manifest` support for evaluation runs.
- Runtime safety via `output_dir/.run.lock` and lock-aware cleaning.

## Notes / assumptions
- Old-teacher specifics are taken from the old interim report markdown and the teacher training PDF (ArcFace params, warmup+cosine schedule, 224 preprocessing, and temperature scaling).
- Current-teacher specifics are verified directly from the implementation in the repo’s teacher training script.
