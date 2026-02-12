# Recovery Playbook (Windows) — Real-time Facial Expression Recognition System

This document is a practical recovery guide for rebuilding or restoring the project after a machine reset, disk failure, or copying the repository to a new PC.

It covers multiple situations:

1) You have **old/original training data** (best-case)
2) You have **different/new training data** (rebuild with new datasets)
3) You have **only code** and need to reset the Python environment (`.venv`) and rerun minimal checks
4) Common edge cases (missing `outputs/`, missing cleaned manifests, GPU mismatch)

---

## 0) What “recovery” means in this repo

The project has three independent “layers”:

- **Code layer**: `src/`, `scripts/`, `tools/`, `demo/`, `docs/`, `research/`, plus `requirements*.txt`
- **Data layer**: `Training_data/` (raw/original) and `Training_data_cleaned/` (cleaned + manifests)
- **Artifacts layer**: `outputs/` (trained checkpoints, metrics JSONs, softlabels, validation outputs)

You can restore the project at different levels:

- **Code-only recovery**: can run scripts, but no results or datasets.
- **Rebuildable recovery**: you can regenerate manifests + retrain if you still have datasets.
- **Full-state recovery**: you restore `outputs/` so results are immediately available without retraining.

---

## 1) Quick checklist (what you have)

Before choosing a path, check what exists:

- Code exists? (repo folders `src/`, `scripts/`, `tools/`)
- Raw datasets exist? (`Training_data/`)
- Cleaned datasets + manifests exist? (`Training_data_cleaned/`)
- Artifacts exist? (`outputs/`)
- Python env exists? (`.venv/`) — optional (can rebuild)

---

## 2) Common prerequisites (all situations)

### 2.1 Use repo root

Open PowerShell in the repo root:

- `C:\Real-time-Facial-Expression-Recognition-System_v2_restart`

### 2.2 Create / rebuild Python environment (recommended even if `.venv` exists)

If you want a clean reset:

1. Delete the old venv folder (optional but recommended):
   - Delete `.venv/`

2. Create a new venv:

```powershell
py -m venv .venv
```

3. Activate it:

```powershell
. .\.venv\Scripts\Activate.ps1
```

4. Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

5. Install dependencies:

- For standard CUDA setup (default):

```powershell
python -m pip install -r requirements.txt
```

- If using DirectML stack (only if you know you need it):

```powershell
python -m pip install -r requirements-directml.txt
```

### 2.3 Check GPU / device

```powershell
python scripts/check_gpu.py
```

If you are using DirectML:

```powershell
python scripts/check_gpu_dml.py
```

---

## 3) Situation A — You have OLD/original training data (best-case)

### Goal
Restore the project so it can reproduce the full pipeline:

- Clean/validate datasets
- Train teachers
- Sweep ensemble / export softlabels
- Train student CE → KD → DKD
- Run real-time demo

### Steps

#### Step A1 — Confirm raw datasets exist

- Check `Training_data/` contains the datasets you used previously (AffectNet balanced, FERPlus, FER2013 uniform 7, RAF-DB basic, ExpW, etc.).

If these are missing, jump to **Situation B**.

#### Step A2 — (Re)build cleaned manifests + cleaned structure

The cleaning tools live under `tools/data/`.

Typical sequence (high-level):

- Build/merge manifests
- Clean into canonical 7-class format
- Validate manifest integrity

Minimum required outputs:

- `Training_data_cleaned/classification_manifest.csv`
- `Training_data_cleaned/classification_manifest_hq_train.csv`
- Validation JSONs in `outputs/` (e.g., `outputs/manifest_validation_all_with_expw.json`)

If you already have `Training_data_cleaned/` and trust it, you can skip to A3.

Validation (recommended):

```powershell
python tools/data/validate_manifest.py --csv Training_data_cleaned/classification_manifest.csv
python tools/data/validate_manifest.py --csv Training_data_cleaned/classification_manifest_hq_train.csv
```

#### Step A3 — Train teachers (RN18 / B3 / CNXT)

Use the teacher training script:

- `scripts/train_teacher.py`

You likely have PowerShell runners in `scripts/` for overnight training. If not, run the Python script directly with your chosen args.

After training, confirm each run folder under `outputs/teachers/` contains:

- `best.pt`
- `history.json`
- `reliabilitymetrics.json`
- `calibration.json`
- `alignmentreport.json`

#### Step A4 — Ensemble selection + softlabels export

Use the tools/scripts that export softlabels into `outputs/softlabels/`.

Confirm each softlabels folder contains:

- `softlabels.npz`
- `softlabels_index.jsonl`
- `ensemble_metrics.json` (for the benchmark run)
- `alignmentreport.json`

#### Step A5 — Train student (CE → KD → DKD)

Use:

- `scripts/train_student.py`

Confirm each student run folder under `outputs/students/` contains:

- `best.pt`
- `history.json`
- `reliabilitymetrics.json`
- `calibration.json`

#### Step A6 — Real-time demo

Run:

- `demo/realtime_demo.py`

Check:

- `demo/outputs/` contains CSV logs

---

## 4) Situation B — You have DIFFERENT/NEW training data

### Goal
Recover the pipeline, but **results will differ**. The key is to rebuild manifests correctly and keep the canonical 7-class mapping.

### Steps

#### Step B1 — Put new datasets into `Training_data/`

- Keep dataset folders organized per dataset.
- Make sure paths are stable (avoid moving folders mid-project).

#### Step B2 — Adapt/extend cleaning scripts if needed

Your new dataset may have different:

- label names
- folder structure
- file formats

The adapters live in `tools/data/`.

Target outputs must still be canonical 7 classes:

- `Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral`

#### Step B3 — Rebuild cleaned manifests + validate

Rebuild:

- `Training_data_cleaned/classification_manifest.csv`
- `Training_data_cleaned/classification_manifest_hq_train.csv`

Then validate with `tools/data/validate_manifest.py` (same as Situation A).

#### Step B4 — Re-train teachers / ensemble / student

Same as Situation A (A3–A6). Expect different scores.

---

## 5) Situation C — You have ONLY CODE and need to reset `.venv`

### Goal
Make the repo runnable again (lint-free is not required), verify the environment and scripts launch.

### Steps

#### Step C1 — Recreate `.venv` and install requirements

Follow Section 2.2.

#### Step C2 — Sanity checks

- Device check:

```powershell
python scripts/check_device.py
python scripts/check_gpu.py
```

- Quick smoke training (if it exists and does not require huge datasets):

```powershell
python scripts/smoke_train_rn18.py --help
```

If you have no data, you can only verify that entrypoints import successfully.

#### Step C3 — What you cannot do without data

Without `Training_data/` or `Training_data_cleaned/`, you cannot:

- regenerate manifests
- train teachers/students
- export softlabels
- run meaningful evaluations

---

## 6) Situation D — You have code + cleaned manifests, but NOT raw datasets

This depends on how `Training_data_cleaned/` was produced.

- If cleaned data is real copies of images: you may still train.
- If cleaned data was created by links (common in this repo): the cleaned structure may break if the original files are missing.

### Minimum check

Validate a sample of cleaned paths:

```powershell
python tools/data/validate_manifest.py --csv Training_data_cleaned/classification_manifest.csv
```

If it reports many missing paths, you must restore raw datasets (Situation A or B).

---

## 7) Situation E — You have everything except `outputs/` (lost results)

You can fully recover by re-running the pipeline:

1) Validate manifests
2) Train teachers
3) Export softlabels
4) Train student

But you cannot “restore old numbers” unless you exactly match:

- datasets
- seeds
- code version
- hyperparameters
- hardware/software environment

Best practice: back up `outputs/` if you want fast, exact restoration.

---

## 8) Situation F — You restored `outputs/` but checkpoints fail to load

Symptoms may include `pickle.UnpicklingError` from `torch.load`.

Common causes:

- partial/corrupted checkpoint copy
- mismatched PyTorch versions
- incomplete file transfer

Recovery steps:

1) Re-copy the checkpoint from the backup (prefer robocopy / verified transfer)
2) Ensure you installed the same major PyTorch version used originally
3) Try loading with `map_location="cpu"` (for debugging)
4) If still broken: retrain that run

---

## 9) Recommended local backup sets (for future safety)

### Minimal backup (smallest; rebuild results later)

- `src/`, `scripts/`, `tools/`, `demo/`, `docs/`, `research/`
- `requirements.txt`, `requirements-directml.txt`

### Fast restore backup (recommended)

Everything in “Minimal backup” plus:

- `outputs/`
- `Training_data_cleaned/`

### Full backup (largest; most reliable)

Everything above plus:

- `Training_data/`

Notes:

- `.venv/` is optional (rebuildable)
- `__pycache__/` is not needed

---

## 10) Quick command summary (PowerShell)

Create venv + install:

```powershell
py -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Validate manifests:

```powershell
python tools/data/validate_manifest.py --csv Training_data_cleaned/classification_manifest.csv
python tools/data/validate_manifest.py --csv Training_data_cleaned/classification_manifest_hq_train.csv
```

GPU check:

```powershell
python scripts/check_gpu.py
```
