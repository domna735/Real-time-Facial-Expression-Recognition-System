# Old Interim Report → Rebuild/Remake Checklist (Status Map)

This checklist maps the *claims / requirements* in:
- `research/Real-time-Facial-Expression-Recognition-System Interim Report/version 2 Real-time-Facial-Expression-Recognition-System Interim Report old.md`

to what exists in this repo now.

Legend:
- ✅ Done = implemented + has a repeatable script and expected artifacts
- 🟡 Partial = some parts exist, but not fully wired end-to-end or missing a key artifact
- ❌ Not done = not implemented (or no script / no artifacts)

---

## A) Dataset integrity & reproducibility gates

- ✅ Dataset manifest + hashing recorded per run
  - Evidence: teacher run artifact `alignmentreport.json` includes `manifest_sha256` and source split counts.
  - Example: `outputs/teachers/B3_tf_efficientnet_b3_seed1337_stageA_img224/alignmentreport.json`

- ✅ “Fail-fast integrity gates” (paths exist / decode / label set)
  - Status: implemented as a dedicated validator and used to generate frozen validation artifacts.
  - Script: `tools/data/validate_manifest.py`
  - Evidence artifacts:
    - `outputs/manifest_validation_all_with_expw.json` (466,284 rows; 0 missing paths; decode sampling recorded)
    - `outputs/manifest_validation.json`

---

## B) Teacher training protocol (ArcFace)

- ✅ Teacher training script with ArcFace protocol + calibration artifacts
  - Script: `scripts/train_teacher.py`
  - Outputs (per run):
    - `alignmentreport.json`, `history.json`, `calibration.json`, `reliabilitymetrics.json`
    - `checkpoint_last.pt`, `best.pt` (and periodic `checkpoint_epochXYZ.pt`)

- ✅ Resume semantics + provenance preserved
  - `--resume` or auto-resume from `<output-dir>/checkpoint_last.pt`
  - `--init-from` starts a NEW run for Stage B (epoch resets) while preserving init provenance in `alignmentreport.json`.

- ✅ Two-stage teacher runner for EfficientNet-B3 (Stage A + Stage B)
  - Script: `scripts/run_teachers_overnight_b3_2stage.ps1`
  - Policy:
    - Stage A: include FERPlus (224)
    - Stage B: exclude FERPlus (384) and init from Stage A

---

## C) Stage B (B3) — “exclude FERPlus + 384px + 60 epochs”

- ✅ Implemented and scripted
  - Runner: `scripts/run_teachers_overnight_b3_2stage.ps1`
  - Output dirs (seed=1337):
    - Stage A: `outputs/teachers/B3_tf_efficientnet_b3_seed1337_stageA_img224/`
    - Stage B: `outputs/teachers/B3_tf_efficientnet_b3_seed1337_stageB_img384/`

- 🟡 Recovery status in this repo (artifacts)
  - Stage A (B3) is present: `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/best.pt`
  - A full Stage B (B3) run folder under the top-level `outputs/teachers/` is not currently present.
    - Note: some Stage-B smoke artifacts exist under `outputs/teachers/test and other data/old data/`.

- Notes:
  - Stage B folder doesn’t need to exist beforehand; the script creates it.
  - If Stage B already has `checkpoint_last.pt`, it will resume Stage B without re-initializing.

---

## D) Softlabel export & alignment safety

- ✅ Teacher logits export aligned to dataset rows (shuffle-safe)
  - Script: `scripts/export_softlabels.py`
  - Expected outputs:
    - `softlabels.npz`
    - `softlabels_index.jsonl` (keyed by `image_path`)
    - alignment/class order/hash artifacts

- ✅ Alignment diagnostics
  - Script: `scripts/diagnose_alignment.py`

---

## E) Student training (KD / DKD)

- ✅ Student training script supports CE / KD / DKD
  - Script: `scripts/train_student.py`
  - Expected outputs: `history.json`, `calibration.json`, `reliabilitymetrics.json`, checkpoints.

- ✅ Student CE/KD/DKD runs recovered (end-to-end)
  - CE: `outputs/students/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/`
  - KD: `outputs/students/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/`
  - DKD: `outputs/students/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/`
  - Evidence: each run contains `best.pt` and `reliabilitymetrics.json` (plus calibration artifacts).

- ✅ Reliability metrics tool (ECE/NLL/Brier + temperature scaling)
  - Script: `scripts/compute_reliability.py`

---

## F) Multi-teacher / “Four-Way Split KD”

- 🟡 Partially reproduced
  - Current status: we have the core building blocks (per-teacher softlabels, student KD/DKD, alignment safety).
  - Missing: a dedicated orchestration script that implements the full *Four-Way Split KD* strategy described in the report (data partitioning + per-part teacher assignment + summary artifacts).

---

## G) Calibration at inference + per-class thresholds + logit bias

- ✅ Calibration is computed and saved in offline training artifacts
  - Teacher: `scripts/train_teacher.py` writes `calibration.json`
  - Student: `scripts/train_student.py` writes `calibration.json`

- ❌ Real-time demo does NOT apply temperature scaling at inference
  - Demo: `demo/realtime_demo.py`
  - Status: demo logs optional `thresholds.json`, but does not load/apply `calibration.json`.

- ❌ Real-time demo does NOT support optional `logit_bias.json`
  - Report mentions using logit bias for minority classes; demo currently has no such loading/apply path.

---

## H) Real-time demo (manual labeling + protocol-lite scoring)

- ✅ Manual labeling + CSV artifacts exist
  - Demo: `demo/realtime_demo.py`
  - Outputs include: `per_frame.csv`, `events.csv` and optional `thresholds.json`.

- ✅ “Protocol-lite” scoring exists
  - Script: `scripts/score_live_results.py`

---

## I) NL / NegL

- 🟡 Scaffolds exist, not integrated into training loops
  - Code: `src/fer/nl/`, `src/fer/negl/`, `src/fer/utils/grad_accum.py`
  - Smoke: `scripts/smoke_nl.py`
  - Docs/config: `research/nl_negl/neglconfig.json`, `research/nl_negl/neglrules.md`

---

## K) Report pack / documentation parity (Dec 24, 2025)

- ✅ 10 mini-reports + consolidated report exist (artifact-grounded)
  - Mini-reports: `research/report 24-12-2025/report md file/00_index.md` through `10_nl_negl_research_note.md`
  - Consolidated report: `research/report 24-12-2025/report of 24 12 2025.md`
  - Interim report (v3): `research/report 24-12-2025/Interim Report/version 3 Real-time-Facial-Expression-Recognition-System Interim Report (24-12-2025).md`

## J) What you can run *now* (minimal commands)

### Continue / start B3 Stage B (60 epochs)

Use the existing two-stage runner but request Stage B only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_teachers_overnight_b3_2stage.ps1 -Stage B -ManifestPreset full -CudaDevice 0 -Epochs 60 -StageAImageSize 224 -StageBImageSize 384
```

If you hit CUDA OOM at 384px, retry with smaller batch and/or accumulation (effective batch = BatchSize×AccumSteps):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_teachers_overnight_b3_2stage.ps1 -Stage B -ManifestPreset full -CudaDevice 0 -Epochs 60 -BatchSize 24 -AccumSteps 2
```

---

## Next “remake” candidates (if you want full report parity)

1) Add a dedicated `FourWaySplitKD` orchestration script and artifacts.
2) Make the real-time demo optionally load/apply `calibration.json` (+ optional `logit_bias.json`).
3) Add an explicit dataset verification script so the integrity gates are one command.
