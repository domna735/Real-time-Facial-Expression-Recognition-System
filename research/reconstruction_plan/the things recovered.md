# The things recovered (v2 restart)

Date: 2025-12-24  
Workspace: `C:\Real-time-Facial-Expression-Recognition-System_v2_restart`

## Purpose / 目的

This report summarizes what has already been recovered (rebuilt and verified) in the v2-restart repository, what is still missing compared to the old interim report, and the concrete next recovery steps.

本報告總結：目前 v2-restart 已經「成功重建並驗證」嘅部分、仍未重建/未完成嘅部分（相對舊 interim report），以及下一步要點做。

---

## Quick status table (Recovered vs Missing)

Legend: ✅ done / ⚠ partial / ❌ not done

### A) Teachers

- RN18
  - Stage A (224): ✅ done (`outputs/teachers/RN18_resnet18_seed1337_stageA_img224/best.pt`)
  - Stage B (384): ✅ done (currently stored under `outputs/teachers/test and other data/stage B/`)
    - `outputs/teachers/test and other data/stage B/RN18_resnet18_seed1337_stageB_img384/best.pt`
- EfficientNet-B3
  - Stage A (224, pretrained=true retrain): ✅ done (`outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/best.pt`)
  - Stage B (384): ❌ not recovered as a complete top-level run folder yet
- ConvNeXt-Tiny
  - Stage A: ✅ done (`outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/best.pt`)
  - Stage B: ❌ not started
- ViT
  - Stage A: ❌ not started
  - Stage B: ❌ not started
- RN50 (optional teacher in old report narrative): ⚠ not part of the main Dec-24 report pack; older/experimental artifacts may exist under `outputs/teachers/test and other data/`.

### B) Teacher Ensemble

- Minimal ensemble (RN18 + B3): ✅ done (export + eval + alignment artifacts)
- Softlabels export: ✅ done (many runs under `outputs/softlabels/`)
- Alignment check: ✅ done (alignment artifacts written per run)
- Metadata (weights, class order, manifest hash): ✅ done (`alignmentreport.json`, `classorder.json`, `hash_manifest.json`)
- “Best overall softlabels folder” decision: ✅ done (documented in the ensemble report)

### C) Student (minimal viable)

- Student v0 training (MobileNetV3-Large): ✅ done (seed=1337)
  - CE: `outputs/students/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/`
  - KD: `outputs/students/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/`
  - DKD: `outputs/students/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/`
- KD baseline (α=0.5, T=2): ✅ done (folder above)
- DKD baseline (α=0.5, β=4, T=2): ✅ done (folder above)
- Student checkpoint (`best.pt`): ✅ done
- Student calibration artifacts: ✅ done (`calibration.json` + `reliabilitymetrics.json` per run)

### D) Real-time demo

- Teacher checkpoint live inference (webcam) + manual labeling + CSV artifacts: ✅ done
- Load student checkpoint: 🟡 partial (student checkpoints exist; a dedicated logged demo run using a student ckpt is not yet recorded under `demo/outputs/`)
- Webcam inference smoke test (student): 🟡 partial (pipeline supports model loading; needs a recorded run + CSV summary)
- Label overlay + UI correctness + per-frame/events logs: ✅ done
- FPS / latency benchmark (ONNX FP16 etc): ❌ not done

### E) KD / DKD / NL / NegL (late stage)

- KD/DKD pipeline code exists: ✅ implemented in `scripts/train_student.py`
- DKD baseline experiments: ✅ done (seed=1337 run folder exists)
- NL module scaffolding: ✅ scaffold exists (`src/fer/nl/`)
- NegL module scaffolding: ✅ scaffold exists (`src/fer/negl/`)
- NL + NegL + KD/DKD integration into training: ❌ not done
- Hard-case validation set: ❌ not done

### F) Data / Validation

- Cleaned datasets view + unified manifest: ✅ done (`Training_data_cleaned/` + `Training_data_cleaned/classification_manifest.csv`)
- Missing-file bias prevention (exporter warns/drops missing): ✅ done in ensemble export tooling
- Uncleaned RAFDB-basic manifest path fix (`aligned/aligned`): ✅ done (`Training_data/uncleaned_manifests/rafdb_basic_manifest.csv`)
- Small “hard validation set”: ❌ not done
- Webcam validation subset: ❌ not done

---

## What has been recovered (details)


### 1) Reproducible manifests + data integrity gates (Recovered)

- Cleaned, standardized 7-emotion datasets under `Training_data_cleaned/`.
- Unified manifest `Training_data_cleaned/classification_manifest.csv` validated (path existence + decode sampling outputs under `outputs/manifest_validation*.json`).
- ExpW HQ / ExpW full manifests created and validated.
- Export tooling avoids silent “missing files = biased subset” by warning/dropping missing rows.

This is the v2-restart equivalent of the old report’s “path crisis + integrity recovery” principle.

### 2) Teacher training entrypoints + artifacts (Recovered for RN18 + B3 Stage A)

Implemented and validated:

- `scripts/train_teacher.py` supports ArcFace protocol, warmup/ramp, cosine+warmup scheduling, checkpointing (`checkpoint_last.pt`), and artifacts.
- RN18 Stage A & Stage B complete with artifacts (Stage B checkpoint currently stored under `outputs/teachers/test and other data/stage B/`):
  - `alignmentreport.json`, `calibration.json`, `reliabilitymetrics.json`, `history.json`
  - `best.pt`, `best.onnx`, `last.onnx`
- B3 Stage A completed with corrected setting parity (pretrained=true retrain).

### 3) Teacher ensemble exporter + aligned softlabels (Recovered)

- `scripts/export_ensemble_softlabels.py` exports:
  - `softlabels.npz`, `softlabels_index.jsonl`
  - `classorder.json`, `hash_manifest.json`, `alignmentreport.json`
  - `ensemble_metrics.json`
- Both prob-space and logit-space fusion supported; logit-space used as the default recommendation for KD “logits/T semantics”.

### 4) “Best overall” softlabels folder chosen (Recovered)

Decision is already made and documented in the ensemble report:

- Overall best for unified multi-source student training:
  - Dec-24 pack uses: `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856/` (for student KD/DKD)
- RAF-DB-targeted best:
  - `outputs/softlabels/_ens_rn18_0p5_b3_0p5_rafdb_test_logit_clahe_20251220_154146`

Note: mixed-source benchmark selection evidence for RN18/B3/CNXT 0.4/0.4/0.2 is recorded in the report pack under `research/report 24-12-2025/report md file/02_teacher_model_ensemble_report.md`.

### 5) Real-time demo core loop + labeling artifacts (Recovered for teacher inference)

- Demo outputs are produced correctly (example folder):
  - `demo/outputs/test_RN18_stageA/` contains `per_frame.csv`, `events.csv`, `demoresultssummary.csv`, `thresholds.json`, `per_class_correctness.csv`.
- YuNet ONNX download issue (Git LFS pointer) fixed so demo can run reliably.

---

## What is NOT yet recovered (details)


### 1) Teachers not finished / not started

This is the biggest remaining gap vs the old interim report’s “teacher suite”.

- B3 Stage B: ❌ not done
- B3 Stage B: ❌ not recovered as a complete top-level run folder yet
  - Some Stage-B smoke artifacts exist under `outputs/teachers/test and other data/old data/` but are not the primary run folders referenced in the Dec-24 report pack.
- ConvNeXt-Tiny Stage A/B: ❌ not started
- ViT Stage A/B: ❌ not started
- RN50: ⚠ code supports training it, but no finished run exists.

**Note on ConvNeXt sensitivity (your reminder):**
ConvNeXt is the right backbone to “stress test” the pipeline settings because it is sensitive to augmentation/LR/warmup/margin schedule. But at this moment it is not trained yet, so the system is not fully recovered to the old report’s breadth.

### 2) Students (core deliverable) are still missing

This gap has been closed for a first seed (1337): CE/KD/DKD run folders exist with `best.pt` and `reliabilitymetrics.json`.

Remaining student gaps vs the old report:

- Multi-seed experiments (42, 2025) and a consolidated cross-seed summary
- Mixed-source evaluation of the student checkpoints on `Training_data_cleaned/test_all_sources.csv`

### 3) Four-way split KD + multi-teacher diversity (not recovered)

Old report emphasizes Four-Way Split KD and class-aware teacher assignments.

In v2-restart right now:

- Minimal RN18+B3 ensemble is recovered.
- Four-way split teacher set (RN18 + ConvNeXt + B3 + ViT) is **not possible yet** because ConvNeXt/ViT teachers are not trained.

### 4) NL / NegL are only scaffolds

- Scaffolding files exist, but they are not wired into `scripts/train_student.py` to run real experiments.
- No NL smoke logs in `outputs/`.

### 5) Deployment engineering gaps (ONNX FP16 + latency benchmarks)

Old report includes ONNX FP16 export + latency goals.

Current:

- Teacher ONNX exists.
- Student ONNX + latency benchmark artifacts (e.g. `onnxexportreport.json`, `onnxparity.json`, `latencybenchmarks.csv`) are ❌ not done.

### 6) Domain adaptation & validation sets (hard/webcam)

Old report mentions webcam domain gap and a custom webcam dataset/validation.

Current:

- Demo can be labeled and logged.
- But we have not built:
  - a small hard-case validation set
  - a webcam validation subset
  - before/after “gap closure” logs

---

## Important differences vs the old interim report (why numbers won’t match)

If you try to directly compare old table metrics to new runs, note:

- Old report: canonical benchmark is **RAF-DB test** with calibrated temperature (T*≈1.2).
- Current v2-restart: many evaluations were run across different splits (fulltest multi-source, ExpW, AffectNet, uncleaned RAFDB, etc.). Those are harder and not apples-to-apples.
- Dataset index scale changed (old report states ~228k; current unified manifest is larger).

So “recovering the system” should focus on recovering the **pipeline + artifacts + reproducible gates**, then re-running the exact old benchmark configuration when needed.

---

## Next recovery steps (dependency-ordered)


### Step 1 — Finish the teacher set needed for four-way split

1) Complete B3 Stage B (or decide a smaller Stage B image size / fewer epochs if laptop power-capped).
2) Train ConvNeXt-Tiny Stage A/B using conservative settings first (because it’s sensitive).
3) Train ViT Stage A/B only if still needed for diversity; otherwise optional.

### Step 2 — Run student v0 (MobileNetV3-Large) minimal KD

- Train student with one teacher source first (use the recommended `fulltest` ensemble softlabels if the goal is generalization).
- Produce required artifacts:
  - `best.pt`, `history.json`, `metrics.json`, `calibration.json`, `reliabilitymetrics.json`
- Then run DKD baseline.

### Step 3 — Realtime demo with student

- Load student checkpoint in demo, run webcam smoke test, and record:
  - per-frame logs, events logs, per-class correctness
- Only then proceed to ONNX FP16 export + latency bench.

### Step 4 — NL/NegL integration (after KD/DKD stable)

- Add NL/NegL into training with an explicit smoke gate.
- Build small hard-case + webcam validation subsets for fair comparisons.

---

## References (in-repo)

- Old interim baseline: `research/Real-time-Facial-Expression-Recognition-System Interim Report/version 2 Real-time-Facial-Expression-Recognition-System Interim Report old.md`
- Current ensemble report: `research/report of project restart/report of ensemble v2 restart.md`
- Week log: `research/process_log/Dec_week3_process_log.md`

Dec-24 report pack outputs:

- Consolidated report: `research/report 24-12-2025/report of 24 12 2025.md`
- Interim report (v3): `research/report 24-12-2025/Interim Report/version 3 Real-time-Facial-Expression-Recognition-System Interim Report (24-12-2025).md`
