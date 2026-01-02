0. Freeze dataset index + integrity gates (do this first)
    - Status (Dec 24, 2025): ✅ Done for the current manifests
    - Goal: make training reproducible and prevent silent data drops.
    - Frozen manifests (current):
       - `Training_data_cleaned/classification_manifest.csv` (validated; includes ExpW)
       - `Training_data_cleaned/classification_manifest_hq_train.csv` (used for student CE/KD/DKD)
       - `Training_data_cleaned/test_all_sources.csv` (mixed-source benchmark)
    - Mandatory checks (fail-fast):
       - paths exist; image decodes; label ∈ {Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral}
    - Evidence artifacts:
       - `outputs/manifest_validation_all_with_expw.json` (466,284 rows; 0 missing; decode sample recorded)
       - `outputs/manifest_validation.json`

1. Rebuild minimum teacher models
   - Models: ResNet18, ResNet50, EfficientNet-B3, ViT, ConvNeXt-Tiny
    - Purpose: generate strong/calibrated teachers for KD/DKD
    - Match interim report teacher protocol where possible:
       - ArcFace head (m=0.35, s=30), warmup + margin schedule, balanced sampling, effective-number weighting
       - Optimizer: AdamW (lr=3e-4, wd=0.05), cosine schedule w/ warmup, ~60 epochs
       - Augmentation: random crop/flip/color jitter + CLAHE (and whatever you used originally)
      - Status (Dec 24, 2025):
          - ✅ Stage A (img224) recovered: RN18, B3, CNXT
             - `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/`
             - `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/`
             - `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/`
          - ✅ RN18 Stage B (img384) exists (currently stored under “test and other data”):
             - `outputs/teachers/test and other data/stage B/RN18_resnet18_seed1337_stageB_img384/best.pt`
          - ❌ Full B3 Stage B (img384) not recovered as a top-level run folder yet
          - ❌ ViT Stage A/B not recovered as a top-level run folder yet
      - Required artifacts (per run folder): alignmentreport.json, calibration.json, reliabilitymetrics.json, history.json, best.pt

2. Rebuild ensumable teacher models
    - Pairwise (e.g., RN18+B3) and 3-teacher ensembles are recovered; “four-way split” orchestration is not yet rebuilt.
    - For multi-teacher KD/DKD training
    - Status (Dec 24, 2025): ✅ Core ensemble export + evaluation exists
       - Softlabels export dir used for student KD/DKD:
          - `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856/`
    - Required artifacts (current naming):
       - `softlabels.npz`, `softlabels_index.jsonl`
       - `classorder.json`, `hash_manifest.json`, `alignmentreport.json`
       - `ensemble_metrics.json`

3. Rebuild student model (core group only)
   - Backbone: MobileNetV3-Large (timm mobilenetv3large100)
   - KD/DKD training with seeds (1337, 2025, 42)
   - Match interim report student protocol where possible:
     - KD: α≈0.5, T=2.0 with T^2 scaling
     - DKD: α≈0.5, β≈4.0, T=2.0 with T^2 scaling
     - Evaluate macro-F1, per-class F1, minority-F1, ECE/NLL/Brier
    - Status (Dec 24, 2025): ✅ One full seed (1337) end-to-end recovered
       - CE/KD/DKD run folders:
          - `outputs/students/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/`
          - `outputs/students/mobilenetv3_large_100_img224_seed1337_KD_20251223_225031/`
          - `outputs/students/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/`
    - Next: run the same pipeline for seed=42 and seed=2025 and produce a cross-seed summary CSV.
    - Required artifacts (current naming): best.pt, history.json, reliabilitymetrics.json, calibration.json

4. Rebuild real-time demo
   - Manual labeling (keyboard mapping, clickable bar → per-frame CSV + events CSV)
   - Video manual labeling (video input + manual events logging)
   - Real-time parameter adjustment (EMA α, hysteresis δ, vote window/min-count, emo-ratio overlay)
   - Detection methods: YuNet (preferred), DNN, Haar
    - Status (Dec 24, 2025): ✅ Manual labeling + CSV artifacts exist (teacher demo)
       - Example: `demo/outputs/test_RN18_stageA/` contains `demoresultssummary.csv`, `per_frame.csv`, `events.csv`, `thresholds.json`
    - Remaining gap: student checkpoint demo run + timed KPI report (FPS/latency/flip-rate)
    - Required artifacts: demoresultssummary.csv, thresholds.json (optional), logit_bias.json (optional)

5. Begin NL + NegL training (student model)
   - NL safeguards: AMP, memory downsizing, gradient accumulation, KD ramp, gradient clipping
   - NegL design: teacher confusion matrix guidance, class-aware ratio, uncertainty gate, minority protection
   - Required artifacts: neglconfig.json, neglrules.md, NL smoke logs

Additional artifacts needed for reproducibility  
- Alignment artifacts: alignmentreport.json, classorder.json, hash_manifest.json for each teacher softlabel export  
- Standardized calibration.json schema across teacher and student models  
- ONNX + latency benchmarks: onnxexportreport.json, onnxparity.json, latencybenchmarks.csv  
- Domain adaptation validation set: small webcam validation set with before/after logs, fairness checks (lighting/pose)
- Training logs and summaries: comprehensive logs for all training runs, including hyperparameters and performance metrics.

Report pack outputs (Dec 24, 2025)
- Consolidated report: `research/report 24-12-2025/report of 24 12 2025.md`
- Interim report (v3): `research/report 24-12-2025/Interim Report/version 3 Real-time-Facial-Expression-Recognition-System Interim Report (24-12-2025).md`