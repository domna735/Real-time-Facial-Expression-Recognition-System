import re

file_path = r"c:\Real-time-Facial-Expression-Recognition-System_v2_restart\research\final report\final report version 3.md"

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

old_block = """## A.1 Evidence Inventory (Key Artifacts)

**Dataset and integrity:**
- `outputs/manifest_validation_all_with_expw.json`

**Teachers:**
- `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/reliabilitymetrics.json`
- `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/reliabilitymetrics.json`
- `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/reliabilitymetrics.json`

**Ensemble benchmark:**
- `outputs/softlabels/_archive/bad_list_20251223_121501/_ens_test_all_sources_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_111523/ensemble_metrics.json`

**Student runs:**
- CE: `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/reliabilitymetrics.json`
- KD/DKD: `outputs/students/_archive/2025-12-23/` (KD and DKD)

**Domain shift:**
- `demo/outputs/20260126_205446/score_results.json`
- `demo/outputs/20260126_215903/score_results.json`"""

new_block = """## A.1 Evidence Inventory Data Dictionary (Key Artifacts)

To facilitate offline reading and formal print submission, the fundamental data outputs from the JSON reproducibility artifacts referenced throughout this report are tabulated below. *(Note: Raw JSON files housing complete multi-class confusion matrices, prediction arrays, and hyperparameters remain archived in the project repository).*

### 1. Dataset Integrity Artifact (`manifest_validation_all_with_expw.json`)

| Metric | Recorded Value |
| :--- | :--- |
| **Total Rows (Images)** | 466,284 |
| **Missing Paths** | 0 |
| **Bad / Corrupt Labels** | 0 |
| **Primary Sources** | FERPlus (138k), FER2013-Uniform (140k), ExpW (91k), AffectNet (71k) |

### 2. Teacher Ensemble Offline Benchmarks (`ensemble_metrics.json`)

Evaluation of the frozen Teacher pseudo-labeler on the `test_all_sources.csv` split, combined via logits from RN18, EfficientNet-B3, and ConvNeXt-Tiny.

| Metric | Score |
| :--- | ---: |
| **Target Accuracy** | 0.687 |
| **Macro-F1 (7-class)** | 0.660 |
| **Highest Class F1** | Happy (0.839) |
| **Lowest Class F1** | Fear (0.535) |
| **Ensemble ECE** | 0.288 |

### 3. Student Calibration & Baseline Artifacts (`reliabilitymetrics.json`)

Comparative records validating the MobileNetV3 Student models (Cross-Entropy vs DKD) on the HQ-Train validation split. 

| Metric | CE Baseline | DKD Distilled |
| :--- | ---: | ---: |
| **Macro-F1** | 0.728 | 0.728 |
| **TS Expected Calibration Error (ECE)** | 0.050 | 0.027 |
| **TS Negative Log Likelihood (NLL)** | 0.793 | 0.791 |

### 4. Continuous Domain Shift (Webcam Replay) Artifacts (`score_results.json`)

Records from the dual-gate live deployment module measuring same-session webcam replays.

| Metric | Unadapted Baseline (205446) | Candidate NR-1 (215903) |
| :--- | ---: | ---: |
| **Smoothed Accuracy** | 0.588 | 0.527 |
| **Smoothed Macro-F1** | 0.525 | 0.467 |
| **Minority-F1 (lowest 3)** | 0.161 | 0.138 |
| **Jitter (flips/min)** | 14.86 | 14.16 |
| **Frames Evaluated** | 4,154 | 4,154 |"""

text = text.replace(old_block, new_block)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Appendix successfully updated with nice tables.")
