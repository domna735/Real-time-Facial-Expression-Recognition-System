# Teacher performance on hard gates (eval_only / ExpW full / test_all_sources)

Date: 2026-02-09

This note answers the question: *“If the teacher is ~0.78–0.79 macro-F1, why does the real test look much worse?”* by running the **same Stage-A teacher checkpoints** on the **hard/mixed-domain gate manifests** used for student stress-testing.

## What was evaluated

Teachers (Stage A, img224):

- RN18: `outputs/teachers/RN18_resnet18_seed1337_stageA_img224/best.pt`
- B3: `outputs/teachers/B3_tf_efficientnet_b3_seed1337_pretrained_true_v1_stageA_img224/best.pt`
- CNXT: `outputs/teachers/CNXT_convnext_tiny_seed1337_stageA_img224/best.pt`

Hard-gate manifests:

- `Training_data_cleaned/classification_manifest_eval_only.csv`
- `Training_data_cleaned/expw_full_manifest.csv`
- `Training_data_cleaned/test_all_sources.csv`

All numbers are produced by `scripts/train_teacher.py --evaluate-only` and stored as `reliabilitymetrics.json` under `outputs/evals/teachers/...`.

## Summary table

The consolidated table is here:

- `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`

Key observations (from that table):

- **Stage-A validation (~0.79 macro-F1) is not a hard gate.** The hard gates are intentionally cross-domain / mixed-domain and are expected to be lower.
- On `eval_only`, macro-F1 drops to roughly **0.37–0.39**.
- On `expw_full`, macro-F1 drops to roughly **0.37–0.41**, with **Fear/Disgust especially low**.
- On `test_all_sources`, macro-F1 is much higher (**~0.62–0.65**) because this benchmark is closer to the multi-source training distribution than ExpW full.

## Interpretation

This confirms the earlier interpretation note:

- The teacher’s reported ~0.78–0.79 macro-F1 is an **in-distribution, source-filtered Stage-A validation** result.
- The “real test looks worse” because the evaluation distribution is different: harder labels, different capture conditions, and greater domain shift (especially for ExpW).

Related note:

- `research/issue__teacher_metrics_interpretation__20260209.md`
