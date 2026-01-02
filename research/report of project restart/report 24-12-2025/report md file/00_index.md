# Report Pack Index (24-12-2025)
Date: 2025-12-24

This file is the table of contents for the 10 mini-reports in this folder.

## How to reproduce (high level)
- Teachers (Stage A): train via `scripts/train_teacher.py` (or the provided PowerShell overnight runners under `scripts/run_teachers_overnight*.ps1`).
- Ensembles / softlabels: export via `scripts/export_multi_ensemble_softlabels.py` (triage helper: `tools/triage_softlabel_runs.py`).
- Student CE→KD→DKD: run the PowerShell runner `scripts/run_student_mnv3_ce_kd_dkd.ps1`.
- Metrics / calibration JSONs are emitted into each run folder (e.g., `reliabilitymetrics.json`, `calibration.json`, `history.json`).

Key artifact folders used by this report pack:
- Teachers: `outputs/teachers/`
- Softlabels: `outputs/softlabels/`
- Students: `outputs/students/`
- Cleaned manifests: `Training_data_cleaned/`

## Mini-reports
1. [01_teacher_model_training_report.md](01_teacher_model_training_report.md)
2. [02_teacher_model_ensemble_report.md](02_teacher_model_ensemble_report.md)
3. [03_student_model_training_report.md](03_student_model_training_report.md)
4. [04_dataset_cleaning_report.md](04_dataset_cleaning_report.md)
5. [05_dataset_usage_report.md](05_dataset_usage_report.md)
6. [06_basic_demo_report.md](06_basic_demo_report.md)
7. [07_alignment_reproducibility_report.md](07_alignment_reproducibility_report.md)
8. [08_evaluation_protocol_report.md](08_evaluation_protocol_report.md)
9. [09_realtime_pipeline_report.md](09_realtime_pipeline_report.md)
10. [10_nl_negl_research_note.md](10_nl_negl_research_note.md)
