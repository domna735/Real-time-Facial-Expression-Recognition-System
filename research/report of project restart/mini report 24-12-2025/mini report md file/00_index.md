# Report Pack Index (24-12-2025)
Date: 2025-12-24

This file is the table of contents for the mini-reports in this folder.

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
11. [11_analytical_comparison_tradeoff_report.md](11_analytical_comparison_tradeoff_report.md)

---

## Feb 2026 addendum pointers (new evidence)

The Dec-24 mini reports reflect the *reconstruction baseline*. Subsequent work (Jan–Feb 2026) added evidence-backed benchmarks, diagnostics, and paper-comparison notes.

Primary artifacts / reports:

- Offline benchmark suite results (CSV + index):

  - `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
  - `outputs/benchmarks/offline_suite__20260208_192604/benchmark_index.json`
- “Bad dataset” investigation report (eval-only, ExpW, FER2013-uniform):

  - `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`
- Final report addendum sections (Week 2):

  - `research/final report/final report.md` → Sections **9.3.5** and **9.3.6**

- Paper comparison deliverables (professor-facing):

  - One-page table: `research/paper_vs_us__20260208.md`
  - Paper protocol/metric extraction notes: `research/paper_metrics_extraction__20260208.md`

- FER2013 (folder dataset under `Training_data/FER2013`) additional benchmark (not official FER2013 split):

  - Manifest: `Training_data/fer2013_folder_manifest.csv`
  - Count summary: `outputs/manifest_counts__fer2013_folder.md`
  - Eval artifact example (DKD checkpoint): `outputs/evals/students/DKD_20251229_223722__fer2013_folder__test__20260208/reliabilitymetrics.json`
- Domain shift + real-time demo evidence:

  - live scoring artifacts under `demo/outputs/*/score_results.json`
  - summary note: `research/Real time demo/real time demo report.md`

- Supervisor-aligned “analytical comparison” framing (trade-off analysis):

  - `research/final report/final report.md` → Section **6.1**
  - Mini-report: `research/report of project restart/mini report 24-12-2025/mini report md file/11_analytical_comparison_tradeoff_report.md`
