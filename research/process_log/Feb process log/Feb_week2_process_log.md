# Process Log - Week 2 of February 2026

This document captures daily activities, decisions, and reflections during the second week of February 2026, focusing on reconstructing the FER system under an evidence-first workflow.

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## 2026-02-08 | Diagnose “bad dataset” results (evidence-first)

Intent:
- Investigate why 3 benchmark datasets show unexpectedly low macro-F1: eval-only, ExpW, and FER2013 uniform-7.
- Produce artifact-backed diagnostics that directly point to a root cause hypothesis.

Action:
- Ran the full offline benchmark suite and exported CSV summaries (see suite index + CSV outputs).
- Generated a dedicated issue report with manifest audits + per-class F1 comparisons.
- Ran follow-up diagnostics on the CE student checkpoint:
	- Eval-only per-source breakdown (`--report-by-source`) to isolate which source dominates the drop.
	- CLAHE ablation (`--no-clahe`) on ExpW and FER2013 uniform-7 to test preprocessing mismatch.
	- Saved per-sample `preds.csv` and sampled error lists for manual inspection.

- Prepared professor-facing paper comparison deliverables:
	- Consolidated paper protocol/metric notes.
	- Built/updated a one-page “paper vs us” table with explicit comparability flags.
	- Added FER2013(msambare) as an additional *partial comparable* benchmark (folder split differs from official FER2013 split).

- Summarized the current status of NL/NegL screening runs (short-budget comparisons) and aligned the written conclusion to the stored compare artifacts (`outputs/students/_compare*.md`).

Result (artifacts):
- Offline suite outputs:
	- `outputs/benchmarks/offline_suite__20260208_192604/benchmark_index.json`
	- `outputs/benchmarks/offline_suite__20260208_192604/benchmark_results.csv`
- Bad-datasets issue report:
	- `research/issue__bad_results__evalonly_expw_fer2013__20260208.md`
- Eval-only (CE checkpoint) per-source breakdown + preds:
	- `outputs/diagnostics/bad_datasets/source_breakdown__CE__eval_only__clahe_on/reliabilitymetrics_by_source.json`
	- `outputs/diagnostics/bad_datasets/source_breakdown__CE__eval_only__clahe_on/preds.csv`
- CLAHE ablation (CE checkpoint):
	- ExpW full: `outputs/diagnostics/bad_datasets/clahe_ablation__CE__expw_full__clahe_on/` vs `...__clahe_off/`
	- FER2013 uniform-7: `outputs/diagnostics/bad_datasets/clahe_ablation__CE__fer2013_uniform7__clahe_on/` vs `...__clahe_off/`
- Error samples (CSV lists for inspection):
	- `outputs/diagnostics/bad_datasets/error_samples__CE__expw_full__fear_disgust__confident_wrong.csv`
	- `outputs/diagnostics/bad_datasets/error_samples__CE__expw_full__fear_disgust__ambiguous_wrong.csv`
	- `outputs/diagnostics/bad_datasets/error_samples__CE__fer2013_uniform7__fear__confident_wrong.csv`
	- `outputs/diagnostics/bad_datasets/error_samples__CE__eval_only__expw_hq__fear_disgust__confident_wrong.csv`

- Paper comparison deliverables:
	- One-page table: `research/paper_vs_us__20260208.md`
	- Paper protocol/metric extraction notes: `research/paper_metrics_extraction__20260208.md`

- FER2013(msambare) benchmark (not official split):
	- Manifest: `Training_data/fer2013_folder_manifest.csv`
	- Count summary: `outputs/manifest_counts__fer2013_folder.md`
	- DKD eval artifact: `outputs/evals/students/DKD_20251229_223722__fer2013_folder__test__20260208/reliabilitymetrics.json`

- Report updates linking and interpreting these artifacts:
	- `research/final report/final report.md` (addendum links + interpretation notes)
	- Mini-report pack index updated with Feb-2026 pointers: `research/report of project restart/mini report 24-12-2025/mini report md file/00_index.md`

Decision / Interpretation (based on stored evidence):
- Eval-only drop is strongly source-dependent: `expw_hq` is substantially worse than `expw_full` on macro-F1 and especially hurts Fear/Disgust.
- CLAHE is not the cause of the poor ExpW/FER2013 scores: turning CLAHE off makes both ExpW and FER2013 worse for macro-F1.
- FER2013 Fear remains a consistent failure mode even when balanced; likely a domain/preprocessing mismatch beyond calibration.

- NL/NegL screening (short-budget): NL(proto) appears stable but does not consistently improve macro-F1/minority-F1 vs KD/DKD baselines under tested configurations; NegL entropy-gating shows mixed effects and can regress TS calibration; synergy settings can improve raw loss/calibration signals but did not translate into macro-F1 gains in the recorded comparisons.

Next:
- Inspect the sampled error CSVs to categorize failure modes: label noise vs occlusion/pose vs low-res vs detector crop issues.
- Run the same diagnostics on the best teacher checkpoint(s) for comparison (per-source + CLAHE ablation).
- Plan mitigations aimed at long-tail + domain shift (class-balanced sampling/logit adjustment, targeted augmentations, and source-aware weighting).


- Strict FER2013 official-split evaluation (completed 2026-02-09 / 2026-02-11):
	- Converted `fer2013.csv` (Usage=PublicTest/PrivateTest) into official manifests and evaluated CE/KD/DKD checkpoints.
	- Summary: `outputs/benchmarks/fer2013_official_summary__20260209/fer2013_official_summary.md`
	- Paper-table updated to mark protocol mismatch (paper ten-crop vs our single-crop): `research/paper_vs_us__20260208.md`

- Paper-style comparison hardening:
	- Prefer protocol-matched evaluations (official splits + paper-reported metric). If a paper uses ten-crop, implement and report ten-crop separately rather than mixing protocols.

- Domain shift mitigation continuation:
	- Continue Self-Learning + NegL experiments with strict safety gating (eval-only + ExpW) and add standardized deployment-facing metrics (flip-rate/jitter + replay scoring) for decisions.

---

## 2026-02-09 | Supervisor clarification: “comparison” = analytical trade-off explanation

Intent:
- Correct a misunderstanding: the report does **not** need to “beat SOTA” on accuracy/macro-F1.
- Align the write-up to the FYP requirement: provide an **analytical comparison** explaining *why* results differ and whether the gap is reasonable under real-time constraints.

Action:
- Added an English “analytical comparison / trade-off analysis” section to the final report.
- Positioned SOTA papers as **reference points** for explaining gaps, not as a win/lose benchmark.
- Explicitly documented why direct 1:1 comparisons are often invalid (protocol mismatch: dataset split, label mapping, test-time augmentation, metric definition).

Result (artifacts / report updates):
- Final report section:
	- `research/final report/final report.md` → Section **6.1** (Analytical comparison vs papers)
- Evidence used to ground the explanation:
	- One-page comparability table: `research/paper_vs_us__20260208.md`
	- Protocol extraction notes: `research/paper_metrics_extraction__20260208.md`
	- Overall sanity table (CE vs KD vs DKD across hard gates): `outputs/benchmarks/overall_summary__20260208/overall_summary.md`

Decision / Interpretation:
- Our system is **deployment-oriented** (CPU real-time, stability, calibration, reproducibility, domain shift), so “lower than paper SOTA” can be fully reasonable.
- The academically correct comparison is to explain the trade-offs (capacity vs latency; curated dataset vs multi-source noise; offline-only vs webcam shift) and to claim protocol-matched numbers only when truly comparable.

Note:
- Teacher macro-F1 ≈ 0.78–0.79 is **real** but is measured on the Stage-A in-distribution validation split after source filtering; it should not be mistaken for the hard gate results. Clarification: `research/issue__teacher_metrics_interpretation__20260209.md`.

Additional action (same day): hard-gate evaluation of teachers

- Ran the three Stage-A teacher checkpoints on the same hard/mixed-domain gates used for student stress testing (`eval_only`, `expw_full`, `test_all_sources`).
- Consolidated a teacher-only summary table under `outputs/benchmarks/teacher_overall_summary__20260209/`.

Result (artifacts):
- Summary table: `outputs/benchmarks/teacher_overall_summary__20260209/teacher_overall_summary.md`
- Raw outputs per eval: `outputs/evals/teachers/overall__*__{eval_only|expw_full|test_all_sources}__test__20260209/reliabilitymetrics.json`
- Write-up note: `research/issue__teacher_hard_gates__20260209.md`

Communication (same day): clarification email to supervisor

- Sent an email update to Prof. Lam clarifying that the previously reported teacher macro-F1 ≈ 0.78–0.79 refers specifically to the **Stage-A in-distribution validation split** (as recorded in `alignmentreport.json`), and that cross-dataset / mixed-domain gates are expected to be much lower.
- Follow-up evidence attached in the repo via the hard-gate teacher benchmark artifacts listed above.

Key corrected takeaway (artifact-grounded):

- Stage-A in-distribution validation: macro-F1 ≈ 0.78–0.79 (see `outputs/teachers/*/reliabilitymetrics.json`).
- Hard gates (same checkpoints):
	- `eval_only`: macro-F1 ≈ 0.373–0.393
	- `expw_full`: macro-F1 ≈ 0.374–0.407
	- `test_all_sources`: macro-F1 ≈ 0.617–0.645

Next (reporting):

- Keep a strict “two-block” presentation in the report: (1) Stage-A val (in-distribution; filtered) and (2) hard-gate results (domain shift / mixed-domain), each with `n` and linked artifacts.

Additional action (same day): dataset provenance snapshots (Kaggle/Drive downloads)

Intent:
- Because some datasets are downloaded from Kaggle or shared drives (packaging may differ from “official split” definitions), snapshot the **exact local dataset copy** used for evaluation.

Action:
- Generated stable provenance fingerprints (file counts + stable SHA256 over the relative file list) for key dataset folders.

Result (artifacts):
- RAFDB-basic snapshot: `outputs/provenance/dataset_snapshot__RAFDB-basic__20260209.json`
- FER2013 folder snapshot: `outputs/provenance/dataset_snapshot__FER2013__20260209.json`
- ExpW snapshot: `outputs/provenance/dataset_snapshot__Expression in-the-Wild (ExpW) Dataset__20260209.json`
- Generator script: `scripts/snapshot_dataset_provenance.py`

---

## 2026-02-11 | FER2013 official split results + paper-comparison update

Intent:
- Close the “FER2013 official split pending” gap so paper comparison is evidence-backed and protocol-aware.

Action:
- Converted Kaggle/ICML-format `fer2013.csv` into official split manifests (PublicTest + PrivateTest).
- Evaluated the standard CE/KD/DKD student checkpoints on both splits (single-crop), writing `reliabilitymetrics.json` per run.
- Generated a consolidated summary table and updated paper-comparison documentation to avoid apples-to-oranges claims (paper ten-crop vs our single-crop).

Result (artifacts):
- Official manifests (n=3589 each):
	- `Training_data/FER2013_official_from_csv/manifest__publictest.csv`
	- `Training_data/FER2013_official_from_csv/manifest__privatetest.csv`
- Student eval outputs:
	- `outputs/evals/students/fer2013_official__*__*test__20260209/reliabilitymetrics.json`
- Consolidated summary:
	- `outputs/benchmarks/fer2013_official_summary__20260209/fer2013_official_summary.md`
	- (CSV/JSON alongside the MD in the same folder)

Decision / Interpretation:
- Official split is now available for fairer comparison, but strict numeric equivalence still depends on test-time protocol.
- Because the referenced FER2013 paper reports **ten-crop averaging**, our current **single-crop** official-split results are marked as protocol-mismatched and should be treated as gap-analysis evidence, not as a win/lose benchmark.

---

## 2026-02-12 | FC2 completed: FER2013 ten-crop evaluation (protocol-matched block)

Intent:
- Remove the key FER2013 protocol mismatch by adding **ten-crop** evaluation and reporting it as a separate, labeled block (single-crop stays unchanged).

Action:
- Implemented `--tta {singlecrop,tencrop}` in `scripts/eval_student_checkpoint.py`.
	- `tencrop` uses deterministic `Resize -> (optional CLAHE) -> TenCrop -> normalize each crop -> average logits`.
	- Output directory naming encodes protocol to prevent mixing.
- Ran official-split evaluations for CE/KD/DKD on PublicTest/PrivateTest for date tag `20260212`.
- Generated an updated official summary table that includes **both** protocols.

Result (artifacts):
- New eval outputs (per protocol):
	- `outputs/evals/students/fer2013_official__CE_20251223_225031__publictest__test__20260212__singlecrop/`
	- `outputs/evals/students/fer2013_official__CE_20251223_225031__publictest__test__20260212__tencrop/`
	- `outputs/evals/students/fer2013_official__KD_20251229_182119__privatetest__test__20260212__tencrop/`
	- `outputs/evals/students/fer2013_official__DKD_20251229_223722__publictest__test__20260212__tencrop/`
	- `outputs/evals/students/fer2013_official__DKD_20251229_223722__privatetest__test__20260212__singlecrop/`
	- `outputs/evals/students/fer2013_official__DKD_20251229_223722__privatetest__test__20260212__tencrop/`
	- (All runs contain: `reliabilitymetrics.json`, `calibration.json`, `eval_meta.json`)

- Consolidated summary (protocol-aware):
	- `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.md`
	- `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.csv`
	- `outputs/benchmarks/fer2013_official_summary__20260212/fer2013_official_summary.json`

Key observation (from on-disk artifacts):
- Ten-crop improves FER2013 official-split metrics only modestly (no “paper-level jump”).
- Therefore, if there is still a large gap to a paper number, it is likely dominated by **non-protocol factors** (capacity, training recipe, preprocessing/alignment, data regime), not by single-crop vs ten-crop alone.

Suggested next step (to explain the paper gap cleanly, without over-claiming):
- Proceed with FC3 (controlled gap analysis checklist) for the specific paper target:
	- Confirm backbone/capacity differences (MobileNetV3 vs paper backbone).
	- Confirm preprocessing differences (grayscale vs RGB, normalization, alignment/crop policy).
	- Confirm training regime differences (data sources, augmentation strength, class balancing, loss details).
	- Keep the report statement: “Protocol matched (official split + ten-crop) but results still differ; here are the likely causes and constraints.”

FC3 support artifact (paper hyperparameter/protocol extraction → actionable checklist):
- `research/paper_training_recipe_checklist__20260212.md`