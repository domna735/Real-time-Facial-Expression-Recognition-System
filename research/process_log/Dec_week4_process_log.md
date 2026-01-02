# Process Log - Week 4 of December
This document captures the daily activities, decisions, and reflections during the fourth week of December 2025, focusing on reconstructing the facial expression recognition system as per the established plan.

## 2025-12-20 | Baseline 2-teacher ensembles + triage start
Intent:
- Re-establish a reproducible teacher-ensemble evaluation baseline (RN18 + B3) on cleaned splits.
- Identify a sensible default ensemble-space (logit vs prob) and CLAHE setting.

Action:
- Ran multiple RN18/B3 weightings on RAFDB test and full unified test using logit-space fusion + CLAHE.
- Ensured output runs emit `ensemble_metrics.json` plus alignment artifacts.

Result:
- RAFDB test strongly prefers balanced RN18/B3 (0.5/0.5) among the RN18/B3-only options.
- Full unified test (“fulltest”) prefers B3-heavier (0.3/0.7) among RN18/B3-only options.

Decision / Interpretation:
- Keep `--ensemble-space logit` as the default for KD/DKD semantics.
- Keep `--use-clahe` enabled to match the legacy/expected preprocessing effect.

Next:
- Expand to include ConvNeXt (CNXT) and evaluate 3-teacher ensembles.
- Add a single consolidated report + archive workflow for low-value runs.


## 2025-12-21 | Windows multiprocessing stability for ensemble export
Intent:
- Make ensemble exporting stable on Windows (DataLoader spawn + worker pickling).

Action:
- Removed dynamic “import-from-filepath” patterns that were causing worker pickling failures.
- Made `scripts/` importable as a package so transforms/classes resolve as real modules.

Result:
- Exporters run with `--num-workers > 0` without crashing due to pickling/import errors.

Decision / Interpretation:
- Treat “importability” as a hard requirement for any code used inside DataLoader workers on Windows.

Next:
- Implement multi-teacher exporter to support 3-model ensembles.


## 2025-12-22 | Add 3-teacher ensemble export + consolidate weak runs
Intent:
- Support 3-teacher ensembles (RN18 + B3 + CNXT) for both evaluation and softlabel export.
- Produce a single “bad ensembles” list file and archive those runs.

Action:
- Implemented multi-teacher exporter supporting repeated `--teacher` + `--weight`.
- Generated a triage report scanning `outputs/softlabels/**/ensemble_metrics.json`.
- Generated a consolidated bad-list file and archived listed runs into one folder.

Result:
- Triage artifacts created:
	- `outputs/softlabels/_ensemble_triage.md`
	- `outputs/softlabels/_ensemble_bad_list.txt`
- Archived low-value/duplicate runs:
	- `outputs/softlabels/_archive/bad_list_20251223_121501`

Decision / Interpretation:
- Keep the repo’s “ensemble state” clean: keep best-per-group + actively referenced runs; archive the rest.

Next:
- Build a larger test manifest to reduce variance in comparisons.


## 2025-12-23 | Bigger test benchmark + best ensemble selection + start HQ-train softlabels
Intent:
- Reduce uncertainty from RAFDB-only testing by evaluating on a larger, mixed-source test.
- Pick the winning teacher ensemble to generate training softlabels for student KD/DKD.

Action:
- Built merged test manifest `Training_data_cleaned/test_all_sources.csv` (48,928 rows) from multiple `test_*.csv` sources.
- Evaluated ensembles on this bigger test; compared RN18+B3, RN18+CNXT, and RN18+B3+CNXT.
- Started exporting HQ-train softlabels using the winning 3-teacher ensemble.

Result:
- Best on `test_all_sources`:
	- RN18/B3/CNXT weights 0.4/0.4/0.2, logit+CLAHE: macro-F1=0.659608, acc=0.687255
	- RN18/B3 0.3/0.7: macro-F1=0.654132, acc=0.682186
	- RN18/CNXT 0.5/0.5: macro-F1=0.649794, acc=0.677383
- Started HQ-train export (split=train, 213,144 rows; total=259,004) to:
	- `outputs/softlabels/_ens_hq_train_rn18_0p4_b3_0p4_cnxt_0p2_logit_clahe_20251223_152856`

Decision / Interpretation:
- Stage B teachers are not required for student KD/DKD in this repo (student consumes `softlabels.npz`, not teacher ONNX).
- Lower macro-F1 on the bigger mixed-source test is expected; proceed with the best overall ensemble and validate on deployment-domain samples.

Next:
- Wait for HQ-train export completion and then start student training (KD + DKD) using the exported `softlabels.npz`.


## 2025-12-23 | Student training kickoff (MobileNetV3) + Windows reliability fixes
Intent:
- Start the first student experiment (MobileNetV3-Large) with a reproducible CE → KD → DKD workflow.
- Remove Windows-specific failure modes (PowerShell arg handling, dataset path resolution, DataLoader shared-memory crash).

Action:
- Created a single “one kick” PowerShell runner for student training:
	- `scripts/run_student_mnv3_ce_kd_dkd.ps1`
	- Supports smoke/full modes and writes logs under `outputs/students/_logs_<timestamp>/`.
- Updated student trainer so CE does **not** require softlabels:
	- `scripts/train_student.py` now only loads `softlabels.npz` + `softlabels_index.jsonl` when `--mode kd|dkd`.
- Fixed PowerShell REPL launch bug:
	- Renamed the runner’s parameter from `-Args` (PowerShell automatic variable collision) to `-ArgList`.
- Fixed dataset path resolution:
	- Ensured `--data-root Training_data_cleaned` is used so manifest-relative `image_path` entries resolve and are not dropped.
- Fixed student model name mismatch:
	- Updated model id to timm’s `mobilenetv3_large_100`.
- Mitigated Windows DataLoader crash `error code: 1455` (“Couldn't open shared file mapping”):
	- Reduced DataLoader `prefetch_factor` on Windows when `num_workers > 0`.
	- Reduced default workers in the runner (safer default on Windows).

Result:
- Short CE sanity run completed successfully (2 epochs) with stable settings (`BatchSize=128`, `NumWorkers=2`).
- Full CE→KD→DKD pipeline was launched with stable settings:
	- Command used: `-BatchSize 128 -NumWorkers 2 -CeEpochs 10 -KdEpochs 20 -DkdEpochs 10`.
	- Output root: `outputs/students/` (run folders include timestamp; logs under `_logs_20251223_225031/`).

Decision / Interpretation:
- For Windows training stability, prefer smaller `num_workers` + moderate batch size; scale up only after confirming no 1455 errors.
- Keep student experiments aligned to the same manifest used for HQ-train softlabels (`classification_manifest_hq_train.csv`) to avoid confounds.

Next:
- Monitor the long run and record CE vs KD vs DKD metrics (macro-F1/acc + calibration) from:
	- `history.json`
	- `reliabilitymetrics.json`
	- `calibration.json`
- If KD/DKD gains are unclear, tune (`temperature`, `alpha`, `beta`) before changing teachers or datasets.


## 2025-12-24 | DKD resume fix + record CE/KD/DKD metrics (HQ-train)
Intent:
- Diagnose why DKD produced an empty output folder and finish the DKD stage cleanly.
- Record the student run metrics for CE vs KD vs DKD using the same HQ-train manifest and preprocessing.

Action:
- Investigated DKD log output and found DKD was resuming from KD at `start_epoch=16` while DKD was launched with `--epochs 10`.
	- This resulted in a zero-length epoch loop and an “instant finish” with no artifacts.
- Fixed DKD resume semantics in the runner:
	- Updated `scripts/run_student_mnv3_ce_kd_dkd.ps1` so DKD treats `-DkdEpochs` as “extra epochs after resume”.
	- The runner now reads the KD checkpoint epoch and sets DKD `--epochs = (resume_epoch + 1 + DkdEpochs)`.
- Re-ran DKD-only (resume from KD `best.pt`) until completion.

Result:
- DKD now runs and produces artifacts (`best.pt`, `history.json`, `reliabilitymetrics.json`, `calibration.json`) under:
	- `outputs/students/mobilenetv3_large_100_img224_seed1337_DKD_20251223_225031/`
- Metrics snapshot (HQ-train manifest, img224, CLAHE+AMP, seed=1337):
	- CE: acc=0.750174, macro-F1=0.741952 | raw NLL=1.315335, raw ECE=0.131019 | TS NLL=0.777757, TS ECE=0.049897 (T=3.228)
	- KD: acc=0.734688, macro-F1=0.733351 | raw NLL=2.093148, raw ECE=0.215289 | TS NLL=0.768196, TS ECE=0.027764 (T=5.000)
	- DKD: acc=0.737432, macro-F1=0.737511 | raw NLL=1.511788, raw ECE=0.209450 | TS NLL=0.765203, TS ECE=0.026605 (T=3.348)

Decision / Interpretation:
- The DKD failure was a run-configuration bug (resume epoch > total epochs), not a training/code crash.
- With current hyperparameters, CE slightly outperforms KD/DKD on acc/macro-F1, while KD/DKD show much better calibrated ECE after temperature scaling.

Next:
- If the goal is accuracy/macro-F1 gains from distillation, tune KD/DKD hyperparameters (`temperature`, `alpha`, `beta`) and/or train longer.
- Keep the corrected DKD resume logic for future CE→KD→DKD runs.


## 2025-12-24 | Mini-report pack (10 reports) + data verification
Intent:
- Produce 10 short, deliverable mini-reports for the project restart.
- Verify that all reported numbers match on-disk artifacts (no fabricated metrics).

Action:
- Created/updated report pack folder:
	- `research/report 24-12-2025/` (10 markdown reports)
- Cross-checked report numbers against:
	- Teacher metrics: `outputs/teachers/*/reliabilitymetrics.json`
	- Student metrics: `outputs/students/*/reliabilitymetrics.json`
	- Ensemble metrics: `outputs/softlabels/**/ensemble_metrics.json`
	- Dataset validation: `outputs/manifest_validation*.json`
	- Manifest row counts: `Training_data_cleaned/classification_manifest_hq_train.csv` and `Training_data_cleaned/test_all_sources.csv`

Result:
- Verified (matches artifacts):
	- Stage A teacher metrics (RN18/B3/CNXT) in Report 01.
	- Student CE/KD/DKD metrics in Report 03.
	- Ensemble benchmark numbers (test_all_sources) in Report 02.
	- Dataset validation totals (466,284 rows; 0 missing paths) in Report 04.
- Corrected a previously misstated HQ-train train split size:
	- Train split is 213,144 (total 259,004), not 209,661.

Decision / Interpretation:
- Keep the report pack grounded to JSON artifacts; demo runtime KPIs (FPS/latency/flip-rate) remain “measurable/TBD” until a dedicated demo run is executed.

Next:
- Optional: add an index/TOC markdown in `research/report 24-12-2025/` linking all 10 reports.
- Optional: run the realtime demo and summarize CSV logs into FPS/latency/flip-rate for Report 06/09.