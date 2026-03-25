# Process Log - Week 5 of January 2026

This document captures the daily activities, decisions, and reflections during the fifth week of January 2026, focusing on reconstructing the facial expression recognition system as per the established plan.

---

Follow the template below to document your activities, decisions, and reflections for each day of the week.

## YYYY-MM-DD | Short Title

Intent:
Action:
Result:
Decision / Interpretation:
Next:

---

## 2026-01-26 | Webcam-mini labeled baseline + buffer builder

Intent:

- Create a first **target-domain (webcam)** labeled run that we can score fairly (raw vs smoothed) and reuse for training/adaptation.

Action:

- Ran the real-time demo on CPU with manual labeling + video recording.
- Generated live scoring metrics using `scripts/score_live_results.py`.
- Implemented and ran a buffer-builder script to extract labeled training images + manifest from the recorded run.

Result:

- Output folder: `demo/outputs/20260126_205446/`
  - `per_frame.csv`, `events.csv`, `thresholds.json`, `session_annotated.mp4`
  - `score_results.json`
  - Buffer output: `demo/outputs/20260126_205446/buffer_manual/` containing `images/` (426 crops) + `manifest.csv`
- Baseline metrics (protocol-scored):
  - `raw.macro_f1_present` = 0.4721, `raw.accuracy` = 0.5284
  - `smoothed.macro_f1_present` = 0.5248, `smoothed.accuracy` = 0.5879
- Key failure modes in this session:
  - `Fear` F1 = 0.0
  - `Sad` F1 ≈ 0.03

Decision / Interpretation:

- The end-to-end “webcam-mini → score → reuse for training” pipeline is now working.
- Stabilization (smoothed) improves live metrics over raw, but there is still a large robustness gap on harder classes (Fear/Sad).

Next:

- Use the generated buffer manifest (`buffer_manual/manifest.csv`) for a **conservative fine-tune** (head-only / BN-only) and then re-score a second labeled run for acceptance.
- Run offline regression check on `Training_data_cleaned/classification_manifest_eval_only.csv`.

---

## 2026-01-26 | Head-only adaptation run + 2nd webcam-mini check

Intent:

- Start Step 3 (self-learning fine-tune MVP) using the extracted webcam buffer.
- Check acceptance gates: new webcam-mini scoring + offline regression vs baseline student.

Action:

- Ran a conservative head-only fine-tune from the baseline student checkpoint using:
  - Buffer manifest: `demo/outputs/20260126_205446/buffer_manual/manifest.csv`
  - Init checkpoint: `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`
  - Output: `outputs/students/FT_webcam_head_20260126_1/`
- Evaluated offline regression on `Training_data_cleaned/classification_manifest_eval_only.csv` for:
  - Baseline checkpoint
  - Adapted checkpoint
- Recorded a second labeled webcam run and scored it with the same protocol.

Result:

- Adapted checkpoint run:
  - `outputs/students/FT_webcam_head_20260126_1/best.pt`
- Offline eval-only regression check:
  - Baseline (CE20251223): accuracy=0.5674, macro-F1=0.4859
  - Adapted (FT_webcam_head_20260126_1): accuracy=0.5480, macro-F1=0.4508
  - TS calibration improved (ECE ~0.06 both), but core macro-F1 dropped.
- New labeled run folder: `demo/outputs/20260126_215903/`
  - Scoring: `demo/outputs/20260126_215903/score_results.json`
  - Live (protocol-scored):
    - raw: macro-F1=0.4933, acc=0.4638
    - smoothed: macro-F1=0.5552, acc=0.5139
  - Smoothed per-class F1 highlights: Fear=0.3432, Sad=0.4372

Decision / Interpretation:

- Webcam-mini macro-F1 (smoothed) improved vs baseline run (0.5248 → 0.5552), but offline eval-only macro-F1 regressed (0.4859 → 0.4508).

Why webcam-mini improved (likely reasons):

- **Target-domain alignment**: The webcam buffer contains the same capture pipeline as deployment (camera sensor, lighting, background, face size/pose), so even a small update can reduce domain shift.
- **Class-specific weak points improved**: In the second run, previously weak classes (`Fear`, `Sad`) are no longer near-zero, suggesting the model is learning cues that are more consistent with our webcam appearance.
- **Smoothing interaction**: The deployment metric is smoothed; if the adapted model produces slightly more stable logits around the correct class, the smoothing pipeline can amplify the gain.

Why offline eval regressed (likely reasons):

- **Overfitting / catastrophic drift risk**: The webcam buffer is small and correlated (same subject/session), so updating weights (even head-only) can shift decision boundaries away from the multi-source distribution.
- **Label noise + temporal correlation**: Manual labels are correct at the event level, but per-frame assignment and face-crop noise can add mislabeled/blur frames; this can hurt generalization.
- **Distribution mismatch**: The buffer may over-represent certain expressions/poses/lighting. Even with per-class caps, the “style” is narrow compared to `Training_data_cleaned`.
- **Run-to-run variance**: The second webcam run is not identical to the baseline run (expression mix / duration). Improvement is encouraging but should be confirmed with repeat runs.

Decision:

- Do not “promote” this adapted checkpoint yet; keep it as an experiment output until we adjust the adaptation recipe to avoid offline regression.
- Treat this as evidence the direction (target-domain adaptation) is promising, but we need a safer update policy.

Next:

- Next experiment: try BN-only adaptation (`--tune bn`) with smaller LR/epochs, and re-check both acceptance gates.
  - Rationale: BN parameters/statistics are closely tied to image appearance (brightness/contrast/color distribution). Updating BN can help adapt to webcam lighting without moving class decision boundaries as aggressively as head weight updates.
  - Conservative recipe: `--epochs 1` (or 2), `--lr 1e-5` (or 2e-5), keep `--weight-decay 0.0`, and keep batch size moderate.
- If BN-only still regresses offline, try making the update even smaller (1 epoch, lower LR) and/or reducing buffer size per class.
- Optionally tighten buffer sampling to reduce noise (increase `--min-frame-gap`, enable `--stable-only` if needed, keep `--face-crop`).

---

## 2026-01-26 | BN-only adaptation run + offline regression gate

Intent:

- Try a safer adaptation variant (BN-only) that targets webcam appearance shift while minimizing changes to class decision boundaries.

Action:

- Ran BN-only fine-tune from the baseline student checkpoint using:
  - Buffer manifest: `demo/outputs/20260126_205446/buffer_manual/manifest.csv`
  - Init checkpoint: `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`
  - Tune policy: `--tune bn` (BN-only)
  - Conservative hyperparams: `--epochs 1`, `--lr 1e-5`
  - Output: `outputs/students/FT_webcam_bn_20260126_1/`
- Ran offline eval-only regression check for the BN-only checkpoint.

Result:

- BN-only adapted checkpoint:
  - `outputs/students/FT_webcam_bn_20260126_1/best.pt`
- Offline eval-only regression metrics:
  - `outputs/evals/students/FT_webcam_bn_20260126_1_eval_only_test/`
  - accuracy=0.5486, macro-F1=0.4513
  - TS ECE=0.0606

Decision / Interpretation:

- BN-only at this setting did **not** fix the offline regression gate (macro-F1 still ~0.451, similar to head-only).
- This suggests the regression is not only from “head boundary drift”; it may be driven by **buffer correlation/overfit**, **frame-level noise**, or **insufficiently diverse target data**.
- Still need a webcam-mini (labeled) run using this BN-only checkpoint to see whether it improves target-domain metrics more cleanly than head-only.

Next:

- Record a new labeled webcam-mini run using `outputs/students/FT_webcam_bn_20260126_1/best.pt`, then score it with `scripts/score_live_results.py`.
- If webcam improves but offline still regresses, reduce update size further (LR 5e-6, keep epochs=1) and/or rebuild a smaller/cleaner buffer (larger `--min-frame-gap`, `--stable-only`).

---

## 2026-01-28 | Final report evidence audit + automated checker

Intent:

- Perform a full correctness pass on the final report with the rule: **all numeric claims must be artifact-backed**.
- Remove any remaining placeholder artifact references and make the report portable (repo-relative paths).

Action:

- Patched `research/final report/final report.md` to replace placeholder paths like `<run>`, `<stage>`, and `...` with either:
  - concrete artifact paths (when a specific artifact is being cited), or
  - safe repo-relative globs (e.g., `demo/outputs/*/score_results.json`) when describing a class of artifacts.
- Implemented an automated audit script: `scripts/audit_final_report.py`.
- Iterated until the audit was robust to common report formatting patterns:
  - rounding-aware numeric comparisons (e.g., 4 d.p. in per-class tables)
  - compare-table verification by mapping row tokens (e.g., `KD_YYYYMMDD_HHMMSS`) and checking overlapping numeric columns against the referenced `_compare*.md` sources.
- Ran the audit multiple times to catch and fix issues:
  - initial failures from placeholder paths and over-strict compare-table header matching
  - final run passes with `FAIL: 0`.

Result:

- The final report now has artifact references that resolve to real files/directories in this repo.
- `scripts/audit_final_report.py` reports `FAIL: 0` for the audited sections (dataset counts, teacher/student tables, ensemble table, webcam scoring, offline gate, ExpW compare, and NL/NegL compare tables).

Decision / Interpretation:

- We now have a repeatable “evidence gate” for the report: if any numbers drift or artifacts move, the audit will catch mismatches.
- Placeholder artifact citations were a correctness risk; replacing them improves report reliability and portability.

Next:

- Optional: extend the audit to include **reference section structure checks** (sequential numbering, no duplicates, and every reference resolves to an existing repo path).
- Continue domain-shift work (collect BN-only webcam-mini run and re-score) while keeping the report evidence-gated.