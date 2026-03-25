# Process Log - Week 1 of February 2026

This document captures daily activities, decisions, and reflections during the first week of February 2026, focusing on reconstructing the FER system under an evidence-first workflow.

## Week theme

**Goal:** move from paper study → implementation-ready experiments, while keeping the workflow **artifact-grounded** (every claim traceable to JSON outputs / manifests) and **gate-safe** (no deployment-facing adaptation promoted unless it passes eval-only).

**Main decision of the week:** start with **KD first** (stable baseline), then add new research losses (LP-loss) only after a KD baseline is confirmed.

## Key constraints / reminders

- Evidence-first requirement: do not claim numeric improvements unless backed by stored artifacts (`history.json`, `reliabilitymetrics.json`, `outputs/evals/**`).
- Safety gate: any “domain shift improvement” idea must be checked against `Training_data_cleaned/classification_manifest_eval_only.csv`.
- Domain shift target: ExpW and live webcam behavior; evaluation distributions must be controlled.

---

## Daily log

## 2026-02-01 | Week kickoff: evidence-first + gate-safe
Intent:
- Reconfirm the evaluation philosophy and how we will claim results.

Action:
- Consolidated the “easy → research-y” experiment progression plan.
- Reconfirmed that all numeric claims must be traceable to artifacts (`history.json`, `reliabilitymetrics.json`, `outputs/evals/**`).

Result:
- Locked the evaluation roles:
  - eval-only = offline regression gate (`Training_data_cleaned/classification_manifest_eval_only.csv`)
  - ExpW = repeatable in-the-wild proxy (`Training_data_cleaned/expw_full_manifest.csv`)
  - webcam labeled runs = deployment-facing behavior evidence (`demo/outputs/*/score_results.json`)

Decision / Interpretation:
- Start with stable baselines (KD) before adding research losses.

Next:
- Continue paper study and extract one low-risk implementation target.

## 2026-02-02 | Paper study → repo-compatible ablation mapping
Intent:
- Translate paper ideas into minimal, default-off, auditable code changes.

Action:
- Continued the multi-paper study track and drafted implementation mapping for each idea.
- Defined a rule: new objectives must be CLI-flagged and logged into `history.json`.

Result:
- Identified Paper #5 Track A (LP-loss) as the best first “paper → code” step.

Decision / Interpretation:
- Prefer the smallest safe intervention first: add LP-loss as an auxiliary term in student training.

Next:
- Implement LP-loss in `scripts/train_student.py` with default-off behavior.

## 2026-02-03 | Paper #5 deep dive: lock Track A (LP-loss)
Intent:
- Ensure the LP-loss definition is implementable with the current training pipeline.

Action:
- Completed a detailed cross-check pass for the 5-paper set (Paper #5 treated as the primary target).
- Scoped Track A as a supervised auxiliary loss (no new data flows required).

Result:
- Track A selected: implement LP-loss computed on penultimate (or logits) embeddings, with robust handling of batch composition.

Decision / Interpretation:
- LP-loss is low-risk because it is optional and does not change the data pipeline.

Next:
- Plan implementation details: feature extraction point, neighbor rule, and logging.

## 2026-02-04 | Implementation planning: LP-loss + safety logging
Intent:
- Design LP-loss so it is safe, debuggable, and evidence-backed.

Action:
- Specified LP-loss computation requirements:
  - compute on penultimate features when possible
  - degrade gracefully when a class has too few samples in a batch
  - log `included_frac` so we know whether the loss is active

Result:
- Implementation plan ready, but deferred training runs until backup + clean compile.

Decision / Interpretation:
- Do not run experiments until a backup exists (user-requested safeguard).

Next:
- Create backup, implement LP-loss, add post-training evaluation hook.

## 2026-02-05 | Implement LP-loss + run KD baseline vs KD+LP (5-epoch screening)
Intent:
- Implement Paper #5 Track A (LP-loss) in a safe, default-off way.
- Run KD baseline first, then KD+LP, with post-eval generating offline gate artifacts.

Action:
- Created backup snapshot before edits:
  - `backups/before_lp_loss_20260205_144641/`
  - Backed up: `scripts/train_student.py`, `scripts/eval_student_checkpoint.py`
- Implemented LP-loss + logging in `scripts/train_student.py` (default-off; enabled via `--lp-weight` / `--lp-k` / `--lp-embed`).
- Added optional post-training evaluation hook (`--post-eval`) to run eval-only + ExpW and write `post_eval.json`.
- Ran two 5-epoch KD screenings:
  - KD baseline run
  - KD + LP-loss run (`--lp-weight 0.01 --lp-k 20 --lp-embed penultimate`)

Result:
- Code validation:
  - Python syntax check passed for `scripts/train_student.py`.

- KD baseline (training artifacts):
  - Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/`
  - HQ-train val (from run `reliabilitymetrics.json`):
    - Raw: accuracy=0.7297586, macro-F1=0.7281613
    - TS: ECE=0.0373908, NLL=0.7926007, global T=4.4717526
  - Post-eval summary (from `post_eval.json`): eval-only ok=true, ExpW ok=true
  - Offline gate artifacts:
    - eval-only test: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__classification_manifest_eval_only__test__20260205_163424/reliabilitymetrics.json`
      - Raw: accuracy=0.5162321, macro-F1=0.4385411
      - TS: ECE=0.0217606, NLL=1.2961859, global T=3.4225309
    - ExpW test: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__expw_full_manifest__test__20260205_163538/reliabilitymetrics.json`
      - Raw: accuracy=0.6311145, macro-F1=0.4595847
      - TS: ECE=0.0276567, NLL=1.0635237, global T=3.0844002

- KD + LP-loss (training artifacts):
  - Run dir: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/`
  - LP config (from run `history.json`): weight=0.01, k=20, embed=penultimate
    - LP diagnostics at epoch 4: train_lp_loss=4.2430, included_frac=1.0
  - HQ-train val (from run `reliabilitymetrics.json`):
    - Raw: accuracy=0.7296656, macro-F1=0.7276670
    - TS: ECE=0.0252364, NLL=0.7612492, global T=3.4970691
  - Post-eval summary (from `post_eval.json`): eval-only ok=true, ExpW ok=true
  - Offline gate artifacts:
    - eval-only test: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__classification_manifest_eval_only__test__20260205_171945/reliabilitymetrics.json`
      - Raw: accuracy=0.5207738, macro-F1=0.4411229
      - TS: ECE=0.0374865, NLL=1.2773255, global T=3.0327940
    - ExpW test: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__expw_full_manifest__test__20260205_172039/reliabilitymetrics.json`
      - Raw: accuracy=0.6356902, macro-F1=0.4583109
      - TS: ECE=0.0197645, NLL=1.0421315, global T=2.6985071

Decision / Interpretation:
- Interpretation constraint: these are 5-epoch screenings; treat deltas as signals, not final conclusions.
- LP-loss was actually active (included_frac=1.0), so the experiment is valid as a first wiring+effect check.
- In this run pair:
  - ExpW raw macro-F1 does not improve under LP-loss at weight=0.01.
  - Calibration metrics (TS ECE/TS NLL) generally improve, especially on ExpW.

Next:
- If the goal is ExpW macro-F1: try a smaller `--lp-weight` (e.g., 0.001) and/or switch `--lp-embed logits`, then re-run the same post-eval gates.
- If the goal is deployment stability: use KD baseline and KD+LP as candidate base checkpoints for the webcam loop, but only claim wins when re-scoring the same labeled webcam session.

## 2026-02-06 | Real-time demo: CE feels most stable (subjective)
Intent:
- Capture a deployment-facing signal from live webcam use.
- Decide how to convert subjective stability into repeatable, artifact-backed evidence.

Action:
- Ran the real-time webcam demo (`demo/realtime_demo.py`) with multiple student checkpoints.
- Compared these checkpoints in live usage:
  - CE: `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`
  - KD + LP-loss: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt`
  - DKD: `outputs/students/DKD/mobilenetv3_large_100_img224_seed1337_DKD_20260101_221602/best.pt`
- Reviewed why real-time behavior can differ from offline ranking:
  - the demo uses EMA + hysteresis + (optional) voting on probabilities
  - temperature scaling (`logits / T`) changes probability sharpness, affecting flicker even when argmax labels do not change
- Wrote a deployment-facing note in: `research/Real time demo/real time demo report.md`.

Result:
- Subjective finding: **CE** feels **most stable** (less label flicker) and has the best perceived accuracy in live webcam use, compared to KD+LP and DKD.
- Working interpretation: this is consistent with (a) domain shift ($P_{train}(x) \neq P_{webcam}(x)$) and (b) deployment objective mismatch (offline macro-F1 vs live “looks correct + doesn’t flicker”).

Decision / Interpretation:
- Treat this as a valid deployment signal, but not as a final “best model” claim until it is made repeatable.
- Keep CE as the **default demo checkpoint** while running a controlled replay-based comparison.

Next:
- Record one labeled webcam session and replay/score it across checkpoints with fixed demo parameters and a comparable temperature policy, producing `demo/outputs/*/score_results.json`.
- Add a small “replay-based A/B scoring checklist” to the domain shift plan if repeated comparisons become frequent.
