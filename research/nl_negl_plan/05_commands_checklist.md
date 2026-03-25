# Practical checklist + commands (Windows)

## 1) Verify NL scaffold works (no dataset required)
From repo root with `.venv` activated:
- `python scripts/smoke_nl.py --steps 50 --accum 4`

If CUDA is available:
- `python scripts/smoke_nl.py --steps 50 --accum 4 --amp`

Expected: prints `[OK] NL smoke done ...`.

## 2) Baseline student run (reference)
- Use your existing runner:
  - `scripts/run_student_mnv3_ce_kd_dkd.ps1`

Recommended for ablations:
- Do **KD-only first** (so we isolate NegL effects without CE/DKD confounds).
- For a fair comparison, run **baseline KD 5 epochs (NegL off)** and **KD+NegL 5 epochs** with the same batch size/workers/CLAHE/AMP.

## 3) NegL experiment (after wiring)
- Run NegL using the existing student runner (recommended start: KD only):
  - `scripts/run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -UseNegL -NegLWeight 0.05 -NegLRatio 0.5 -NegLGate entropy -NegLEntropyThresh 0.7 -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0`

Baseline KD (no NegL), same schedule:
- `scripts/run_student_mnv3_ce_kd_dkd.ps1 -SkipSmoke -CeEpochs 0 -KdEpochs 5 -DkdEpochs 0`

Notes on “CE/KD/DKD all 5 epochs?”
- It is runnable, but **less valuable as the first step** because it mixes 3 different objectives.
- If KD+NegL looks promising, then run DKD as a follow-up (e.g., DKD +5 extra epochs after KD).

Expected:
- A new run folder under `outputs/students/`.
- `history.json` includes a `negl` block per epoch.
- `reliabilitymetrics.json` + `calibration.json` are generated after eval epochs.

## 4) NL experiment (after wiring)
- Prefer per-sample loss weighting gate as first integration (safer on Windows).

## 5) Combined
- Only after isolated ablations.
