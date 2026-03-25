# 06 — Assumption Check & Next Steps (2026-01-28)

## Assumptions (still true?)

- We can collect small labeled webcam runs for evaluation (webcam-mini).
- Adaptation is offline/periodic (not during live inference).
- We must protect offline eval-only performance (rollback safety gate).

## Current status (from main plan)

See:
- [plan of self-learning + negative learning for domain shift.md](plan%20of%20self-learning%20%2B%20negative%20learning%20for%20domain%20shift.md)

Known artifacts:
- Baseline run: `demo/outputs/20260126_205446/`
- Buffer: `demo/outputs/20260126_205446/buffer_manual/`
- Head-only FT: `outputs/students/FT_webcam_head_20260126_1/` (offline gate failed)
- BN-only FT: `outputs/students/FT_webcam_bn_20260126_1/` (offline gate failed; webcam gate pending)

## Feb 2026 update (offline baselines refreshed)

On 2026-02-05, two short-budget KD runs were executed and evaluated on both offline gates (eval-only and ExpW). These are now convenient **base checkpoints** for the next webcam experiments.

- KD baseline (run dir): `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/`
- KD + LP-loss (run dir): `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/`

Gate outputs are under `outputs/evals/students/*20260205*/` (see `research/domain_shift/evaluation_plan.md` for exact paths and baseline values).

## Immediate next step

1) Record a labeled webcam-mini run using BN-only checkpoint:
   - `outputs/students/FT_webcam_bn_20260126_1/best.pt`
2) Score it using the same protocol.
3) If webcam improves but offline regresses:
   - shrink update (lower LR), and/or
   - tighten buffer sampling (stable-only, higher min-frame-gap, lower per-class cap)

## Next engineering step

Start implementing domain-shift NegL wiring into fine-tune mode (off by default).

Optional (low-risk) pre-step:

- Run the same labeled webcam session twice, re-scored using the two fresh KD checkpoints above (KD baseline vs KD+LP), with identical labels and stabilizer settings.
