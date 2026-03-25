# 03 — Implementation Steps

This file tracks engineering tasks specifically needed to run Self-Learning + NegL for domain shift.

## Already implemented (current repo)

- Live logging + scoring: `demo/realtime_demo.py`, `scripts/score_live_results.py`
- Buffer builder: `scripts/build_webcam_buffer.py`
- Conservative fine-tune modes: `scripts/train_student.py` (`--init-from`, `--tune head|bn|lastblock_head`)
- Offline eval-only: `scripts/eval_student_checkpoint.py`

## Feb 2026 note: base checkpoint candidates

Before adding any NegL wiring, keep the “base checkpoint” explicit in each experiment run folder and report.

As of 2026-02-05, two refreshed KD base checkpoints exist:

- KD baseline: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/best.pt`
- KD + LP-loss: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/best.pt`

## Next implementation (domain-shift NegL)

### A) Decide training data streams

We want two subsets from webcam frames:
- High-confidence frames → positive pseudo-label CE
- Medium-confidence frames → NegL only

### B) Wire NegL into student training (targeted)

- Add flags to `scripts/train_student.py` to enable NegL during fine-tune:
  - `--negl-enable`
  - `--negl-weight`, `--negl-entropy-thresh` (or `--negl-pmax-band`)
  - negative class selection policy (start: lowest-prob classes)

Implementation should call existing loss in `src/fer/negl/losses.py`.

### C) Extend buffer builder to emit confidence bands (optional)

Option 1 (fast): compute bands on-the-fly in training using model logits.

Option 2 (more reproducible): in `scripts/build_webcam_buffer.py`, also write columns:
- `pmax`, `entropy`, `stable`
Then training can deterministically select samples.

### D) Logging

Per epoch, log:
- NegL applied fraction
- average entropy / pmax of NegL-selected samples
- offline eval-only summary (if running inside training loop)

### E) Safety

- Keep NegL disabled by default.
- Enforce gates outside training before promoting a checkpoint.
