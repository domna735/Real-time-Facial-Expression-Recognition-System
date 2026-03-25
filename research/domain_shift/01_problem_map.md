# 01 — Problem Map (Domain Shift)

## What we are trying to fix

Observed: the same student checkpoint performs differently on:
- offline datasets (multi-source, curated)
- real-time webcam inference (lighting, camera, pose, compression, background)

We treat this as **domain shift**: $P_{train}(x, y) \neq P_{webcam}(x, y)$.

## Why the offline metric can disagree with live

Common reasons (most are already seen in this repo):
- Webcam frames are correlated + narrow style (single subject/session)
- Real-time pipeline introduces different crops/scale/blur
- Live scoring uses smoothing (EMA/vote/hysteresis), so stability changes matter a lot
- Transition/ambiguous frames (label noise) affect training and evaluation

## Failure modes to avoid

- Catastrophic drift: improving webcam but regressing offline eval-only
- Overfitting to one session: improved “style match” but worse generalization
- Stability regression: more flip-rate/jitter even if framewise accuracy is similar
- Confidence miscalibration: thresholds stop making sense after adaptation

## Why Self-Learning + NegL is relevant

- Self-learning uses **high-confidence pseudo-labels** to adapt features to webcam appearance.
- NegL (complementary learning) uses **medium-confidence frames** as “NOT class k” signals to avoid reinforcing wrong pseudo-labels.

Key idea: only update when it is safe, and only accept checkpoints that pass gates.

## Evidence sources in this repo

- Live run artifacts: `demo/outputs/<stamp>/per_frame.csv`, `score_results.json`, optional `session_*.mp4`
- Buffer builder: `scripts/build_webcam_buffer.py`
- Training entry: `scripts/train_student.py` with tune modes (`head`, `bn`, ...)
- Offline gate: `scripts/eval_student_checkpoint.py` on `Training_data_cleaned/classification_manifest_eval_only.csv`
