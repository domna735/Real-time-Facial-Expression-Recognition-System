# 05 — Commands Checklist (Domain Shift Loop)

Fill in paths as you run; keep commands copy-pasteable.

## 0a) Offline baseline checkpoints + gates (Feb 2026)

These were produced by running `scripts/train_student.py` with `--post-eval`.

Training run dirs:

- KD baseline: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308/`
- KD + LP-loss: `outputs/students/KD/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653/`

Gate outputs (eval-only + ExpW):

- Baseline eval-only: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__classification_manifest_eval_only__test__20260205_163424/`
- Baseline ExpW: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_baseline_20260205_160308__expw_full_manifest__test__20260205_163538/`
- KD+LP eval-only: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__classification_manifest_eval_only__test__20260205_171945/`
- KD+LP ExpW: `outputs/evals/students/mobilenetv3_large_100_img224_seed1337_KD_LP0p01_20260205_163653__expw_full_manifest__test__20260205_172039/`

## 0) Baseline webcam run

```powershell
python demo/realtime_demo.py --model-kind student --device cpu --record-video --record-video-mode annotated
python scripts/score_live_results.py --per-frame demo/outputs/<run_stamp>/per_frame.csv --out demo/outputs/<run_stamp>/score_results.json --pred-source both
```

## 1) Build buffer (manual labels)

```powershell
python scripts/build_webcam_buffer.py --per-frame demo/outputs/<run_stamp>/per_frame.csv --video demo/outputs/<run_stamp>/session_annotated.mp4 --out-dir demo/outputs/<run_stamp>/buffer_manual --min-frame-gap 10 --max-per-class 250 --face-crop
```

## 2) Fine-tune (conservative)

### Head-only

```powershell
python scripts/train_student.py --mode ce --manifest demo/outputs/<run_stamp>/buffer_manual/manifest.csv --data-root . --init-from <base_ckpt> --tune head --epochs 1 --batch-size 64 --num-workers 0 --lr 1e-5 --weight-decay 0.0 --warmup-epochs 0 --output-dir outputs/students/<ft_run>
```

### BN-only

```powershell
python scripts/train_student.py --mode ce --manifest demo/outputs/<run_stamp>/buffer_manual/manifest.csv --data-root . --init-from <base_ckpt> --tune bn --epochs 1 --batch-size 64 --num-workers 0 --lr 1e-5 --weight-decay 0.0 --warmup-epochs 0 --output-dir outputs/students/<ft_run>
```

## 3) Offline eval-only regression gate

```powershell
python scripts/eval_student_checkpoint.py --checkpoint outputs/students/<ft_run>/best.pt --eval-manifest Training_data_cleaned/classification_manifest_eval_only.csv --eval-data-root Training_data_cleaned --eval-split test --batch-size 256 --num-workers 0
```

## 4) Webcam-mini gate (re-score with adapted ckpt)

```powershell
python demo/realtime_demo.py --model-kind student --device cpu --model-path outputs/students/<ft_run>/best.pt --record-video --record-video-mode annotated
python scripts/score_live_results.py --per-frame demo/outputs/<new_run_stamp>/per_frame.csv --out demo/outputs/<new_run_stamp>/score_results.json --pred-source both
```
