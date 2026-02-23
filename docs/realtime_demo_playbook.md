# Real-time Demo Playbook (Best Student + Manual Labels)

This playbook explains how to run the real-time demo with the **best student checkpoint** (MobileNetV3) and how to do **real-time manual labeling** during a session.

## 1) Pick the model checkpoint

### Option A (recommended): auto-pick the best student

The demo can auto-select a student checkpoint by scanning `outputs/students/**/reliabilitymetrics.json` and choosing the run with the **highest `raw.macro_f1`** (tie-break: `raw.accuracy`).

You don’t need to provide a path.

### Option B: provide a specific checkpoint path

Example (your current best CE student run):

- `outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt`

## 2) Run the demo

You can run either:

- Directly via the demo script: `demo/realtime_demo.py`
- Or via the wrapper: `scripts/realtime_infer_arcface.py`

### 2.1 Run student (auto-pick best)

```powershell
python demo/realtime_demo.py --model-kind student
```

Or with the wrapper:

```powershell
python scripts/realtime_infer_arcface.py --model-kind student
```

### 2.2 Run student (explicit checkpoint)

```powershell
python demo/realtime_demo.py --model-kind student --model-ckpt outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt
```

### 2.3 Temperature scaling (student)

By default, student inference will use the run’s `calibration.json` (if present next to the checkpoint) and apply:

- `probs = softmax(logits / T)`

You can override it:

```powershell
python demo/realtime_demo.py --model-kind student --temperature 1.0
```

Or point to a specific JSON containing `global_temperature`:

```powershell
python demo/realtime_demo.py --model-kind student --temperature-json outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/calibration.json
```

### 2.4 Camera / source / outputs

The demo already supports webcam/video inputs and an output directory (see `--help`). Typical examples:

- Webcam (default):

```powershell
python demo/realtime_demo.py --model-kind student
```

- Select camera index:

```powershell
python demo/realtime_demo.py --model-kind student --camera-index 1
```

- Save outputs to a custom folder:

```powershell
python demo/realtime_demo.py --model-kind student --output-dir demo/outputs
```

## 3) Real-time labeling workflow

The demo supports **manual labeling** while it runs.

### 3.1 Keyboard labeling (fast)

- Press `1..7` to set the **current ground-truth label** (canonical 7-class index).
- Press `c` to clear the current label.
- Press `q` to quit.
- Press `o` to toggle the probability overlay.

### 3.2 Mouse labeling

You can also click the bottom label bar to assign a label.

### 3.3 Smoothing controls (optional)

These are helpful to stabilize the displayed prediction:

- `[` / `]` adjust EMA alpha
- `-` / `=` adjust hysteresis delta
- `v` / `b` adjust voting window
- `n` / `m` adjust vote min count

If you’re doing **evaluation-like labeling**, consider reducing smoothing so the UI reflects the raw model more closely.

## 4) Outputs produced

The demo writes session artifacts in the chosen output directory (typically timestamped subfolder). Common files:

- `per_frame.csv` — per-frame predictions + probabilities + label (if set)
- `events.csv` — event-level changes (depends on smoothing/event logic)
- `demoresultssummary.csv` — summary metrics for the session
- `per_class_correctness.csv` — per-class correctness summary
- `thresholds.json` — smoothing parameters used

## 5) Troubleshooting

- **`timm is required` error**: install dependencies from `requirements.txt` (student uses timm).
- **Very low FPS**: try `--device cpu` vs GPU, or switch detector (YuNet is usually fastest).
- **No faces detected**: check camera exposure/lighting, then try a different `--detector`.
- **Checkpoint not found**: verify the `.pt` path exists and matches the run folder.
