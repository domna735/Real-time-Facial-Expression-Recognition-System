# Real-time FER backup pack

This ZIP contains the minimum code + tools + model artifacts needed to run the real-time FER demo and reproduce key evaluation outputs.

## Included
- demo/: real-time UI + face detector logic
- src/: FER utilities
- tools/: diagnostics/helpers
- scripts/: runtime wrapper + eval + ONNX exporter
- Selected student run: outputs\students\CE\mobilenetv3_large_100_img224_seed1337_CE_20251223_225031\\best.pt (+ calibration/metrics when present)
- ONNX export: models\\student_best.onnx

## Quick start (after unzip)
1) Create venv and install deps:
```powershell
python -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install --upgrade pip
.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt
```
2) Run the demo (student):
```powershell
python demo\\realtime_demo.py --model-kind student --model-ckpt outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt
```

## Provenance
- This pack is created from the repo at packaging time; install dependencies from requirements files.

