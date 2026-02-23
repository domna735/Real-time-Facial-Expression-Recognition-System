# Models (Git LFS)

This folder is intended for **shareable runtime artifacts** that are safe to include in a GitHub repo.

## What goes here

- `student_best.pt`: best student PyTorch checkpoint
- `student_best.onnx`: ONNX export for deployment
- `calibration.json`, `reliabilitymetrics.json`: evaluation/calibration metadata (optional but recommended)
- `student_best_onnx_export_meta.json`: metadata produced by `scripts/export_student_onnx.py`

## What does NOT go here

- Training datasets (e.g. `Training_data/`, `Training_data_cleaned/`)
- Large experiment folders under `outputs/`

## Notes

- This repo uses Git LFS for `*.pt` and `*.onnx` (see `.gitattributes`).
- If you publish this repo, make sure you have the right to redistribute these trained weights.
