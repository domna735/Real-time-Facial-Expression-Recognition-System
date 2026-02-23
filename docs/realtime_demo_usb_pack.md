# Real-time Demo (USB ZIP Pack)

This guide creates a **minimal ZIP** you can copy to your USB (`D:\`) and run the real-time demo using the **best student checkpoint** (`best.pt`).

It packages only what the demo needs:

- `demo/` (real-time UI + face detector logic)
- `src/` (FER utilities)
- `tools/` (diagnostics/helpers used by scripts)
- `scripts/train_teacher.py` (needed for transforms)
- the **best student run’s** `best.pt` (+ `calibration.json` if present)

## 1) Build the ZIP (on your dev machine)

From repo root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/make_realtime_demo_zip.ps1
```

Output:

- `outputs/realtime_demo_usb.zip`

The script auto-picks the best student by scanning:

- `outputs/students/**/reliabilitymetrics.json`

Selection rule:

- highest `raw.macro_f1` (tie-break: `raw.accuracy`)

## 2) Copy ZIP to USB (D:\)

```powershell
Copy-Item -LiteralPath outputs/realtime_demo_usb.zip -Destination D:\ -Force
```

## 3) Unzip on USB (D:\)

This will create a folder like `D:\Real-time-Facial-Expression-Recognition-System_v2_restart`.

```powershell
Expand-Archive -LiteralPath D:\realtime_demo_usb.zip -DestinationPath D:\ -Force
Set-Location D:\Real-time-Facial-Expression-Recognition-System_v2_restart
```

## 4) Python environment (on the target machine)

You need Python 3.10+.

Create a venv and install basic deps:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Install **student model** deps:

```powershell
.\.venv\Scripts\python.exe -m pip install timm
```

Install **PyTorch** (pick ONE option):

- CUDA GPU (NVIDIA): install from the PyTorch site for your CUDA version.
- DirectML (often works on many Windows GPUs):

```powershell
.\.venv\Scripts\python.exe -m pip install torch-directml
```

## 5) Run the demo (explicit best.pt)

```powershell
.\.venv\Scripts\python.exe demo/realtime_demo.py --model-kind student --model-ckpt outputs/students/CE/mobilenetv3_large_100_img224_seed1337_CE_20251223_225031/best.pt
```

Or auto-pick the best student (no path needed):

```powershell
.\.venv\Scripts\python.exe demo/realtime_demo.py --model-kind student
```

## Notes / common issues

- First run may download face detector models into `demo/models` (needs internet). If you are offline, try:

```powershell
.\.venv\Scripts\python.exe demo/realtime_demo.py --model-kind student --detector haar
```

- If you see **`timm is required`**, install `timm`.
- If you see **very low FPS**, try `--detector yunet` (fastest when available) or reduce camera resolution (driver settings).
