# GPU server how-to (PolyU intranet)

This guide is written for:
- Local machine: Windows + PowerShell
- Server: Linux (home folder `/home/donovan`)
- Goal: copy this repo + needed data to server, then run 4 trainings in parallel on 4 GPUs, safely surviving SSH disconnects.

## 0) Prereqs

- You must be on PolyU intranet (classroom / PolyU Wi-Fi) OR connected via PolyU Research VPN.
- Server login:
  - Host: `10.21.20.153`
  - Port: `22`
  - User: `donovan`

## 1) Connect to the server (Windows PowerShell)

Open PowerShell in your repo root and run:

- Test SSH:
  - `ssh -p 22 donovan@10.21.20.153`

Notes:
- Do NOT wrap the command in backticks when you type it in PowerShell. In PowerShell, the backtick character is a line-continuation/escape character and can corrupt the command.
- Do NOT append `yes` as an argument. If SSH asks “Are you sure you want to continue connecting (yes/no/[fingerprint])?”, then you type `yes` as an interactive response.
- You do NOT put the password in the command. SSH will prompt you for the password after it successfully reaches the server.

If you see `Connection timed out`, first confirm you can reach the server on port 22:
- `Test-NetConnection 10.21.20.153 -Port 22`
If it fails, you are not reaching the server network (Wi‑Fi/VPN issue) or the server is not reachable.

### VS Code Remote-SSH (recommended)

1. Install the VS Code extension: **Remote - SSH**.
2. Add an SSH config entry in `C:\Users\<you>\.ssh\config`:

```
Host polyu-gpu
  HostName 10.21.20.153
  User donovan
  Port 22
```

3. In VS Code: Command Palette → “Remote-SSH: Connect to Host…” → `polyu-gpu`.

## 2) Create folders on the server

After you SSH in:

- `mkdir -p ~/projects ~/data`

Suggested layout:
- Code repo: `~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart`
- Datasets/manifests: live inside the repo (`Training_data_cleaned/`) per current pipeline

## 3) Copy files from your local PC to the server

### Option A (recommended): copy only what is needed for training

This is much faster than copying raw datasets.

From **local PowerShell** (repo root):

1) Copy the code + small files:
- `scp -P 22 -r .\scripts .\src .\tools .\demo .\requirements.txt .\requirements-directml.txt donovan@10.21.20.153:~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart/`

2) Copy the cleaned training data (this can still be large):
- `scp -P 22 -r .\Training_data_cleaned donovan@10.21.20.153:~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart/`

3) Copy your *RN18 Stage B output dir* so resume works (epoch 39 checkpoint):
- `scp -P 22 -r .\outputs\teachers\RN18_resnet18_seed1337_stageB_img384 donovan@10.21.20.153:~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart/outputs/teachers/`

If you also want to sweep B3 Stage A checkpoints later, copy that Stage A folder too:
- `scp -P 22 -r .\outputs\teachers\B3_tf_efficientnet_b3_seed1337_stageA_img224 donovan@10.21.20.153:~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart/outputs/teachers/`

### Option B (not recommended): copy *everything*

This may take extremely long (raw data in `Training_data/`).

- `scp -P 22 -r . donovan@10.21.20.153:~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart/`

## 4) Create a conda env on the server

SSH into the server, then:

1) Install Miniconda (if needed)
- Follow: https://docs.anaconda.com/miniconda/

2) Create env (example):
- `conda create -n fer python=3.11 -y`
- `conda activate fer`

3) Install PyTorch for the server CUDA
- Run `nvidia-smi` and note the **CUDA Version** reported.
- Install a matching PyTorch build (recommended: conda):
  - Example (CUDA 12.1):
    - `conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia`

4) Install repo requirements:
- `cd ~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart`
- `python -m pip install -r requirements.txt`

## 5) Verify GPUs

- `nvidia-smi -L`

You should see 4 GPUs. If not, tell me the output.

## 6) Run 4 trainings in parallel (tmux)

Key idea:
- Use `tmux` so runs keep going after you disconnect.
- Pin each job to a GPU via `CUDA_VISIBLE_DEVICES`.

### Start tmux session

- `tmux new -s teachers`

Inside tmux, split into 4 panes:
- Press `Ctrl+b` then `%` (vertical split)
- Press `Ctrl+b` then `"` (horizontal split)
Repeat until you have 4 panes.

In each pane:
- `cd ~/projects/Real-time-Facial-Expression-Recognition-System_v2_restart`
- `conda activate fer`

### Pane 1 (GPU0): B3 Stage A & B retrain

- `CUDA_DEVICE=0 MANIFEST_PRESET=full EVAL_EVERY=10 CLEAN=1 bash scripts/linux/run_teachers_2stage_b3.sh`

### Pane 2 (GPU1): ConvNeXt Stage A & B

- `CUDA_DEVICE=1 MANIFEST_PRESET=full EVAL_EVERY=10 CLEAN=1 bash scripts/linux/run_teachers_2stage_convnext.sh`

### Pane 3 (GPU2): ViT Stage A & B

- `CUDA_DEVICE=2 MANIFEST_PRESET=full EVAL_EVERY=10 CLEAN=1 bash scripts/linux/run_teachers_2stage_vit.sh`

### Pane 4 (GPU3): RN18 Stage B resume (from epoch 39)

This requires you copied the output dir containing `checkpoint_last.pt`.

- `CUDA_DEVICE=3 MANIFEST_PRESET=hq EVAL_EVERY=10 bash scripts/linux/resume_rn18_stageB.sh`

### Detach and leave it running

- Press `Ctrl+b` then `d`

You can now disconnect SSH (and even turn off your local PC). The jobs keep running on the server.

### Re-attach later

- `tmux attach -t teachers`

## 7) Quick monitoring

- GPU usage:
  - `watch -n 1 nvidia-smi`
- Tail a log:
  - `tail -f outputs/teachers/_overnight_logs_2stage/*/*.log`

## 8) Common issues

- **Permission denied / can’t connect**: you are not on intranet/VPN.
- **`scp` very slow**: prefer copying only `Training_data_cleaned/` + needed `outputs/teachers/...` for resume.
- **`.run.lock` blocks rerun**: in Linux scripts, set `UNLOCK_STALE=1` (and only if safe `FORCE_UNLOCK=1`).
  - Example:
    - `UNLOCK_STALE=1 CUDA_DEVICE=0 bash scripts/linux/run_teachers_2stage_b3.sh`
