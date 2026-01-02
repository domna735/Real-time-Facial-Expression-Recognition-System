#!/usr/bin/env bash
set -euo pipefail

# Resume RN18 Stage B (384) from an existing output dir checkpoint_last.pt.
# This is intended for: "RN18 stage B i stop at epoch 39 help me to resume".
#
# IMPORTANT:
# - This script does NOT pass --init-from.
# - train_teacher.py will auto-resume from <output-dir>/checkpoint_last.pt if it exists.
#
# Usage examples:
#   CUDA_DEVICE=3 bash scripts/linux/resume_rn18_stageB.sh
#   EVAL_EVERY=10 CUDA_DEVICE=3 bash scripts/linux/resume_rn18_stageB.sh

CUDA_DEVICE="${CUDA_DEVICE:-}"
MANIFEST_PRESET="${MANIFEST_PRESET:-hq}"   # hq | full
STAGE_B_IMAGE_SIZE="${STAGE_B_IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1337}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
EVAL_EVERY="${EVAL_EVERY:-10}"

NO_CLAHE="${NO_CLAHE:-0}"
SKIP_ONNX_DURING_TRAIN="${SKIP_ONNX_DURING_TRAIN:-0}"

if [[ -n "$CUDA_DEVICE" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if (( BATCH_SIZE < 14 )); then
  echo "ERROR: BATCH_SIZE must be >= 14 (7 classes x min_per_class=2). Got: $BATCH_SIZE" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

TRAIN_SCRIPT="$REPO_ROOT/scripts/train_teacher.py"
OUT_ROOT="$REPO_ROOT/Training_data_cleaned"

if [[ "$MANIFEST_PRESET" == "full" ]]; then
  MANIFEST="$REPO_ROOT/Training_data_cleaned/classification_manifest.csv"
elif [[ "$MANIFEST_PRESET" == "hq" ]]; then
  MANIFEST="$REPO_ROOT/Training_data_cleaned/classification_manifest_hq_train.csv"
else
  echo "ERROR: MANIFEST_PRESET must be 'full' or 'hq'. Got: $MANIFEST_PRESET" >&2
  exit 2
fi

OUTPUT_DIR="$REPO_ROOT/outputs/teachers/RN18_resnet18_seed${SEED}_stageB_img${STAGE_B_IMAGE_SIZE}"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: train_teacher.py not found at $TRAIN_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found at $MANIFEST" >&2
  exit 2
fi

if [[ ! -f "$OUTPUT_DIR/checkpoint_last.pt" ]]; then
  echo "ERROR: no checkpoint to resume: $OUTPUT_DIR/checkpoint_last.pt" >&2
  echo "You must copy your local Stage B output folder to the server first." >&2
  exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/outputs/teachers/_overnight_logs_2stage/RN18_${STAMP}"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/RN18_resnet18_stageB_img${STAGE_B_IMAGE_SIZE}_resume.log"

echo "Repo: $REPO_ROOT"
echo "OutputDir: $OUTPUT_DIR"
echo "Log: $LOG_PATH"

a=(
  "$TRAIN_SCRIPT"
  --model resnet18
  --manifest "$MANIFEST"
  --out-root "$OUT_ROOT"
  --image-size "$STAGE_B_IMAGE_SIZE"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --seed "$SEED"
  --accum-steps "$ACCUM_STEPS"
  --eval-every "$EVAL_EVERY"
  --max-epochs "$EPOCHS"
  --min-lr 1e-5
  --checkpoint-every 10
  --output-dir "$OUTPUT_DIR"
  --exclude-sources ferplus
)

if [[ "$NO_CLAHE" != "1" ]]; then
  a+=(--clahe)
fi
if [[ "$SKIP_ONNX_DURING_TRAIN" == "1" ]]; then
  a+=(--skip-onnx-during-train)
fi

python "${a[@]}" 2>&1 | tee "$LOG_PATH"

echo "RN18 Stage B resume finished (or still running in tmux)."
