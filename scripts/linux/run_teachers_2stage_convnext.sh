#!/usr/bin/env bash
set -euo pipefail

# ConvNeXt-Tiny two-stage trainer for Linux servers.
# Stage A (224): include FERPlus
# Stage B (384): exclude FERPlus, init from Stage A best.pt (fallback checkpoint_last.pt)

STAGE="${STAGE:-AthenB}"              # A | B | AthenB
STAGE_A_IMAGE_SIZE="${STAGE_A_IMAGE_SIZE:-224}"
STAGE_B_IMAGE_SIZE="${STAGE_B_IMAGE_SIZE:-384}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-60}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-1337}"
ACCUM_STEPS="${ACCUM_STEPS:-1}"
EVAL_EVERY="${EVAL_EVERY:-1}"
MANIFEST_PRESET="${MANIFEST_PRESET:-full}"   # full | hq
CUDA_DEVICE="${CUDA_DEVICE:-}"

NO_CLAHE="${NO_CLAHE:-0}"
SKIP_ONNX_DURING_TRAIN="${SKIP_ONNX_DURING_TRAIN:-0}"
SMOKE="${SMOKE:-0}"

CLEAN="${CLEAN:-0}"
CLEAN_STAGE_A="${CLEAN_STAGE_A:-0}"
CLEAN_STAGE_B="${CLEAN_STAGE_B:-0}"
UNLOCK_STALE="${UNLOCK_STALE:-0}"
FORCE_UNLOCK="${FORCE_UNLOCK:-0}"

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

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
  echo "ERROR: train_teacher.py not found at $TRAIN_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "ERROR: manifest not found at $MANIFEST" >&2
  exit 2
fi

TAG="CNXT"
MODEL_NAME="convnext_tiny"
STAGE_A_INCLUDE="ferplus,rafdb_basic,affectnet_full_balanced,expw_hq"
STAGE_B_EXCLUDE="ferplus"

STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO_ROOT/outputs/teachers/_overnight_logs_2stage/${TAG}_${STAMP}"
mkdir -p "$LOG_DIR"

BASE_OUT="$REPO_ROOT/outputs/teachers/${TAG}_${MODEL_NAME}_seed${SEED}"
OUT_A="${BASE_OUT}_stageA_img${STAGE_A_IMAGE_SIZE}"
OUT_B="${BASE_OUT}_stageB_img${STAGE_B_IMAGE_SIZE}"

is_locked() {
  [[ -f "$1/.run.lock" ]]
}

unlock_if_stale() {
  local dir="$1"
  local lock="$dir/.run.lock"
  [[ -f "$lock" ]] || return 0

  if [[ "$FORCE_UNLOCK" == "1" ]]; then
    rm -f "$lock" || true
    return 0
  fi

  local pid=""
  pid="$(python -c "import json,sys; print(json.load(open(sys.argv[1]))['pid'])" "$lock" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    return 0
  fi

  if [[ -d "/proc/$pid" ]]; then
    return 0
  fi

  rm -f "$lock" || true
}

remove_dir_safe() {
  local dir="$1"
  [[ -e "$dir" ]] || return 0

  if is_locked "$dir"; then
    if [[ "$UNLOCK_STALE" == "1" ]]; then
      unlock_if_stale "$dir"
    fi
    if is_locked "$dir"; then
      echo "ERROR: refusing to delete; lock exists: $dir/.run.lock" >&2
      exit 3
    fi
  fi

  rm -rf "$dir"
}

run_stage() {
  local stage_name="$1"
  local image_size="$2"
  local output_dir="$3"
  local log_path="$4"
  local include_sources="$5"
  local exclude_sources="$6"
  local init_from="$7"

  if is_locked "$output_dir"; then
    if [[ "$UNLOCK_STALE" == "1" ]]; then
      unlock_if_stale "$output_dir"
    fi
    if is_locked "$output_dir"; then
      echo "ERROR: output dir is locked: $output_dir/.run.lock" >&2
      exit 3
    fi
  fi

  local args=(
    "$TRAIN_SCRIPT"
    --model "$MODEL_NAME"
    --manifest "$MANIFEST"
    --out-root "$OUT_ROOT"
    --image-size "$image_size"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --seed "$SEED"
    --accum-steps "$ACCUM_STEPS"
    --eval-every "$EVAL_EVERY"
    --max-epochs "$EPOCHS"
    --min-lr 1e-5
    --checkpoint-every 10
    --output-dir "$output_dir"
  )

  if [[ -n "$include_sources" ]]; then
    args+=(--include-sources "$include_sources")
  fi
  if [[ -n "$exclude_sources" ]]; then
    args+=(--exclude-sources "$exclude_sources")
  fi
  if [[ -n "$init_from" ]]; then
    args+=(--init-from "$init_from")
  fi
  if [[ "$NO_CLAHE" != "1" ]]; then
    args+=(--clahe)
  fi
  if [[ "$SKIP_ONNX_DURING_TRAIN" == "1" ]]; then
    args+=(--skip-onnx-during-train)
  fi
  if [[ "$SMOKE" == "1" ]]; then
    args+=(--smoke)
  fi

  echo ""
  echo "=== [$TAG] $MODEL_NAME | $stage_name | img=$image_size ==="
  echo "OutputDir: $output_dir"
  echo "Log: $log_path"

  python "${args[@]}" 2>&1 | tee "$log_path"

  local required=(alignmentreport.json history.json reliabilitymetrics.json calibration.json checkpoint_last.pt)
  for f in "${required[@]}"; do
    if [[ ! -f "$output_dir/$f" ]]; then
      echo "ERROR: missing artifact: $output_dir/$f" >&2
      exit 4
    fi
  done
}

echo "Repo: $REPO_ROOT"
echo "Logs: $LOG_DIR"
echo "Stage=$STAGE Model=$MODEL_NAME ManifestPreset=$MANIFEST_PRESET BatchSize=$BATCH_SIZE Epochs=$EPOCHS NumWorkers=$NUM_WORKERS EvalEvery=$EVAL_EVERY"

if [[ "$STAGE" == "A" || "$STAGE" == "AthenB" ]]; then
  if [[ "$CLEAN" == "1" || "$CLEAN_STAGE_A" == "1" ]]; then
    echo "Cleaning Stage A: $OUT_A"
    remove_dir_safe "$OUT_A"
  fi
  run_stage "StageA" "$STAGE_A_IMAGE_SIZE" "$OUT_A" "$LOG_DIR/${TAG}_${MODEL_NAME}_stageA_img${STAGE_A_IMAGE_SIZE}.log" "$STAGE_A_INCLUDE" "" ""
fi

if [[ "$STAGE" == "B" || "$STAGE" == "AthenB" ]]; then
  if [[ "$CLEAN" == "1" || "$CLEAN_STAGE_B" == "1" ]]; then
    echo "Cleaning Stage B: $OUT_B"
    remove_dir_safe "$OUT_B"
  fi

  init_from=""
  if [[ ! -f "$OUT_B/checkpoint_last.pt" ]]; then
    if [[ -f "$OUT_A/best.pt" ]]; then
      init_from="$OUT_A/best.pt"
    elif [[ -f "$OUT_A/checkpoint_last.pt" ]]; then
      init_from="$OUT_A/checkpoint_last.pt"
    else
      echo "ERROR: Stage B requested but no Stage A checkpoint found in $OUT_A" >&2
      exit 2
    fi
  fi

  run_stage "StageB" "$STAGE_B_IMAGE_SIZE" "$OUT_B" "$LOG_DIR/${TAG}_${MODEL_NAME}_stageB_img${STAGE_B_IMAGE_SIZE}.log" "" "$STAGE_B_EXCLUDE" "$init_from"
fi

echo ""
echo "ConvNeXt two-stage run completed."
