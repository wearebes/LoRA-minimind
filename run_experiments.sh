#!/bin/bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

CONDA_ENV="${CONDA_ENV:-minimind}"
SAVE_DIR="${SAVE_DIR:-out/lora}"
TRAIN_RUNS_DIR="${TRAIN_RUNS_DIR:-${SAVE_DIR}/train_runs}"
TRAIN_DATA="${TRAIN_DATA:-dataset/lora_dataset/splits/train.jsonl}"
VAL_DATA="${VAL_DATA:-dataset/lora_dataset/splits/val.jsonl}"
TEST_DATA="${TEST_DATA:-dataset/lora_dataset/splits/test.jsonl}"
SPLIT_OUTPUT_DIR="${SPLIT_OUTPUT_DIR:-$(dirname "$TRAIN_DATA")}"
SPLIT_SEED="${SPLIT_SEED:-42}"
TRAIN_RATIO="${TRAIN_RATIO:-0.8}"
VAL_RATIO="${VAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
FROM_WEIGHT="${FROM_WEIGHT:-full_sft}"
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_HIDDEN_LAYERS="${NUM_HIDDEN_LAYERS:-16}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-340}"
LORA_RANK="${LORA_RANK:-8}"
EPOCHS="${EPOCHS:-8}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
USE_SWANLAB="${USE_SWANLAB:-1}"
SWANLAB_PROJECT="${SWANLAB_PROJECT:-MiniMind-LoRA}"

find_conda_sh() {
  if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    echo "$CONDA_SH"
    return 0
  fi

  local candidates=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "$HOME/mambaforge/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )

  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

if ! CONDA_SH_PATH="$(find_conda_sh)"; then
  echo "Unable to locate conda.sh. Set CONDA_SH or install conda under a standard path." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$CONDA_SH_PATH"
conda activate "$CONDA_ENV"

mkdir -p "$SAVE_DIR" "$TRAIN_RUNS_DIR"

if [[ ! -f "$TRAIN_DATA" || ! -f "$VAL_DATA" || ! -f "$TEST_DATA" ]]; then
  echo "Generating deterministic LoRA splits in $SPLIT_OUTPUT_DIR"
  python scripts/split_lora_dataset.py \
    --output-dir "$SPLIT_OUTPUT_DIR" \
    --seed "$SPLIT_SEED" \
    --train-ratio "$TRAIN_RATIO" \
    --val-ratio "$VAL_RATIO" \
    --test-ratio "$TEST_RATIO"
else
  echo "Using existing splits:"
  echo "  train: $TRAIN_DATA"
  echo "  val:   $VAL_DATA"
  echo "  test:  $TEST_DATA"
fi

BASE_CMD=(
  python trainer/train_lora/train.py
  --save_dir "$SAVE_DIR"
  --data_path "$TRAIN_DATA"
  --eval_data_path "$VAL_DATA"
  --from_weight "$FROM_WEIGHT"
  --hidden_size "$HIDDEN_SIZE"
  --num_hidden_layers "$NUM_HIDDEN_LAYERS"
  --max_seq_len "$MAX_SEQ_LEN"
  --lora_rank "$LORA_RANK"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --swanlab_project "$SWANLAB_PROJECT"
)

if [[ "$USE_SWANLAB" == "1" ]]; then
  BASE_CMD+=(--use_swanlab)
fi

TARGET_MODULES_LIST=(
  "q_proj"
  "k_proj"
  "v_proj"
  "q_proj,k_proj"
  "q_proj,k_proj,v_proj"
  "q_proj,k_proj,v_proj,o_proj"
)

TOP_LAYERS_LIST=(4 8 16)

for targets in "${TARGET_MODULES_LIST[@]}"; do
  for top in "${TOP_LAYERS_LIST[@]}"; do
    TARGET_CLEAN="${targets//,/_}"
    NAME="lora_${TARGET_CLEAN}_top${top}"
    METADATA_DIR="${TRAIN_RUNS_DIR}/${NAME}"
    echo "Training: $NAME"
    "${BASE_CMD[@]}" \
      --lora_name "$NAME" \
      --target_modules "$targets" \
      --lora_top_layers "$top" \
      --metadata_dir "$METADATA_DIR"
  done
done

echo "Running batch evaluation on $TEST_DATA"
LORA_TEST_DATA_PATH="$TEST_DATA" python scripts/run_lora_eval_batch.py

echo "Training weights: $SAVE_DIR"
echo "Training metadata: $TRAIN_RUNS_DIR"
echo "Evaluation results: ${SAVE_DIR}/eval_runs"
