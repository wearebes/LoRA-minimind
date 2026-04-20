#!/bin/bash

BASE_CMD="python trainer/train_lora/train.py \
  --save_dir out/lora \
  --lora_rank 8 \
  --epochs 8 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --use_swanlab \
  --data_path dataset/lora_dataset/train.jsonl"

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
    TARGET_CLEAN=$(echo "$targets" | tr ',' '_')
    NAME="lora_${TARGET_CLEAN}_top${top}"
    echo "Training: $NAME"
    $BASE_CMD \
      --lora_name "$NAME" \
      --target_modules "$targets" \
      --lora_top_layers "$top"
  done
done
