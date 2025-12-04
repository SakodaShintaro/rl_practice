#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

python3 train.py pathwise$suffix \
  --env_id CarRacing-v3 \
  --agent_type off_policy \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --encoder qwenvl \
  --num_bins 1 \
  --value_range 200.0 \
  --step_limit 400_000 \
  --eval_range 20 \
  --image_processor_type=ae \
  --disable_state_predictor 1 \
  --seq_len 1 \
  --use_quantization 1 \
  --use_lora 0 \
  --use_pixel_values 1 \
  --target_layer_idx 2 \
  --batch_size 1 \
  --learning_rate 1e-5 \
