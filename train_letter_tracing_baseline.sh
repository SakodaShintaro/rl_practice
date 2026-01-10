#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)
source ./.venv/bin/activate

python3 train.py baseline$suffix \
  --env_id LetterTracing-v0 \
  --agent_type off_policy \
  --action_norm_penalty 0.0 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --temporal_model_type transformer \
  --num_bins 1 \
  --value_range 200.0 \
  --step_limit 40_000 \
  --eval_range 20 \
  --image_processor_type ae \
  --seq_len 1 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --buffer_device cpu \
  --render 0 \
