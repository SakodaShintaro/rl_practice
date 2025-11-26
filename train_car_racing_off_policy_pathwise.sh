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
  --step_limit 40_000 \
  --eval_range 20 \
  --image_processor_type=ae \
  --disable_state_predictor 1 \
  --seq_len 1 \
  --batch_size 16 \
