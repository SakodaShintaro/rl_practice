#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py baseline \
  --env_id CarRacing-v3 \
  --agent_type on_policy \
  --use_action_value=0 \
  --reward_scale 0.1 \
  --action_norm_penalty 1.0 \
  --buffer_size 20000 \
  --learning_starts 2000 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --tempo_block_type transformer \
  --seq_len 8 \
  --num_bins 1 \
  --value_range 60.0 \
  --learning_rate 1e-4 \
  --step_limit 400_000 \
  --eval_range 20 \
  --disable_state_predictor=0 \
  --policy_type Beta \
