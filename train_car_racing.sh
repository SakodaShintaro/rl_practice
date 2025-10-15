#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py baseline \
  --env_id CarRacing-v3 \
  --reward_scale 0.1 \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --agent_type sac \
  --encoder spatial_temporal \
  --tempo_block_type transformer \
  --seq_len 8 \
  --num_bins 1 \
  --learning_rate 1e-4 \
  --step_limit 40_000 \
  --eval_range 20 \
  --disable_state_predictor=0 \
