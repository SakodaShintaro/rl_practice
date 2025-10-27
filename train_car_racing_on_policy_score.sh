#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py score \
  --env_id CarRacing-v3 \
  --agent_type on_policy \
  --use_action_value=0 \
  --reward_scale 0.1 \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --temporal_model_type transformer \
  --seq_len 8 \
  --num_bins 1 \
  --value_range 60.0 \
  --learning_rate 1e-4 \
  --step_limit 200_000 \
  --eval_range 20 \
  --policy_type Beta \
