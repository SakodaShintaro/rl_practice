#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

python3 train.py pathwise$suffix \
  --env_id CarRacing-v3 \
  --agent_type off_policy \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --temporal_model_type transformer \
  --seq_len 8 \
  --num_bins 1 \
  --value_range 60.0 \
  --step_limit 40_000 \
  --eval_range 20 \
