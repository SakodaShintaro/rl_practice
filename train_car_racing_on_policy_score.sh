#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

python3 train.py score$suffix \
  --env_id CarRacing-v3 \
  --agent_type on_policy \
  --use_action_value=0 \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --num_bins 1 \
  --value_range 60.0 \
  --step_limit 200_000 \
  --eval_range 20 \
  --policy_type Beta \
  --image_processor_type=ae \
