#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

python3 train.py baseline$suffix \
  --env_id CarRacing-v3 \
  --agent_type on_policy \
  --network_class actor_critic_with_state_value \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --num_bins 1 \
  --value_range 60.0 \
  --step_limit 200_000 \
  --eval_range 20 \
  --image_processor_type ae \
  --seq_len 8 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --buffer_capacity 2048 \
  --policy_type Beta \
