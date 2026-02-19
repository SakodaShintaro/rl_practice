#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py baseline$suffix \
  --env_id CarRacing-v3 \
  --agent_type streaming \
  --network_class actor_critic_with_action_value \
  --action_norm_penalty 1.0 \
  --encoder spatial_temporal \
  --num_bins 1 \
  --value_range 200.0 \
  --step_limit 400_000 \
  --eval_range 20 \
  --image_processor_type ae \
  --seq_len 8 \
  --batch_size 16 \
  --accumulation_steps 1 \
  --learning_rate 1e-5 \
  --policy_type diffusion \
