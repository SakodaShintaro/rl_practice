#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py baseline$suffix \
  --env_id Hopper-v5 \
  --agent_type off_policy \
  --network_class actor_critic_with_action_value \
  --action_norm_penalty 0.1 \
  --encoder spatial_temporal \
  --num_bins 1 \
  --value_range 100.0 \
  --step_limit 500_000 \
  --image_processor_type ae \
  --seq_len 8 \
  --batch_size 16 \
  --accumulation_steps 1 \
  --learning_rate 1e-4 \
  --policy_type diffusion \
