#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py baseline$suffix \
  --env_id Hopper-v5 \
  --agent_type off_policy \
  --network_class actor_critic_with_action_value \
  --step_limit 500_000 \
  --seq_len 8 \
  --batch_size 16 \
  --learning_rate 1e-4 \
