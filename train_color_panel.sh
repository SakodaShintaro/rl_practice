#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py baseline$suffix \
  --env_id ColorPanel-v0 \
  --agent_type off_policy \
  --network_class vlm_actor_critic_with_action_value \
  --step_limit 100_000 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --use_done 0 \
  --disable_state_predictor 1 \
  --state_mode expert \
