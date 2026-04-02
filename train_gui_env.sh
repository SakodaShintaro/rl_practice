#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

ENV_ID=FourQuadrant-v0
ENV_ID=ColorPanel-v0
ENV_ID=TrackingSquare-v0

uv run python scripts/train.py baseline$suffix \
  --env_id $ENV_ID \
  --agent_type off_policy \
  --network_class vlm_actor_critic_with_action_value \
  --step_limit 50_000 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --disable_state_predictor 1 \
  --state_mode expert \
  --use_lora 1 \
