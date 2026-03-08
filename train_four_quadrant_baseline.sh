#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py baseline$suffix \
  --env_id FourQuadrant-v0 \
  --agent_type off_policy \
  --step_limit 40_000 \
  --batch_size 64 \
  --learning_rate 1e-5 \
  --render 1 \
  --use_done 0 \
  --disable_state_predictor 1 \
