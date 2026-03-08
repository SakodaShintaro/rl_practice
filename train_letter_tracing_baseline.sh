#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)
uv run python scripts/train.py baseline$suffix \
  --env_id LetterTracing-v0 \
  --agent_type off_policy \
  --step_limit 40_000 \
  --image_processor_type ae \
  --seq_len 1 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --render 1 \
  --use_done 0 \
  --gamma 0.0 \
  --disable_state_predictor 1 \
