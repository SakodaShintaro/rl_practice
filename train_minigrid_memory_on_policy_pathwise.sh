#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py pathwise \
  --agent_type on_policy \
  --use_action_value=1 \
  --encoder temporal_only \
  --temporal_model_type gru \
  --disable_state_predictor=1 \
