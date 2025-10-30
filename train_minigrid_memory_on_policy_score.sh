#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

python3 train.py score$suffix \
  --agent_type on_policy \
  --use_action_value=0 \
  --encoder temporal_only \
  --temporal_model_type gru \
  --disable_state_predictor=1 \
