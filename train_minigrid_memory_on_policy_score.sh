#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py score \
  --agent_type on_policy \
  --use_action_value=0 \
  --encoder temporal_only \
  --temporal_model_type gru \
  --use_done=1 \
