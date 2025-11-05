#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

python3 train.py score$suffix \
  --agent_type on_policy \
  --use_action_value=0 \
  --encoder temporal_only \
  --disable_state_predictor=1 \
  --image_processor_type=simple_cnn \
  --critic_loss_weight=0.25 \
