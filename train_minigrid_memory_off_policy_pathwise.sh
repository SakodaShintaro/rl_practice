#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py pathwise \
  --agent_type off_policy \
  --encoder temporal_only \
  --temporal_model_type transformer \
  --step_limit 400_000 \
  --use_done=1 \
  --denoising_time 0.8 \
