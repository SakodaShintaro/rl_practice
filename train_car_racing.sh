#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py baseline \
  --env_id CarRacing-v3 \
  --action_norm_penalty 0.1 \
  --target_score 800.0 \
  --agent_type sac \
  --seq_len 8 \
  --value_range 60 \
  --learning_rate 1e-4 \
  --step_limit 40_000 \
  --eval_range 20 \
