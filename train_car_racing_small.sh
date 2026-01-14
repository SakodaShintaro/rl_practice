#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

python3 train.py small$suffix \
  --env_id CarRacing-v3 \
  --agent_type off_policy \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --temporal_model_type transformer \
  --num_bins 1 \
  --value_range 200.0 \
  --step_limit 400_000 \
  --eval_range 20 \
  --image_processor_type ae \
  --seq_len 1 \
  --batch_size 1 \
  --learning_rate 1e-5 \
  --network_class actor_critic \
