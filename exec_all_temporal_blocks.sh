#!/bin/bash
set -eux

cd $(dirname $0)

type_list=("gru" "transformer" "gdn" "mamba")

for type in "${type_list[@]}"; do
    python3 train.py seq_len16_$type \
    --env_id CarRacing-v3 \
    --agent_type off_policy \
    --action_norm_penalty 1.0 \
    --target_score 800.0 \
    --encoder temporal_only \
    --temporal_model_type $type \
    --num_bins 1 \
    --value_range 200.0 \
    --step_limit 40_000 \
    --eval_range 20 \
    --image_processor_type ae \
    --seq_len 16 \
    --batch_size 32 \
    --learning_rate 1e-4
done
