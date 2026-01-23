#!/bin/bash

# CARLA Leaderboard環境でのオフポリシー学習
# 実行前にCARLAサーバーを起動してください:
#   cd ~/CARLA_0.9.16
#   ./CarlaUE4.sh

export TOKENIZERS_PARALLELISM=false

python train.py carla_test \
    --env_id CARLA-Leaderboard-v0 \
    --agent_type off_policy \
    --encoder qwenvl \
    --image_processor_type ae \
    --step_limit 400_000 \
    --target_score 100.0 \
    --disable_state_predictor 1 \
    --seq_len 1 \
    --use_quantization 0 \
    --use_lora 0 \
    --target_layer_idx -1 \
