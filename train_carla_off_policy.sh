#!/bin/bash

# Off-policy training in the CARLA Leaderboard environment
# Start the CARLA server before running:
#   cd ~/CARLA_0.9.16
#   ./CarlaUE4.sh

export TOKENIZERS_PARALLELISM=false

uv run python scripts/train.py carla_test \
    --env_id CARLA-Leaderboard-v0 \
    --agent_type off_policy \
    --step_limit 400_000 \
    --disable_state_predictor 1 \
