#!/bin/bash

trap 'kill 0' EXIT

pgrep -f CarlaUE4 > /dev/null || ~/CARLA_0.9.16/CarlaUE4.sh &

uv run python scripts/train.py carla_test \
    --env_id CARLA-Leaderboard-v0 \
    --agent_type off_policy \
    --step_limit 400_000 \
    --disable_state_predictor 1 \
