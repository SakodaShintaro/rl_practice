#!/bin/bash
set -eux

trap 'kill 0' EXIT

suffix=${1:-""}
cd $(dirname $0)

pgrep -f CarlaUE4 > /dev/null || ~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen &

uv run python scripts/train.py carla$suffix \
    --env_id CARLA-Leaderboard-v0 \
    --agent_type off_policy \
    --step_limit 400_000 \
    --learning_rate 1e-5 \
    --network_class vlm_actor_critic_with_action_value \
