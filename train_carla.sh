#!/bin/bash
set -eux

trap 'kill 0' EXIT

suffix=${1:-""}
cd $(dirname $0)

CARLA_ROOT=${CARLA_ROOT:-$HOME/CARLA_0.9.16}
B2D_ROOT=$(pwd)/external/Bench2Drive
export PYTHONPATH=${B2D_ROOT}/leaderboard:${B2D_ROOT}/scenario_runner:${CARLA_ROOT}/PythonAPI/carla:${PYTHONPATH:-}
export SCENARIO_RUNNER_ROOT=${B2D_ROOT}/scenario_runner

pgrep -f CarlaUE4 > /dev/null || setsid ${CARLA_ROOT}/CarlaUE4.sh -RenderOffScreen &

uv run python scripts/train.py \
  agent=vlm_streaming \
  env=carla \
  exp_name=carla$suffix \
