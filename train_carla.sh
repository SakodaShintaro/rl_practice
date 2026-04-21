#!/bin/bash
set -eux

trap 'kill 0' EXIT

suffix=${1:-""}
cd $(dirname $0)

pgrep -f CarlaUE4 > /dev/null || ~/CARLA_0.9.16/CarlaUE4.sh -RenderOffScreen &

uv run python scripts/train.py \
  agent=vlm_streaming \
  env=carla \
  exp_name=carla$suffix
