#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  --config-name car_racing_streaming \
  exp_name=baseline$suffix
