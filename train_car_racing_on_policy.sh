#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  --config-name car_racing_on_policy \
  exp_name=baseline$suffix
