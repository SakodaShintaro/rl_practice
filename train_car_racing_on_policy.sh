#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=base_on_policy \
  env=car_racing \
  exp_name=baseline$suffix
