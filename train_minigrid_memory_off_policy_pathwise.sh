#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  --config-name minigrid_memory_off_policy_pathwise \
  exp_name=pathwise$suffix
