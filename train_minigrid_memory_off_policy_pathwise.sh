#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=vlm_off_policy_bs8 \
  env=minigrid_memory \
  exp_name=pathwise$suffix
