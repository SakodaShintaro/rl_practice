#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  agent=vlm_off_policy_bs16 \
  env=babyai_goto_local \
  exp_name=exp$suffix
