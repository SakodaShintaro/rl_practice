#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  --config-name babyai_goto_local \
  exp_name=exp$suffix
