#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py \
  --config-name gui_env \
  exp_name=baseline$suffix
