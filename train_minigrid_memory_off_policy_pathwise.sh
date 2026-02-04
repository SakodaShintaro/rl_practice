#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py pathwise$suffix \
  --agent_type off_policy \
  --encoder temporal_only \
  --step_limit 800_000 \
  --denoising_time 0.8 \
  --image_processor_type=ae \
