#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python train.py pathwise$suffix \
  --env_id MemoryMaze-9x9-v0 \
  --agent_type off_policy \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --num_bins 1 \
  --value_range 60.0 \
  --step_limit 400_000 \
  --eval_range 20 \
  --image_processor_type=ae \
  --denoising_time 0.8 \
