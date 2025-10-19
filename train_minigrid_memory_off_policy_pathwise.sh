#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py pathwise \
  --agent_type off_policy \
  --encoder temporal_only \
  --tempo_block_type gru \
