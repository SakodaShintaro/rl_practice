#!/bin/bash
set -eux

cd $(dirname $0)

python3 train.py baseline \
  --env_id CarRacing-v3 \
  --action_norm_penalty 0.1 \
