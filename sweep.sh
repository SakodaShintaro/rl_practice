#!/bin/bash
set -eux

wandb sweep ./configs/sweep.yaml \
  --project vla_streaming_rl_CarRacing-v3 \
