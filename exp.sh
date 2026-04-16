#!/bin/bash
set -eux

ENV_PATH=${1:?"Usage: $0 <env_config_path>  (e.g. configs/env/gui.yaml)"}
cd $(dirname $0)

ENV_NAME=$(basename "$ENV_PATH" .yaml)
ENV_ID=$(grep '^env_id:' "$ENV_PATH" | awk '{print $2}')

sed "s|value: gui|value: ${ENV_NAME}|" configs/exp.yaml \
  | wandb sweep /dev/stdin --project "vla_streaming_rl_${ENV_ID}"
