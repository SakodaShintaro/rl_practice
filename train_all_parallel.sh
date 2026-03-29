#!/bin/bash
# SPDX-License-Identifier: MIT
set -eux

trap 'kill 0' EXIT

RESULT_DIR=${1:?"Usage: $0 <result_dir>"}
RESULT_DIR=$(readlink -f "$RESULT_DIR")

NUM_GPUS=$(nvidia-smi -L | wc -l)

pids=()
for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
  CUDA_VISIBLE_DEVICES=$gpu_id bash train_all.sh "${RESULT_DIR}/${gpu_id}" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  wait "$pid"
done
