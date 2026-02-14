#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# ゲームウィンドウが立ち上がっていなければ自動起動
if ! wmctrl -l | grep -qi "Four Quadrant Game"; then
  uv run python src/rl_practice/envs/four_quadrant_game.py &
  sleep 2
fi

uv run python scripts/train.py baseline$suffix \
  --env_id FourQuadrant-v0 \
  --agent_type off_policy \
  --action_norm_penalty 0.0 \
  --target_score 800.0 \
  --encoder spatial_temporal \
  --num_bins 1 \
  --value_range 200.0 \
  --step_limit 40_000 \
  --eval_range 20 \
  --image_processor_type ae \
  --seq_len 1 \
  --batch_size 64 \
  --learning_rate 1e-5 \
  --render 1 \
  --use_done 0 \
  --gamma 0.0 \
  --disable_state_predictor 1 \
