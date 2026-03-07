#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py qwenvl$suffix \
  --env_id CarRacing-v3 \
  --agent_type off_policy \
  --vlm_model_id "Qwen/Qwen3.5-0.8B" \
  --action_norm_penalty 1.0 \
  --network_class vlm_actor_critic_with_action_value \
  --num_bins 1 \
  --value_range 200.0 \
  --step_limit 400_000 \
  --image_processor_type ae \
  --disable_state_predictor 0 \
  --seq_len 1 \
  --use_lora 0 \
  --batch_size 16 \
  --learning_rate 1e-5 \
