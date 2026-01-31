#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

uv run python scripts/train.py baseline$suffix \
  --env_id CarRacing-v3 \
  --agent_type on_policy \
  --network_class actor_critic_with_state_value \
  --action_norm_penalty 1.0 \
  --target_score 800.0 \
  --encoder temporal_only \
  --temporal_model_type identity \
  --num_bins 1 \
  --value_range 60.0 \
  --step_limit 200_000 \
  --eval_range 20 \
  --image_processor_type simple_cnn \
  --seq_len 1 \
  --batch_size 16 \
  --learning_rate 1e-5 \
  --buffer_capacity 2048 \
  --policy_type beta \
  --on_policy_epoch 5 \
  --disable_state_predictor 1 \
  --detach_actor 0 \
  --critic_loss_weight 2.0 \
  --max_grad_norm 0.5 \
  --separate_critic 1 \
  --accumulation_steps 1 \
