#!/bin/bash
set -eux

RESULT_DIR=${1:?"Usage: $0 <result_dir> [suffix]"}
RESULT_DIR=$(readlink -f $RESULT_DIR)
suffix=${2:-""}
cd $(dirname $0)

WANDB_GROUP="train_all_$(date +%Y%m%d_%H%M%S)$suffix"

mkdir -p $RESULT_DIR

# Ensure the results directory is empty
if [ -d "$RESULT_DIR" ] && [ "$(ls -A $RESULT_DIR)" ]; then
  echo "Error: $RESULT_DIR is not empty. Please move or delete the existing results before running this script."
  exit 1
fi

git show -s > $RESULT_DIR/git_show.txt
git diff > $RESULT_DIR/git_diff.txt

STEP_LIMIT=100_000
ENV_ID=CarRacing-v3

for network_class in actor_critic_with_action_value vlm_actor_critic_with_action_value; do

  # Off-policy, batch size 16, learning rate 1e-5
  uv run python scripts/train.py ${network_class}_off_policy_bs16$suffix \
    --env_id $ENV_ID \
    --agent_type off_policy \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --batch_size 16 \
    --learning_rate 1e-5 \
    --result_dir $RESULT_DIR \
    --wandb_group $WANDB_GROUP \

  # Off-policy, batch size 1, learning rate 5e-6
  uv run python scripts/train.py ${network_class}_off_policy_bs1$suffix \
    --env_id $ENV_ID \
    --agent_type off_policy \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --batch_size 1 \
    --learning_rate 5e-6 \
    --result_dir $RESULT_DIR \
    --wandb_group $WANDB_GROUP \

  # Streaming, without eligibility trace, learning rate 5e-6
  uv run python scripts/train.py ${network_class}_streaming$suffix \
    --env_id $ENV_ID \
    --agent_type streaming \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --use_eligibility_trace 0 \
    --learning_rate 5e-6 \
    --result_dir $RESULT_DIR \
    --wandb_group $WANDB_GROUP \

  # Streaming, with eligibility trace, learning rate 5e-6
  uv run python scripts/train.py ${network_class}_streaming_et$suffix \
    --env_id $ENV_ID \
    --agent_type streaming \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --use_eligibility_trace 1 \
    --learning_rate 5e-6 \
    --result_dir $RESULT_DIR \
    --wandb_group $WANDB_GROUP \

done
