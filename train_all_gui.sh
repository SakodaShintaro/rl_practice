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

for use_prompt in 1 0; do
  uv run python scripts/train.py \
    agent=vlm_streaming \
    env=gui \
    exp_name=use_prompt${use_prompt} \
    step_limit=50_000 \
    learning_rate=1e-5 \
    result_dir=$RESULT_DIR \
    wandb_group=$WANDB_GROUP \
    use_prompt=$use_prompt

done
