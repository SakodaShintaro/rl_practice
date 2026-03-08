#!/bin/bash
set -eux

suffix=${1:-""}
cd $(dirname $0)

# resultsが空であることを確認する
if [ -d "results" ] && [ "$(ls -A results)" ]; then
  echo "Error: results directory is not empty. Please move or delete the existing results before running this script."
  exit 1
fi

git show -s > results/git_show.txt
git diff > results/git_diff.txt

STEP_LIMIT=400_000

for network_class in actor_critic_with_action_value vlm_actor_critic_with_action_value; do

  # 方策オフ・バッチサイズ16・学習率1e-4
  uv run python scripts/train.py ${network_class}_off_policy_bs16$suffix \
    --env_id CarRacing-v3 \
    --agent_type off_policy \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --batch_size 16 \
    --learning_rate 1e-4

  # 方策オフ・バッチサイズ1・学習率5e-6
  uv run python scripts/train.py ${network_class}_off_policy_bs1$suffix \
    --env_id CarRacing-v3 \
    --agent_type off_policy \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --batch_size 1 \
    --learning_rate 5e-6

  # ストリーミング・適格度トレースなし・学習率5e-6
  uv run python scripts/train.py ${network_class}_streaming$suffix \
    --env_id CarRacing-v3 \
    --agent_type streaming \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --use_eligibility_trace 0 \
    --learning_rate 5e-6

  # ストリーミング・適格度トレースあり・学習率5e-6
  uv run python scripts/train.py ${network_class}_streaming_et$suffix \
    --env_id CarRacing-v3 \
    --agent_type streaming \
    --network_class $network_class \
    --step_limit $STEP_LIMIT \
    --use_eligibility_trace 1 \
    --learning_rate 5e-6

done
