#!/bin/bash
set -x

suffix=${1:-""}
cd $(dirname $0)

./train_car_racing_off_policy_pathwise.sh $suffix
# ./train_car_racing_on_policy_pathwise.sh $suffix
# ./train_car_racing_on_policy_score.sh $suffix
./train_minigrid_memory_off_policy_pathwise.sh $suffix
# ./train_minigrid_memory_on_policy_pathwise.sh $suffix
./train_minigrid_memory_on_policy_score.sh $suffix
