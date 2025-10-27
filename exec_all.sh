#!/bin/bash
set -x

cd $(dirname $0)

./train_car_racing_off_policy_pathwise.sh
# ./train_car_racing_on_policy_pathwise.sh
# ./train_car_racing_on_policy_score.sh
./train_minigrid_memory_off_policy_pathwise.sh
# ./train_minigrid_memory_on_policy_pathwise.sh
./train_minigrid_memory_on_policy_score.sh
