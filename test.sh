#!/bin/bash
set -eux

source .venv/bin/activate

python3 train.py test --debug --batch_size 2 --agent_type off_policy --network_class actor_critic_with_action_value
python3 train.py test --debug --batch_size 2 --agent_type off_policy --network_class actor_critic_with_action_value --encoder qwenvl
python3 train.py test --debug --batch_size 2 --agent_type on_policy --network_class actor_critic_with_state_value
