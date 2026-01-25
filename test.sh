#!/bin/bash
set -eux

uv run python train.py test --debug --batch_size 2 --agent_type on_policy --network_class actor_critic_with_state_value --policy_type beta
uv run python train.py test --debug --batch_size 2 --agent_type off_policy --network_class actor_critic_with_action_value
uv run python train.py test --debug --batch_size 2 --agent_type on_policy --network_class vlm_actor_critic_with_state_value
