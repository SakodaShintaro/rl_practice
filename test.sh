#!/bin/bash
set -eux

source .venv/bin/activate

python3 train.py test --debug --agent_type off_policy --encoder spatial_temporal --temporal_model_type transformer
python3 train.py test --debug --agent_type off_policy --encoder temporal_only
python3 train.py test --debug --agent_type off_policy --network_class actor_critic_with_action_value
python3 train.py test --debug --agent_type on_policy --encoder temporal_only
