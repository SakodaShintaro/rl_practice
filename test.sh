#!/bin/bash
set -eux

python3 train.py test --debug --agent_type sac --encoder spatial_temporal --tempo_block_type transformer
python3 train.py test --debug --agent_type sac --encoder temporal_only
python3 train.py test --debug --agent_type ppo --encoder temporal_only
python3 train.py test --debug --agent_type ppo --encoder temporal_only --use_action_value=1
python3 train.py test --debug --agent_type avg --encoder spatial_temporal --tempo_block_type transformer
