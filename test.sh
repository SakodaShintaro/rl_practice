#!/bin/bash
set -eux

python3 train.py test --debug --agent_type off_policy --encoder spatial_temporal --temporal_model_type transformer
python3 train.py test --debug --agent_type off_policy --encoder temporal_only
python3 train.py test --debug --agent_type on_policy --encoder temporal_only
python3 train.py test --debug --agent_type on_policy --encoder temporal_only --use_action_value=1
