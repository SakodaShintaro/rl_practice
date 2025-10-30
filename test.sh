#!/bin/bash
set -eux

python3 train.py exp_name=test debug=true agent=off_policy encoder=spatial_temporal temporal_model_type=transformer
python3 train.py exp_name=test debug=true agent=off_policy encoder=temporal_only
python3 train.py exp_name=test debug=true agent=on_policy encoder=temporal_only
python3 train.py exp_name=test debug=true agent=on_policy encoder=temporal_only use_action_value=1
