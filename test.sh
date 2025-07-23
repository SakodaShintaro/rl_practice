#!/bin/bash
set -eux

python3 train.py test --debug
python3 train.py test --debug --agent_type avg
python3 train.py test --debug --agent_type ppo
