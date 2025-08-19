#!/bin/bash
set -eux

python3 train.py test --debug
python3 train.py test --debug --agent_type avg
python3 train.py test --debug --agent_type ppo
python3 train.py test --debug --encoder=ae
python3 train.py test --debug --encoder=simple
