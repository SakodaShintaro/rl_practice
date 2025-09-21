#!/bin/bash
set -eux

python3 train.py test --debug
python3 train.py test --debug --agent_type avg
python3 train.py test --debug --agent_type ppo
python3 train.py test --debug --encoder=single_frame
python3 train.py test --debug --encoder=simple
python3 train.py test --debug --encoder=recurrent
python3 train.py test --debug --encoder=recurrent --agent_type ppo
