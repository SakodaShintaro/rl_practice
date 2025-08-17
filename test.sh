#!/bin/bash
set -eux

python3 train.py test --debug
python3 train.py test --debug --agent_type avg
python3 train.py test --debug --agent_type ppo
python3 train.py test --debug --encoder=stt
python3 train.py test --debug --encoder=stt --agent_type avg
python3 train.py test --debug --encoder=simple
python3 train.py test --debug --encoder=simple --agent_type avg
