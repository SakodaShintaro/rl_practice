#!/bin/bash
set -eux

python3 train.py test --debug --agent_type sac --encoder stt
python3 train.py test --debug --agent_type sac --encoder gru
python3 train.py test --debug --agent_type ppo --encoder gru
python3 train.py test --debug --agent_type avg
