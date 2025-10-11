#!/bin/bash
set -eux

python3 train.py test --debug --agent_type sac --encoder stt
python3 train.py test --debug --agent_type sac --encoder simple
python3 train.py test --debug --agent_type sac --encoder recurrent
python3 train.py test --debug --agent_type avg
python3 train.py test --debug --agent_type ppo
