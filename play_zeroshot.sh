#!/bin/bash
set -eux

python3 play_zeroshot.py --render=0
python3 play_zeroshot.py --render=0 --agent_type qwenvl
python3 play_zeroshot.py --render=0 --agent_type mmmamba
