#!/bin/bash
set -eux

cd $(dirname $0)

pip3 install -r ./requirements.txt

# Install packages that need to be compiled for the current PyTorch version
# Use --no-build-isolation to ensure compatibility with current environment
pip3 install mamba-ssm --no-build-isolation
pip3 install causal-conv1d --no-build-isolation
pip3 install flash-attn --no-build-isolation

sudo apt install -y xdotool wmctrlt gnome-screenshot
