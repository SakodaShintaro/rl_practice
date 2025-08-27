#!/bin/bash
set -eux

cd $(dirname $0)

pip3 install -r ./requirements.txt

# バージョン合わせゲーム
# https://pytorch.org/get-started/previous-versions/

# https://github.com/state-spaces/mamba/releases
pip install --force-reinstall https://github.com/state-spaces/mamba/releases/download/v2.2.5/mamba_ssm-2.2.5+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
pip install torch==2.7.0 torchvision==0.22.0

# https://github.com/Dao-AILab/causal-conv1d/releases
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.2/causal_conv1d-1.5.2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# https://github.com/Dao-AILab/flash-attention/releases
pip3 install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.1/flash_attn-2.8.1+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# others
pip install peft
pip install fschat
pip install sentencepiece
