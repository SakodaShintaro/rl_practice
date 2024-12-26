#!/bin/bash
set -eux

docker build \
    --build-arg USER_NAME=$(whoami) \
    --build-arg USER_UID=$(id -u) \
    --build-arg USER_GID=$(id -g) \
    -t rl_practice:latest .

docker run -it \
    --gpus all \
    --ipc=host \
    --env="DISPLAY" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume=$HOME/work/rl_practice:$HOME/work/rl_practice \
    --name rl_practice_container \
    rl_practice:latest bash
