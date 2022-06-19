# docker build -t rl_practice:20220619 .
docker run -it \
    --gpus all \
    --ipc=host \
    # --env="DISPLAY" \
    # --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --name rl_practice_20220619 \
    rl_practice:20220619 bash
