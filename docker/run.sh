# docker build -t rl_practice:latest .
docker run -it \
    --gpus all \
    --ipc=host \
    # --env="DISPLAY" \
    # --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --name rl_practice_container \
    rl_practice:latest bash
