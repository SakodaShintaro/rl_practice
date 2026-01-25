#!/bin/bash
set -eux

cd $(dirname $0)

uv sync

CARLA_VERSION=0.9.16
CARLA_WHL="$HOME/CARLA_${CARLA_VERSION}/PythonAPI/carla/dist/carla-${CARLA_VERSION}-cp310-cp310-manylinux_2_31_x86_64.whl"

test -f "$CARLA_WHL" || {
    echo "CARLA not found. Please download from https://github.com/carla-simulator/carla/releases and extract to \$HOME/CARLA_${CARLA_VERSION}"
    exit 1
}

uv pip install "$CARLA_WHL"

sudo apt install -y xdotool wmctrl gnome-screenshot tesseract-ocr
