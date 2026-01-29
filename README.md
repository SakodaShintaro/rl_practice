# rl_practice

Reinforcement learning experiments with visual observations.

## Installation

NVIDIA GPU with driver is required for training.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install system dependencies

```bash
sudo apt update
sudo apt install -y swig libgl1-mesa-glx libglfw3 mesa-utils
```

### Setup project

```bash
uv sync
```

### Login to Hugging Face (for model downloads)

```bash
uv run huggingface-cli login
```

## Usage

### Training

```bash
./train_car_racing_on_policy.sh
```

### Testing

```bash
./test.sh
```

## Project Structure

```bash
rl_practice/
├── src/rl_practice/     # Library code
│   ├── agents/          # RL agents (on-policy, off-policy)
│   ├── envs/            # Custom environments
│   ├── networks/        # Neural network architectures
│   ├── metrics/         # Metrics computation
│   └── optimizers/      # Custom optimizers
├── scripts/             # Executable scripts
│   └── train.py         # Main training script
└── train_*.sh           # Training shell scripts
```

## Available Environments

- CarRacing-v3
- MiniGrid-MemoryS7-v0
- MiniGrid-MemoryS9-v0
- MiniGrid-MemoryS11-v0
- MiniGrid-MemoryS13-v0
- MiniGrid-MemoryS13Random-v0
- MiniGrid-MemoryS17Random-v0
- CARLA-Leaderboard-v0
- LetterTracing-v0
- FourQuadrant-v0
