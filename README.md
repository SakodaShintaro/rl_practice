# vla_streaming_rl

Reinforcement learning experiments with visual observations.

## Installation

NVIDIA GPU with driver is required for training.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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
vla_streaming_rl/
├── src/vla_streaming_rl/     # Library code
│   ├── agents/          # RL agents (on-policy, off-policy)
│   ├── envs/            # Custom environments
│   ├── networks/        # Neural network architectures
│   ├── metrics/         # Metrics computation
│   └── optimizers/      # Custom optimizers
├── scripts/             # Executable scripts
│   └── train.py         # Main training script
└── train_*.sh           # Training shell scripts
```
