# CARLA Environment Setup

## Setup

```bash
# Download binary from the following URL and extract to $HOME
https://github.com/carla-simulator/carla/releases

# Install dependencies
pip install ~/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
```

## Usage

```bash
# Terminal 1: Start CARLA server
cd ~/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Low

# Terminal 2: Run training
./train_carla_off_policy.sh
```

## Troubleshooting

```bash
# Check CARLA server
ps aux | grep CarlaUE4
lsof -i :2000

# Check CARLA package
python -c "import carla; print(carla.__version__)"
```
