# rl_practice

## 環境構築

### 必要なものをインストール

```bash
sudo apt update
sudo apt install -y swig libgl1-mesa-glx libglfw3 mesa-utils python3.10-venv
```

## venv利用

```bash
python3 -m venv .venv
source .venv/bin/activate
./setup.sh
```

```bash
huggingface-cli login
```

## Blackwellの場合

```bash
pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

## 利用可能なMiniGrid-Memory

```bash
  - MiniGrid-MemoryS7-v0
  - MiniGrid-MemoryS9-v0
  - MiniGrid-MemoryS11-v0
  - MiniGrid-MemoryS13-v0
  - MiniGrid-MemoryS13Random-v0
  - MiniGrid-MemoryS17Random-v0
```
