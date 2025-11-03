# CARLA環境のセットアップ

## セットアップ

```bash
# 1. リポジトリクローン
cd ~/work
git clone -b v0.9.16 https://github.com/carla-simulator/scenario_runner.git
git clone https://github.com/carla-simulator/leaderboard.git

# 2. 依存関係インストール
cd ~/work/rl_practice
source .venv/bin/activate
pip install ~/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
pip install -r ~/work/scenario_runner/requirements.txt
pip install dictor pygame pexpect transforms3d
```

## 使用方法

```bash
# ターミナル1: CARLAサーバー起動
cd ~/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Low

# ターミナル2: テスト実行
cd ~/work/rl_practice
source .venv/bin/activate
python test_carla_env.py

# 学習実行
./train_carla_off_policy.sh
```

## トラブルシューティング

```bash
# CARLAサーバー確認
ps aux | grep CarlaUE4
lsof -i :2000

# CARLAパッケージ確認
python -c "import carla; print(carla.__version__)"
```
