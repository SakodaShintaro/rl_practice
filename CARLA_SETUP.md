# CARLA環境のセットアップ

## セットアップ

```bash
# 以下からバイナリダウンロードして$HOMEに展開
https://github.com/carla-simulator/carla/releases

# 依存関係インストール
pip install ~/CARLA_0.9.16/PythonAPI/carla/dist/carla-0.9.16-cp310-cp310-manylinux_2_31_x86_64.whl
```

## 使用方法

```bash
# ターミナル1: CARLAサーバー起動
cd ~/CARLA_0.9.16
./CarlaUE4.sh -quality-level=Low

# ターミナル2: 学習実行
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
