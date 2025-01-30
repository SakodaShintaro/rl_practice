#!/bin/bash
set -eux

trap "kill 0" EXIT

cd $(dirname $0)

PARALLEL_JOBS=6

SEED=0

datatime=$(date "+%Y%m%d_%H%M%S")

export WANDB_RUN_GROUP=${datatime}

save_dir="results/${datatime}"

# コマンドを配列に格納
commands=(
    "python3 avg.py --seed=${SEED} --save_dir=${save_dir} --normalizer_type=none --save_suffix=none"
    "python3 sac.py --seed=${SEED} --save_dir=${save_dir} --normalizer_type=none --save_suffix=none"
)

# 並列実行を管理
for cmd in "${commands[@]}"; do
    # 現在実行中のジョブ数をカウント
    while (( $(jobs -p | wc -l) >= PARALLEL_JOBS )); do
        # 既存のジョブが終了するのを待つ
        wait -n
    done

    # バックグラウンドでコマンドを実行
    eval "$cmd" &
    sleep 2
done

# 残りのすべてのジョブが完了するのを待つ
wait

echo "All jobs completed"
