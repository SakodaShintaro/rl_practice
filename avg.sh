#!/bin/bash
set -eux

trap "kill 0" EXIT

PARALLEL_JOBS=3

SEED=45

# コマンドを配列に格納
commands=(
    "python3 avg.py --seed=${SEED}"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.0"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.1"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.2"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.4"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.8"
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
    sleep 1
done

# 残りのすべてのジョブが完了するのを待つ
wait

echo "All jobs completed"
