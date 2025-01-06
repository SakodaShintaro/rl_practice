#!/bin/bash
set -eux

trap "kill 0" EXIT

PARALLEL_JOBS=6

SEED=45

# コマンドを配列に格納
commands=(
    "python3 avg.py --seed=${SEED} --save_suffix=baseline"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.0 --save_suffix=et_lambda_0.0"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.1 --save_suffix=et_lambda_0.1"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.2 --save_suffix=et_lambda_0.2"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.4 --save_suffix=et_lambda_0.4"
    "python3 avg.py --seed=${SEED} --use_eligibility_trace --et_lambda=0.8 --save_suffix=et_lambda_0.8"
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
