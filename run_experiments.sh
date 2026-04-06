#!/bin/bash
# Define constants

algo=("local" "fedavg" "fedprox" "fedditto")
dataset=("mnist" "cifar10")
partition=("iid" "shard")

for d in "${dataset[@]}"; do
    for a in "${algo[@]}"; do
        for p in "${partition[@]}"; do
            uv run "$a.py" --dataset "$d" --partition "$p" --optimizer "sgd" --device "mps" --global_rounds 200 --patience 20 --early_stop_burn_in 100
        done
    done
done
