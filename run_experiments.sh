#!/bin/bash
# Define constants

algo=("local" "fedavg" "fedprox" "fedditto")
dataset=("mnist" "cifar10")
partition=("iid" "shard")

for d in "${dataset[@]}"; do
    for a in "${algo[@]}"; do
        for p in "${partition[@]}"; do
            echo "Running $a on $d with $p partitioning"
            uv run "$a.py" --dataset "$d" --partition "$p" --optimizer "sgd" --device "mps" --global_rounds 50 --patience 10
        done
    done
done
