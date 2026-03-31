#!/bin/bash
# Define constants

algo=("local" "fedavg" "fedprox" "fedditto")
dataset=("mnist" "cifar10")
partition=("iid" "shard")

# python local.py --dataset 
for d in "${dataset[@]}"; do
    for a in "${algo[@]}"; do
        for p in "${partition[@]}"; do
            # python "$a.py" -bs $bs -gr $gr -did $did -ien "$ien" -d $d -ss $ss
            uv run "$a.py" --dataset "$d" --partition "$p" --optimizer "sgd" --device "mps" --global_rounds 100
        done
    done
done
