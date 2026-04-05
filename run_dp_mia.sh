#!/bin/bash

algo=("fedavg_dp_MIA" "fedprox_dp_MIA" "fedditto_dp_MIA")
dataset=("mnist" "cifar10")
partition=("iid" "shard")

DEVICE=${DEVICE:-cpu}
GLOBAL_ROUNDS=${GLOBAL_ROUNDS:-50}
NOISE_MULTIPLIER=${NOISE_MULTIPLIER:-1.0}
SAMPLE_RATE=${SAMPLE_RATE:-0.5}
DELTA=${DELTA:-1e-5}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

for d in "${dataset[@]}"; do
    for a in "${algo[@]}"; do
        for p in "${partition[@]}"; do
            uv run "$a.py" \
                --dataset "$d" \
                --partition "$p" \
                --optimizer "dp_sgd" \
                --device "$DEVICE" \
                --global_rounds "$GLOBAL_ROUNDS" \
                --dp_noise_multiplier "$NOISE_MULTIPLIER" \
                --dp_sample_rate "$SAMPLE_RATE" \
                --dp_delta "$DELTA" \
                --dp_max_grad_norm "$MAX_GRAD_NORM"
        done
    done
done
