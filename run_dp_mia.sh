#!/bin/bash

algo=("fedavg_dp_MIA" "fedprox_dp_MIA" "fedditto_dp_MIA")
dataset=("cifar10")
partition=("iid" "shard")

GLOBAL_ROUNDS=50
C=("0.5" "1.0" "5.0" "1e6")
SIGMA=("0.0" "0.1" "0.5" "1.0")
Q=("0.3" "0.5" "0.7" "1.0")

for d in "${dataset[@]}"; do
    for a in "${algo[@]}"; do
        for p in "${partition[@]}"; do
            for c in "${C[@]}"; do
                for s in "${SIGMA[@]}"; do
                    for q in "${Q[@]}"; do
                        echo "Running $a on $d with $p partition, C=$c, sigma=$s, q=$q"
                        uv run "$a.py" \
                            --dataset "$d" \
                            --num_clients 5 \
                            --partition "$p" \
                            --optimizer "dp_sgd" \
                            --device "mps" \
                            --global_rounds "$GLOBAL_ROUNDS" \
                            --dp_noise_multiplier "$SIGMA" \
                            --dp_sample_rate "$Q" \
                            --dp_max_grad_norm "$c" 
                    done
                done
            done
        done
    done
done
