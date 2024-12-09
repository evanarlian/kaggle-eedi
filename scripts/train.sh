#!/usr/bin/env bash
set -Eeuxo pipefail

notify_failure() {
    ./scripts/telegram.sh "training failed"
}

# Trap errors and call the failure notification
trap notify_failure ERR

# 1
accelerate launch eedi/train_sts_hn.py \
    --synthetic-path=data/eedi-synthetic \
    --use-synthetic \
    --model=Qwen/Qwen2.5-1.5B-Instruct \
    --token-pool=last \
    --per-device-bs=8 \
    --lr=2e-4 \
    --warmup-ratio=0.2 \
    --n-epochs=3 \
    --lora-rank=16 \
    --dataset-seed=1 \
    --iterative-hnm \
    --run-name=synthetic-qwen-instruct
./scripts/telegram.sh "synthetic-qwen-instruct done"

# # 2
# accelerate launch eedi/train_sts_hn.py \
#     --synthetic-path=data/eedi-synthetic \
#     --use-synthetic \
#     --model=Qwen/Qwen2.5-1.5B-Instruct \
#     --token-pool=last \
#     --per-device-bs=8 \
#     --lr=2e-4 \
#     --warmup-ratio=0.2 \
#     --n-epochs=5 \
#     --lora-rank=16 \
#     --dataset-seed=1 \
#     --iterative-hnm \
#     --run-name=synthetic-qwen
# ./scripts/telegram.sh "synthetic-qwen done"

# # 3
# accelerate launch eedi/train_sts_hn.py \
#     --synthetic-path=data/eedi-synthetic \
#     --use-synthetic \
#     --model=Qwen/Qwen2.5-1.5B-Instruct \
#     --token-pool=last \
#     --per-device-bs=8 \
#     --lr=2e-4 \
#     --warmup-ratio=0.2 \
#     --n-epochs=5 \
#     --lora-rank=16 \
#     --dataset-seed=1 \
#     --iterative-hnm \
#     --run-name=synthetic-qwen-math-instruct
# ./scripts/telegram.sh "synthetic-qwen-math-instruct done"

# # 4
# accelerate launch eedi/train_sts_hn.py \
#     --synthetic-path=data/eedi-synthetic \
#     --use-synthetic \
#     --model=Qwen/Qwen2.5-1.5B-Instruct \
#     --token-pool=last \
#     --per-device-bs=8 \
#     --lr=2e-4 \
#     --warmup-ratio=0.2 \
#     --n-epochs=5 \
#     --lora-rank=16 \
#     --dataset-seed=1 \
#     --iterative-hnm \
#     --run-name=synthetic-qwen-math
# ./scripts/telegram.sh "synthetic-qwen-math done"
