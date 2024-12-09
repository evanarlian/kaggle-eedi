#!/usr/bin/env bash
set -Eeuxo pipefail

notify_success() {
    ./scripts/telegram.sh "training embedding done"
}

notify_failure() {
    ./scripts/telegram.sh "training failed"
}

# Trap errors and call the failure notification
trap notify_failure ERR

# Main script logic
accelerate launch eedi/train_sts_hn.py \
    --synthetic-path=data/eedi-synthetic \
    --use-synthetic \
    --model=sentence-transformers/all-MiniLM-L6-v2 \
    --token-pool=last \
    --per-device-bs=8 \
    --lr=2e-4 \
    --n-epochs=5 \
    --lora-rank=16 \
    --dataset-seed=1 \
    --run-name=synthetic-test

# If training succeeds, send success notification
notify_success
