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
    --paraphrased-path=data/eedi-paraphrased \
    --model=Alibaba-NLP/gte-large-en-v1.5 \
    --token-pool=last \
    --per-device-bs=32 \
    --lr=1e-4 \
    --n-epochs=3 \
    --lora-rank=16 \
    --run-name=gte-lambda-deepspeed-tpool-last

# If training succeeds, send success notification
notify_success
