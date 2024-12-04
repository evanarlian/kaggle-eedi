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
    --model=Salesforce/SFR-Embedding-2_R \
    --token-pool=last \
    --per-device-bs=8 \
    --lr=2e-4 \
    --n-epochs=5 \
    --lora-rank=16 \
    --run-name=sfr-lambda-8-a100

# If training succeeds, send success notification
notify_success
