#!/usr/bin/env bash
set -Eeuxo pipefail

# export WANDB_DISABLED='true'

torchrun --nproc-per-node=2 eedi/train_sts_hn.py \
    --paraphrased-path=data/eedi-paraphrased \
    --model=Alibaba-NLP/gte-large-en-v1.5 \
    --per-device-bs=32 \
    --lr=1e-4 \
    --n-epochs=5 \
    --lora-rank=16 \
    --run-name=gte-large-proper

./scripts/telegram.sh "training embedding done"
