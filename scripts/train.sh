#!/usr/bin/env bash
set -Eeuxo pipefail

# export WANDB_DISABLED='true'

torchrun --nproc-per-node=1 eedi/train_sts_hn.py \
    --paraphrased-path=data/eedi-paraphrased \
    --model=sentence-transformers/all-MiniLM-L6-v2 \
    --per-device-bs=32 \
    --lr=2e-5 \
    --n-epochs=2 \
    --lora-rank=8 \
    --run-name=testing_multigpu

# python -m eedi.train_sts_hn \
#     --paraphrased-path=data/eedi-paraphrased \
#     --model=Alibaba-NLP/gte-large-en-v1.5 \
#     --per-device-bs=1 \
#     --n-epochs=10 \
#     --lora-rank=8 \
#     --run-name=gte-large
