#!/usr/bin/env bash
set -Eeuxo pipefail

python train_sts_hn.py \
    --paraphrased-path=data/eedi-paraphrased \
    --model=sentence-transformers/all-MiniLM-L6-v2 \
    --per-device-bs=64 \
    --n-epochs=1 \
    --run-name=all-minilm-eedi-simple
