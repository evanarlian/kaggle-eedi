#!/usr/bin/env bash
set -Eeuxo pipefail

python -m eedi.train_sts_hn \
    --paraphrased-path=data/eedi-paraphrased \
    --model=sentence-transformers/all-MiniLM-L6-v2 \
    --per-device-bs=64 \
    --n-epochs=10 \
    --lora-rank=8 \
    --run-name=all-minilm-eedi-simple
