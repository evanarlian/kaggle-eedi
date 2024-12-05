#!/usr/bin/env bash
set -Eeuxo pipefail

python eedi/infer_top25.py \
    --dataset-dir=data \
    --model-path=models/model_orig \
    --lora-path=models/model_lora \
    --token-pool=first
