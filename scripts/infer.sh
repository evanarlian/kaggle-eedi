#!/usr/bin/env bash
set -Eeuxo pipefail

echo "RETRIEVE..."
python eedi/infer_top25.py \
    --dataset-dir=data \
    --model-path=models/model_orig \
    --lora-path=models/model_lora \
    --token-pool=first

# this rerank code is only designed for kaggle uses
echo "RERANK..."
python eedi/vllm_infer.py \
    --dataset-dir=data \
    --top25-path=top25_miscons.json \
    --model-path=Qwen/Qwen2.5-1.5B-Instruct
