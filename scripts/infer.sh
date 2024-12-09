#!/usr/bin/env bash
set -Eeuxo pipefail

echo "RETRIEVE..."
python eedi/infer_top25.py \
    --dataset-dir=data \
    --model-paths models/model_orig models/model_orig \
    --lora-paths models/model_lora models/model_lora \
    --token-pools first last

# this rerank code is only designed for kaggle uses
echo "RERANK..."
python eedi/vllm_infer.py \
    --dataset-dir=data \
    --top25-path=top25_miscons.json \
    --model-path=Qwen/Qwen2.5-1.5B-Instruct
