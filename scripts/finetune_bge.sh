#!/usr/bin/env bash

python bge_data_prep.py --dataset-dir=data --output-dir=data

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --input_file data/bge_dataset/train.jsonl \
    --output_file data/bge_dataset/train_mined_hn_stage1.jsonl \
    --range_for_sampling 1-25 \
    --negative_number 15 \
    --use_gpu_for_searching 

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir models/bge_large_en_ft_stage1 \
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --train_data data/bge_dataset/train_mined_hn_stage1.jsonl \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 150 \
    --passage_max_len 35 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 1000 \
    --query_instruction_for_retrieval ""

python -m FlagEmbedding.baai_general_embedding.finetune.hn_mine \
    --model_name_or_path models/bge_large_en_ft_stage1 \
    --input_file data/bge_dataset/train.jsonl \
    --output_file data/bge_dataset/train_mined_hn_stage2.jsonl \
    --range_for_sampling 1-25 \
    --negative_number 15 \
    --use_gpu_for_searching 

torchrun --nproc_per_node 1 \
    -m FlagEmbedding.baai_general_embedding.finetune.run \
    --output_dir models/bge_large_en_ft_stage2 \
    --model_name_or_path models/bge_large_en_ft_stage1 \
    --train_data data/bge_dataset/train_mined_hn_stage2.jsonl \
    --learning_rate 1e-5 \
    --fp16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --dataloader_drop_last True \
    --normlized True \
    --temperature 0.02 \
    --query_max_len 150 \
    --passage_max_len 35 \
    --train_group_size 2 \
    --negatives_cross_device \
    --logging_steps 10 \
    --save_steps 1000 \
    --query_instruction_for_retrieval "" 
