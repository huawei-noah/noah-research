#!/bin/bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    train_gen_rank.py \
    --train_file data/math23k/infix_math23k_processed.train \
    --valid_file data/math23k/infix_math23k_processed.valid \
    --test_file data/math23k/infix_math23k_processed.test \
    --output_dir output/ranker \
    --model_path output/generator/saved_model \
    --max_source_length 200 \
    --max_target_length 128 \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --logging_steps 100 \
    --per_gpu_train_batch_size 16 \
    --src_lang zh_CN \
    --tgt_lang en_XX \
    --test_per_epoch 1 \
    --regenerate 1 \
    --rule_negatives 1 \
    --num_negatives 10


