#!/bin/bash

model_type=$2


if [ $1 == "mtod_english" ]
then
    python main.py --task mtod \
                   --train_languages en \
                   --dev_languages en \
                   --test_languages en \
                   --model_dir mtod_english \
                   --do_train \
                   --do_eval \
                   --cuda_device cuda:0 \
                   --train_batch_size 2 \
                   --eval_batch_size 2 \
                   --gradient_accumulation_steps 5 \
                   --num_train_epochs 10 \
                   --learning_rate 0.000005 \
                   --save_model \
                   --model_type $2 \
                   --max_seq_len 50
fi

if [ $1 == "mtod_aligned" ]
then
    for lang in es th
    do
        python main.py --task mtod \
                       --train_languages en \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtod_aligned_$lang \
                       --do_train \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --train_batch_size 2 \
                       --eval_batch_size 2 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 5 \
                       --learning_rate 0.000005 \
                       --align_languages $lang \
                       --save_model \
                       --model_type $2 \
                       --max_seq_len 50
    done
fi

if [ $1 == "mtod_zero_shot" ]
then
    for lang in es th
    do
        python main.py --task mtod \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtod_zero_shot_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 10 \
                       --model_type $2 \
                       --load_eval_model mtod_english \
                       --max_seq_len 50
    done
fi

if [ $1 == "mtod_target" ]
then
    for lang in es th
    do
        python main.py --task mtod \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtod_target_$lang \
                       --do_eval \
                       --do_train \
                       --cuda_device cuda:0 \
                       --train_batch_size 2 \
                       --eval_batch_size 2 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 10 \
                       --learning_rate 0.000005 \
                       --model_type $2 \
                       --max_seq_len 50
    done
fi

if [ $1 == "mtod_eval" ]
then
    for lang in es th
    do
        python main.py --task mtod \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtod_eval_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 10 \
                       --load_eval_model mtod_aligned \
                       --model_type $2 \
                       --max_seq_len 70
    done
fi
