#!/bin/bash

model_type=$2

if [ $1 == "paws_x_english" ]
then
    python main.py --task paws_x \
                   --train_languages en \
                   --dev_languages en \
                   --test_languages en \
                   --model_dir paws_x_english \
                   --do_train \
                   --do_eval \
                   --cuda_device cuda:0 \
                   --train_batch_size 4 \
                   --eval_batch_size 4 \
                   --num_train_epochs 10 \
                   --learning_rate 0.000006 \
                   --model_type $2 \
                   --save_model \
                   --max_seq_len 200
fi

if [ $1 == "paws_x_aligned" ]
then
    for lang in de es fr ja ko zh
    do
        python main.py --task paws_x \
                       --train_languages en \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir paws_x_aligned_$lang \
                       --do_train \
                       --do_eval \
                       --save_model \
                       --cuda_device cuda:0 \
                       --train_batch_size 1 \
                       --eval_batch_size 1 \
                       --gradient_accumulation_steps 4 \
                       --num_train_epochs 10 \
                       --align_languages $lang \
                       --learning_rate 0.000006 \
                       --model_type $2 \
                       --max_seq_len 200
    done
fi

if [ $1 == "paws_x_zero_shot" ]
then
    for lang in de es fr ja ko zh
    do
        python main.py --task paws_x \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir paws_x_zero_shot_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 4 \
                       --load_eval_model paws_x_english \
                       --model_type $2 \
                       --max_seq_len 200
    done
fi

if [ $1 == "paws_x_target" ]
then
    for lang in de es fr ja ko zh
    do
        python main.py --task paws_x \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir paws_x_target_$lang \
                       --do_train \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --train_batch_size 4 \
                       --eval_batch_size 4 \
                       --num_train_epochs 10 \
                       --learning_rate 0.000006 \
                       --model_type $2 \
                       --max_seq_len 200
    done
fi

if [ $1 == "paws_x_eval" ]
then
    for lang in en de es fr ja ko zh
    do
        python main.py --task paws_x \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir paws_x_eval_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 4 \
                       --load_eval_model paws_x_MODEL_NAME \
                       --model_type $2 \
                       --max_seq_len 200
    done
fi
