#!/bin/bash

model_type=$2


if [ $1 == "m_atis_english" ]
then
    python main.py --task m_atis \
                   --train_languages en \
                   --dev_languages en \
                   --test_languages en \
                   --model_dir m_atis_english \
                   --do_train \
                   --do_eval \
                   --cuda_device cuda:0 \
                   --train_batch_size 2 \
                   --eval_batch_size 2 \
                   --gradient_accumulation_steps 5 \
                   --num_train_epochs 10 \
                   --learning_rate 0.00002 \
                   --save_model \
                   --model_type $2 \
                   --max_seq_len 100
fi

if [ $1 == "m_atis_aligned" ]
then
    for lang in de es tr zh hi fr ja pt
    do
        python main.py --task m_atis \
                       --train_languages en \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir m_atis_aligned_$lang \
                       --do_train \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --train_batch_size 2 \
                       --eval_batch_size 2 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 10 \
                       --learning_rate 0.00002 \
                       --align_languages $lang \
                       --save_model \
                       --model_type $2 \
                       --max_seq_len 100
    done
fi

if [ $1 == "m_atis_zero_shot" ]
then
    for lang in de es tr zh hi fr ja pt
    do
        python main.py --task m_atis \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir m_atis_zero_shot_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 10 \
                       --model_type $2 \
                       --load_eval_model m_atis_english \
                       --max_seq_len 100
    done
fi

if [ $1 == "m_atis_target" ]
then
    for lang in de es tr zh hi fr ja pt
    do
        python main.py --task m_atis \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir m_atis_target_$lang \
                       --do_eval \
                       --do_train \
                       --cuda_device cuda:0 \
                       --train_batch_size 2 \
                       --eval_batch_size 2 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 10 \
                       --learning_rate 0.00002 \
                       --model_type $2 \
                       --max_seq_len 100
    done
fi

if [ $1 == "m_atis_eval" ]
then
    for lang in de es tr zh hi fr ja pt
    do
        python main.py --task m_atis \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir m_atis_eval_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 10 \
                       --load_eval_model m_atis_aligned \
                       --model_type $2 \
                       --max_seq_len 100
    done
fi
