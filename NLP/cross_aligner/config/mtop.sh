#!/bin/bash

if [ $1 == "mtop_english" ]
then
    python main.py --task mtop \
                   --train_languages en \
                   --dev_languages en \
                   --test_languages en \
                   --model_dir mtop_english \
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
                   --max_seq_len 70
fi

if [ $1 == "mtop_aligned" ]
then
    for lang in de es fr th hi
    do
        python main.py --task mtop \
                       --train_languages en \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtop_aligned_$lang \
                       --do_train \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --train_batch_size 10 \
                       --eval_batch_size 2 \
                       --gradient_accumulation_steps 1 \
                       --num_train_epochs 10 \
                       --learning_rate 0.00002 \
                       --align_languages $lang \
                       --save_model \
                       --model_type $2 \
                       --max_seq_len 70 \
                       --use_aux_losses CA XA \
                       --use_weighting COV
    done
fi

if [ $1 == "mtop_zero_shot" ]
then
    for lang in de es fr th hi
    do
        python main.py --task mtop \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtop_zero_shot_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 10 \
                       --model_type $2 \
                       --load_eval_model mtop_english \
                       --max_seq_len 70
    done
fi

if [ $1 == "mtop_target" ]
then
    for lang in de es fr th hi
    do
        python main.py --task mtop \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtop_target_$lang \
                       --do_eval \
                       --do_train \
                       --cuda_device cuda:0 \
                       --train_batch_size 2 \
                       --eval_batch_size 2 \
                       --gradient_accumulation_steps 5 \
                       --num_train_epochs 10 \
                       --learning_rate 0.00002 \
                       --model_type $2 \
                       --save_model \
                       --max_seq_len 70
    done
fi

if [ $1 == "mtop_eval" ]
then
    for lang in de es fr th hi
    do
        python main.py --task mtop \
                       --train_languages $lang \
                       --dev_languages $lang \
                       --test_languages $lang \
                       --model_dir mtop_eval_$lang \
                       --do_eval \
                       --cuda_device cuda:0 \
                       --eval_batch_size 10 \
                       --load_eval_model mtop_aligned_$lang \
                       --model_type $2 \
                       --max_seq_len 70 \
                       --debug
    done
fi
