#!/bin/bash
vector_len=4 #4
model_type=("o") 
model_layer=( 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29) #

# 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
for typer in "${model_type[@]}";do
    for layer in "${model_layer[@]}";do
        echo "恢复第$layer层$typer的权重"
        python3 recon_llm.py --dataset=llm --ckptpath=/home/ma-user/work/tianye/PocketLLM/llm_codebook/modify_results/${typer}_${layer}/checkpoints/model_best.pth --data-dir=/home/ma-user/work/tianye/PocketLLM/llama3_7B_npy_v$vector_len/ --hidden=16 --k=32768 --batch-size=16384 --rec_lenth=4096 --weight_layer=$layer --weight_type=$typer --inputer=$vector_len
    done
done