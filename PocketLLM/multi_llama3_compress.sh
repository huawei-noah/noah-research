#!/bin/bash
vector_len=4
llm_typer="llama3"
model_type=("up") #"q" "k" "v" "o" "gate" "up" "down" 
model_layer=(3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23  24 25 26 27 28 29 )  #3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23  24 25 26 27 28 29

for typer in "${model_type[@]}";do
    for layer in "${model_layer[@]}";do
        echo "压缩第$layer层$typer的权重"
        python3 llm_compress.py --dataset=llm --data-dir=/home/ma-user/work/tianye/PocketLLM/${llm_typer}_7B_npy_v$vector_len/ --epochs=50 --hidden=16 --k=32768 --lr=1.6e-2 --batch-size=16384 --weight_layer=$layer --weight_type=$typer --inputer=$vector_len
    done
done
