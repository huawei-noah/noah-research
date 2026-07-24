# shape=4096*4096
# typer='v'
# layer=4
# vector_len=4
# python3 recon_llm.py --dataset=llm --ckptpath=/home/ma-user/work/tianye/PocketLLM/llm_codebook/modify_results/${typer}_${layer}/checkpoints/model_best.pth --data-dir=/home/ma-user/work/tianye/PocketLLM/llama2_7B_npy_v$vector_len/ --hidden=16 --k=32768 --batch-size=16384 --rec_lenth=4096 --weight_layer=$layer --weight_type=$typer --inputer=$vector_len

# shape=4096*16384
typer='up' #'gate'
layer=10 #10
vector_len=4
python3 recon_llm.py --dataset=llm --ckptpath=/home/ma-user/work/tianye/PocketLLM/llm_codebook/modify_results/${typer}_${layer}/checkpoints/model_best.pth --data-dir=/home/ma-user/work/tianye/PocketLLM/llama3_7B_npy_v$vector_len/ --hidden=16 --k=32768 --batch-size=16384 --rec_lenth=4096 --weight_layer=$layer --weight_type=$typer --inputer=$vector_len