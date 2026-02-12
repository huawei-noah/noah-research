llm_typer="llama3"
vector_len=4
python3 llm_compress.py --dataset=llm --data-dir=/home/ma-user/work/tianye/PocketLLM/${llm_typer}_7B_npy_v$vector_len/ --epochs=50 --hidden=16 --k=32768 --lr=1.6e-2 --batch-size=16384 --weight_layer=10 --weight_type='up' --inputer=$vector_len