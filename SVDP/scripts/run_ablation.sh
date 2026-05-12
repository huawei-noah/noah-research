#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export DEPLOY_PORT=5070
export BENCHMARKS=triviaqa,arc-e,arc-c,humaneval,mbpp,gsm8k,bbh,cmmlu

export RANK=256
export MODEL_PATH=../models/SparseLLM/prosparse-llama-2-7b
export MODEL_NAME=sparse_llama
python3 ./utils/run_ablation.py

export RANK=352
export MODEL_PATH=../models/Tiiny/TurboSparse-Mistral-Instruct
export MODEL_NAME=sparse_mistral
# WARNING: make sure that is undone
python3 utils/update_postprocess.py humaneval_chatgpt
python3 ./utils/run_ablation.py
python3 utils/update_postprocess.py humaneval_post

export RANK=512
export MODEL_PATH=../models/Tiiny/SparseQwen2-7B
export MODEL_NAME=sparse_qwen2
python3 ./utils/run_ablation.py