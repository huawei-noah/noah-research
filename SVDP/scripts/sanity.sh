export CUDA_VISIBLE_DEVICES=0
export IMPLEMENTATION_NAME=sparse_mistral_baseline
export MODEL_PATH=../models/Tiiny/Bamboo-DPO-v0_1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

export PREDICTORS_PATH=weights/sparse_mistral/r352_s05
python3 utils/sanity.py \
    --model_path ${MODEL_PATH} \
    --vllm_module_path vllm_implementations/${IMPLEMENTATION_NAME}.py \
    --start_prompt "hello, what year is it today?" \
    --temperature 0.0