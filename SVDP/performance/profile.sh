#!/bin/bash
export MODEL="../models/SparseLLM/prosparse-llama-2-7b"
export IMPL="vllm_implementations/prosparse_llama_optimized.py"
CUDA_VISIBLE_DEVICES=4 python3 performance/profiling.py \
        --model ${MODEL} \
        --json llama_optimized_profiling.json \
        --vllm_model_modulename ${IMPL} \
        --num_prompts 1 \
        --dataset-path performance/data/arc-c.json \
        --prompt-len 1000 \
        --generation_length 4 \
        --trust_remote_code \
        --dtype half \
        --enforce-eager \
        --max_model_len 1201 \
        --gpu_memory_utilization 0.8 

# python3 performance/print_layerwise_table.py --json-trace llama_optimized_profiling.json --phase decode_1 --table summary

python3 performance/visualize_layerwise_profile.py --json-trace llama_optimized_profiling.json --output-directory . --level module --plot-metric pct_cuda_time
