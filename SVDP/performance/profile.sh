#!/bin/bash
# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Run from the repository root so the relative paths below resolve regardless
# of the directory this script is launched from.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

export MODEL="../models/SparseLLM/prosparse-llama-2-7b"
export IMPL="vllm_implementations/sparse_llama_svd_predictors.py"
CUDA_VISIBLE_DEVICES=4 python3 utils/profiling.py \
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
