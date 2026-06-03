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

export CUDA_VISIBLE_DEVICES=1
export DEPLOY_PORT=5088
export BENCHMARKS=triviaqa,arc-e,arc-c,humaneval,mbpp,gsm8k,bbh,cmmlu

# Base directory holding the downloaded model weights. Override to point at
# your local model store, e.g. MODELS_DIR=/data/models ./scripts/run_main_results.sh
MODELS_DIR=${MODELS_DIR:-../models}

export RANK=256
export MODEL_PATH=${MODELS_DIR}/SparseLLM/prosparse-llama-2-7b
export MODEL_NAME=sparse_llama
./scripts/main_quality.sh


export RANK=352
export MODEL_PATH=${MODELS_DIR}/Tiiny/TurboSparse-Mistral-Instruct
export MODEL_NAME=sparse_mistral
# WARNING: make sure that is undone
python3 utils/update_postprocess.py humaneval_chatgpt
./scripts/main_quality.sh
python3 utils/update_postprocess.py humaneval_post

export RANK=512
export MODEL_PATH=${MODELS_DIR}/Tiiny/SparseQwen2-7B
export MODEL_NAME=sparse_qwen2
./scripts/main_quality.sh