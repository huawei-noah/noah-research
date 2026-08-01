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

export CUDA_VISIBLE_DEVICES=0
export IMPLEMENTATION_NAME=sparse_mistral_baseline
# Base directory holding the downloaded model weights. Override to point at
# your local model store, e.g. MODELS_DIR=/data/models ./scripts/sanity.sh
MODELS_DIR=${MODELS_DIR:-../models}
export MODEL_PATH=${MODELS_DIR}/Tiiny/Bamboo-DPO-v0_1
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

export PREDICTORS_PATH=weights/sparse_mistral/r352_s05
python3 utils/sanity.py \
    --model_path ${MODEL_PATH} \
    --vllm_module_path vllm_implementations/${IMPLEMENTATION_NAME}.py \
    --start_prompt "hello, what year is it today?" \
    --temperature 0.0