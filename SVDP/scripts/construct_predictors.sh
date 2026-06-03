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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 utils/construct_predictors.py \
    --model_path ../models/Tiiny/SparseQwen2-7B \
    --calibration_prompts_path calibration_prompts.json \
    --predictors_output_path predictors/weights/sparse_qwen2/r512_s05 \
    --rank 512 \
    --s 0.5 \
    --torch_dtype float16 \
    --device_map cuda:0 \
    --ablation_config_whitening cholesky \
    --ablation_config_bias full \
    --ablation_config_penalty full \
    --sparsity_plot_output_file outputs/sparsity.pdf
