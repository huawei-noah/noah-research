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


echo "<STEP> Evaluating quality for ${MODEL_PATH} with ${MODEL_NAME}_baseline implementation on ${BENCHMARKS} at port ${DEPLOY_PORT}!"
export TAG_NAME=${MODEL_NAME}_baseline
echo "Will write results under outputs/quality/${TAG_NAME}"
./UltraEval/evaluate.sh vllm_implementations/${MODEL_NAME}_baseline.py ${MODEL_PATH} ${BENCHMARKS} ${DEPLOY_PORT} ${TAG_NAME}

for s in 0.4 0.5 0.6 0.7
do
    echo "<STEP> Constructing predictors with rank ${RANK} and s ${s}"
    export PREDICTORS_PATH=weights/main_results/${MODEL_NAME}/r${RANK}_s${s}
    export TAG_NAME=${MODEL_NAME}_svd_predictors_r${RANK}_s${s}
    python3 utils/construct_predictors.py \
        --model_path $MODEL_PATH \
        --predictors_output_path $PREDICTORS_PATH \
        --rank $RANK \
        --s $s \
        --sparsity_plot_output_file outputs/tmp.pdf \
        --calibration_prompts_path calibration_prompts.json \
        --torch_dtype float16 \
        --device_map cuda:0

    echo "<STEP> Evaluating quality for ${MODEL_PATH} with ${MODEL_NAME}_svd_predictors implementation on ${BENCHMARKS} at port ${DEPLOY_PORT}!"
    echo "Will write results under outputs/quality/${TAG_NAME}"
    ./UltraEval/evaluate.sh vllm_implementations/${MODEL_NAME}_svd_predictors.py ${MODEL_PATH} ${BENCHMARKS} ${DEPLOY_PORT} ${TAG_NAME}
done