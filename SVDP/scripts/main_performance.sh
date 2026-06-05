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

export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0

echo "<STEP> Evaluating performance for ${MODEL_PATH} with ${MODEL_NAME}_baseline implementation!"
for DATASET in mbpp gsm8k arc-e arc-c humaneval triviaqa bbh cmmlu
do
    python3 utils/profiling.py \
        --model ${MODEL_PATH} \
        --metrics outputs/performance/${MODEL_NAME}/baseline/${DATASET}.json \
        --vllm_model_modulename vllm_implementations/${MODEL_NAME}_baseline.py \
        --num_prompts 5 \
        --dataset-path performance/data/${DATASET}.json \
        --prompt-len 1000 \
        --generation_length 200 \
        --trust_remote_code \
        --dtype half \
        --enforce-eager \
        --max_model_len 1201 \
        --gpu_memory_utilization 0.9
done

for s in 0.4 0.5 0.6 0.7
do 
    echo "<STEP> Evaluating performance for ${MODEL_PATH} with ${MODEL_NAME}_svd_predictors implementation with s ${s} !"
    export PREDICTORS_PATH=weights/main_results/${MODEL_NAME}/r${RANK}_s${s}
    export TAG_NAME=r${RANK}_s${s}
    for DATASET in mbpp gsm8k arc-e arc-c humaneval triviaqa bbh cmmlu
    do
        python3 utils/profiling.py \
            --model ${MODEL_PATH} \
            --metrics outputs/performance/${MODEL_NAME}/${TAG_NAME}/${DATASET}.json \
            --vllm_model_modulename vllm_implementations/${MODEL_NAME}_svd_predictors.py \
            --num_prompts 5 \
            --dataset-path performance/data/${DATASET}.json \
            --prompt-len 1000 \
            --generation_length 200 \
            --trust_remote_code \
            --dtype half \
            --enforce-eager \
            --max_model_len 1201 \
            --gpu_memory_utilization 0.9
    done

    # Calculate average speedup
    echo "<STEP> Calculate average speedup for ${MODEL_PATH} with ${MODEL_NAME}_svd_predictors implementation with s ${s} !"
    suma=0
    counter=0
    for DATASET in humaneval arc-c arc-e bbh cmmlu gsm8k triviaqa mbpp
    do
        e2e_time_baseline=$(python3 utils/print_e2e_time.py outputs/performance/${MODEL_NAME}/baseline/${DATASET}.json)
        e2e_time_optimized=$(python3 utils/print_e2e_time.py outputs/performance/${MODEL_NAME}/${TAG_NAME}/${DATASET}.json)
        e2e_speedup=$(echo "scale=4; $e2e_time_baseline / $e2e_time_optimized" | bc) # P.S. no rounding here
        echo "Speedup on ${DATASET} =====> ${e2e_speedup}"
        suma=$(echo "$suma + $e2e_speedup" | bc)
        counter=$(echo "$counter + 1" | bc)
    done

    average=$(echo "scale=4; $suma / $counter" | bc)
    echo "<STEP> AVERAGE SPEEDUP for ${MODEL_PATH} with ${MODEL_NAME}_svd_predictors implementation with s ${s} and rank ${RANK} =====> $average"
done

echo "Done!"