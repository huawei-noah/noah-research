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

export MODULE_PATH=$1
export MODEL_NAME=$2
export BENCHMARKS=$3
export DEPLOY_PORT=$4
export OUTPUT_TAG=$5

export URL=http://127.0.0.1:${DEPLOY_PORT}/infer
export VLLM_ALLOW_DEPRECATED_BEAM_SEARCH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_USE_V1=0


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # .../SVDP/UltraEval
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"                # .../SVDP
OUTPUT_DIR="${REPO_ROOT}/outputs/quality/${OUTPUT_TAG}"

# Run from the repo root so the relative paths below resolve regardless of the
# directory the script was invoked from.
cd "${REPO_ROOT}"

echo "PID $$ will make requests and wait for responses"
echo "Setting benchmarks"
cd UltraEval
python3 configs/make_config.py --datasets ${BENCHMARKS} --method gen

cd ..
echo "Deploying model"
python3 ${SCRIPT_DIR}/URLs/vllm_url.py \
    --model_name ${MODEL_NAME} \
    --port ${DEPLOY_PORT} \
    --use_chat_template \
    --module_path ${MODULE_PATH} > server-port${DEPLOY_PORT}.log 2>&1 &

LLM_SERVE_PID=$!
echo "LLM is running by PID $LLM_SERVE_PID"

cd UltraEval
echo "Evaluating the model"

mkdir -p "${OUTPUT_DIR}"
python main.py \
    --model general \
    --model_args url=$URL,concurrency=1 \
    --config_path configs/eval_config.json \
    --output_base_path "${OUTPUT_DIR}" \
    --batch_size 1 \
    --postprocess general_torch \
    --write_out

echo "Killing $LLM_SERVE_PID"
kill -9 $LLM_SERVE_PID

echo "Done"