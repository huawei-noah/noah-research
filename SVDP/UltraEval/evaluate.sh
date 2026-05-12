#!/bin/bash
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


SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # script working directory

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

mkdir ../outputs/quality/${OUTPUT_TAG}
python main.py \
    --model general \
    --model_args url=$URL,concurrency=1 \
    --config_path configs/eval_config.json \
    --output_base_path ../outputs/quality/${OUTPUT_TAG} \
    --batch_size 1 \
    --postprocess general_torch \
    --write_out

echo "Killing $LLM_SERVE_PID"
kill -9 $LLM_SERVE_PID

echo "Done"