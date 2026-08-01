#!/bin/bash

# $1 表示模型路径，$2表示用多少并发，$3表示每个模型需要几张卡。所以$2=可用的GPU数/$3.【小于13b的模型，$3一般为1.】
# $4表示log存放路径, $5表示评测的数据集，用逗号隔开 $6表示gen/ppl
# $7表示fewshot大小。如果小于0，表示不穿fewshot，采用任务本身的数值。如果大于等于0，则表示重新赋值。
# $8-指定的一组GPU序号，以逗号分隔，如果为空的话默认运行所有GPU，示例"2,4,5" 即选用序号为2、4、5的三张GPU进行评测
# $9-本地部署的端口号，默认5002
# $10-部署方式，默认为"vllm"，可选"vllm"和"transformers"

echo "打印每个传递的参数："
for arg in "$@"; do
    echo "$arg"
done

# 使用nvidia-smi命令获取GPU数量
gpu_count=$(nvidia-smi -L | wc -l)
default_gpus=$(seq -s ',' 0 $((gpu_count-1)) | sed 's/,$//')
echo "Available GPUs: $gpu_count: $default_gpus"

# hyperparameters
TASK_NAME=$5 # arc-e,arc-c,boolq,hellaswag,piqa,winogrande  # 需要评测的任务，多个用,隔开
HF_MODEL_NAME=$1 #  # huggingface上的模型名
URL="http://127.0.0.1:5002/infer"  # 这里是固定的
NUMBER_OF_THREAD=$2  # 线程数，一般设为 gpu数/per-proc-gpus
CONFIG_PATH=configs/eval_config.json  # 评测文件路径
OUTPUT_BASE_PATH=$4 # /local/logs/test  # 结果保存路径，与HF_MODEL_NAME一致
CUDA_VISIBLE_DEVICES=${8:-$default_gpus} # 指定的一组GPU序号，以逗号分隔，如果为空的话默认运行所有GPU，示例"2,4,5" 即选用序号为2、4、5的三张GPU进行评测
PORT=${9:-5002} # 端口号，默认5002
INFER_TYPE=${10:-vLLM} # 部署方式，默认为"vllm"，可选"vllm"和"transformers"

# 步骤1
# 选择评测的任务，生成评测 config文件。其中method=gen，表示生成式
python configs/make_config.py --datasets $TASK_NAME --method $6 # gen


# 步骤2
# 启动 gunicorn 并保存 PID
bash URLs/start_gunicorn.sh --hf-model-name $HF_MODEL_NAME --per-proc-gpus $3 --cuda-visible-devices $CUDA_VISIBLE_DEVICES --port $PORT --infer-type $INFER_TYPE& 
echo $! > gunicorn.pid


# 步骤3
# 检查服务是否已启动
MAX_RETRIES=60  # 最大尝试次数，相当于等待30分钟
COUNTER=0

while [ $COUNTER -lt $MAX_RETRIES ]; do
    sleep 30
    curl -s $URL > /dev/null
    if [ $? -eq 0 ]; then
        echo "Service is up!"
        break
    fi
    COUNTER=$((COUNTER+1))
    if [ $COUNTER -eq $MAX_RETRIES ]; then
        echo "Service did not start in time. Exiting."
        exit 1
    fi
done


if [ "$6" = "gen" ]; then
    postprocess="general_torch"
    params="models/model_params/vllm_sample.json"
elif [ "$6" = "ppl" ]; then
    postprocess="general_torch_ppl"
    params="models/model_params/vllm_logprobs.json"
else
    echo "Invalid argument for \$6: $6. Expected 'gen' or 'ppl'."
    exit 1
fi

# 下面你可以使用 $postprocess 和 $params 变量
echo "Postprocessing method: $postprocess"
echo "Params file: $params"


# 步骤4

# 构建命令选项
OPTS="--model general"
OPTS+=" --model_args url=$URL,concurrency=$NUMBER_OF_THREAD"
OPTS+=" --config_path $CONFIG_PATH"
OPTS+=" --output_base_path $OUTPUT_BASE_PATH"
OPTS+=" --batch_size 32"
OPTS+=" --postprocess $postprocess"
OPTS+=" --params $params"
OPTS+=" --write_out"
# OPTS+=" --limit 2" # 你可以取消注释这行来包含这个选项



fewshot="$7"

# 检查参数是否是数字（可选）
if ! [ "$fewshot" -eq "$fewshot" ] 2> /dev/null; then
    echo "The provided argument is not an integer."
    exit 1
fi

# 检查参数是否小于0
if [ "$fewshot" -lt 0 ]; then
    # 在这里放入参数小于0的处理逻辑
    # 构建完整命令
    CMD="python main.py ${OPTS}"
else
    # 在这里放入参数大于等于0的处理逻辑
    CMD="python main.py ${OPTS} --num_fewshot ${fewshot}"
fi


# 打印命令
echo "-------final CMD is------"
echo "${CMD}"
echo "-------final CMD end------"

# 执行命令
eval $CMD


# 步骤5
# del gunicorn.pid file
rm gunicorn.pid
# 结束 gunicorn 进程及其 worker 进程
pkill gunicorn; sleep 5; kill -SIGINT $$