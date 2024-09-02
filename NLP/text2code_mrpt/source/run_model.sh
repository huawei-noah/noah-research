# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=2
CUDA_LAUNCH_BLOCKING=1

export HF_DATASETS_CACHE="path_to_cache/"
CACHE_DIR="path_to_cache/"

OUTPUT_DIR="mrpt_trained_models_${SIZE}/"
TOTAL_EPOCHS=10
steps='1M'
LR=1e-5
MIN_LR=1e-6
CLIP_GRAD=1.0

BS=32
ACC=1

EVAL_BS=32
EVAL_STEPS=5000
LOG_STEPS=1000
SAVE_STEPS=50000
MAX_LEN=1024
WARMUP=0.01
DECAY=0.01


save_dir_name="pangu_CodeCLM_full_sep"

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
	lm_trainer.py \
	--num_of_gpus="${GPUS_PER_NODE}" \
	--num_of_nodes="${NNODES}" \
	--do_train=true \
	--gradient_checkpointing=true \
	--fp16=true \
	--overwrite_output_dir=True \
	--preprocessing_num_workers=64 \
	--model_name_or_path="path_to_model/" \
	--dataset_name="path_to_data/" \
	--output_dir="${OUTPUT_DIR}${save_dir_name}" \
	--per_device_train_batch_size=${BS} \
	--per_device_eval_batch_size=${EVAL_BS} \
	--eval_accumulation_steps=10 \
	--optim="adamw_torch" \
	--learning_rate=${LR} \
	--min_learning_rate=${MIN_LR} \
	--weight_decay=${DECAY} \
	--num_train_epochs=${TOTAL_EPOCHS} \
	--gradient_accumulation_steps=${ACC} \
	--warmup_ratio=${WARMUP} \
	--max_grad_norm=${CLIP_GRAD} \
	--seed=1234 \
	--max_seq_length=${MAX_LEN} \
	--save_total_limit=1 \
	--eval_steps=${EVAL_STEPS} \
	--logging_steps=${LOG_STEPS} \
	--save_steps=${SAVE_STEPS} \
	--evaluation_strategy="steps" \
	--logging_strategy="steps" \
	--cache_dir="${ROOT_DIR}${CACHE_DIR}" \
	--overwrite_output_dir=true \
	--disable_tqdm=true \
	--remove_unused_columns=false \
	--validation_percentage=0.001 \
	--report_to="tensorboard" \
	--deepspeed="ds_config_s2_no_offload.json"