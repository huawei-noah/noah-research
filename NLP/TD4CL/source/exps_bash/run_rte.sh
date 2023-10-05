#!/bin/bash

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

cd ../

main_dir="../trained_models"

DATASET='rte'
SETTING=$1
MODEL=${2:-"roberta-base"}
SEED=${3:-123}
LR=${4:-"1e-5"}
EPOCHS=${5:-5}
DYNAMICS=$6
METRIC1=$7
METRIC2=$8

MAX_SEQ_LEN=128
EPSILON=1e-8
INTERVAL=100
WARMUP=0.0

ACC=1
BS=16
LR=2e-5
EPOCHS=5


if [[ ${SETTING} = 'baseline' ]]; then

	model_dir_name="${DATASET}_${MODEL}_${SEED}_LR${LR}_LEN${MAX_SEQ_LEN}_BS${BS}_E${EPOCHS}"

	python main_seq_class.py --dataset_name "${DATASET}" \
							 --mode "train" \
							 --max_seq_length ${MAX_SEQ_LEN} \
							 --batch_size ${BS} \
							 --lr "${LR}" \
							 --evals_per_epoch 10 \
							 --eps ${EPSILON} \
							 --gradient_accumulation_steps ${ACC} \
							 --num_training_epochs "${EPOCHS}" \
							 --warmup ${WARMUP} \
							 --model_name "${MODEL}" \
							 --curriculum "none" \
							 --model_dir "${model_dir_name}" \
							 --log_interval ${INTERVAL} \
							 --seed "${SEED}" \
							 --save_dir "${main_dir}/" \
							 --data_dir "../data/" \
							 --save_steps_epochs \
							 --log_dynamics \
							 --baseline

elif [[ ${SETTING} = 'sharding' ]]; then
	main_dir="../trained_models_shards"

	if [[ ${DATASET} = 'rte' ]]; then
		shards=(0 1 2)
	else
		shards=(0 1 2 3 4 5 6 7 8 9)
	fi

	for i in "${shards[@]}"; do
		model_dir_name="${DATASET}${i}_${MODEL}_${SEED}_LR${LR}_LEN${MAX_SEQ_LEN}_BS${BS}_E${EPOCHS}"

		python main_seq_class.py --dataset_name "${DATASET}${i}" \
								 --mode "train" \
								 --max_seq_length ${MAX_SEQ_LEN} \
								 --batch_size ${BS} \
								 --lr "${LR}" \
								 --evals_per_epoch 10 \
								 --eps ${EPSILON} \
								 --gradient_accumulation_steps ${ACC} \
								 --num_training_epochs "${EPOCHS}" \
								 --warmup ${WARMUP} \
								 --model_name "${MODEL}" \
								 --curriculum "none" \
								 --model_dir "${model_dir_name}" \
								 --log_interval ${INTERVAL} \
								 --seed "${SEED}" \
								 --save_dir "${main_dir}/" \
								 --data_dir "../data_shards/" \
								 --save_steps_epochs \
								 --baseline
	done

else

	model_dir_name="${DATASET}_${MODEL}_${SEED}_LR${LR}_LEN${MAX_SEQ_LEN}_BS${BS}_E${EPOCHS}_${SETTING}_${DYNAMICS}-${METRIC1}"

    # curricula
    python main_seq_class.py --dataset "${DATASET}" \
							 --mode "train" \
							 --max_seq_length ${MAX_SEQ_LEN} \
							 --batch_size ${BS} \
							 --lr "${LR}" \
							 --evals_per_epoch 10 \
							 --eps ${EPSILON} \
							 --gradient_accumulation_steps ${ACC} \
							 --num_training_epochs "${EPOCHS}" \
							 --warmup ${WARMUP} \
							 --model_name "${MODEL}" \
							 --curriculum "${SETTING}" \
							 --log_interval "${INTERVAL}" \
							 --use_dynamics "${DYNAMICS}" \
							 --model_dir "${model_dir_name}" \
							 --save_dir "${main_dir}/" \
							 --data_dir "../data/" \
							 --seed "${SEED}" \
							 --save_steps_epochs \
							 --metric1 "${METRIC1}" \
							 --metric2 "${METRIC2}"

fi