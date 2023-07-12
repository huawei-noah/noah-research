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

dataset=$1
model=$2
LR=$3
EPS=$4
BS=$5
LEN=$6
SHARDS=$7

cd ../

for SEED in 123 456 789; do

	for (( i=0; i<${SHARDS}; i+=1 )); do
		for (( j=0; j<${SHARDS}; j+=1 )); do
			if [[ "${i}" != "${j}" ]]; then
				python evaluate.py \
				   --dataset_name "${dataset}${i}" \
				   --model_dir "../trained_models_shards/${dataset}${j}_${model}_${SEED}_LR${LR}_LEN${LEN}_BS${BS}_E${EPS}" \
				   --data_dir "../data_shards/" \
				   --model_name "${model}"
			fi
		done
	done
done