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
lr=$3
epochs=$4
bs=$5
max_len=$6
STEPS=($7)


# CONFIDENCE
for seed in 123 456 789; do

	if [[ ${seed} == 123 ]]; then
		CURRICS=("competence-0.01-0.9-${STEPS[0]}" "competence-bias-0.01-0.9-${STEPS[0]}")
	elif [[ ${seed} == 456 ]]; then
		CURRICS=("competence-0.01-0.9-${STEPS[1]}" "competence-bias-0.01-0.9-${STEPS[1]}")
	else
		CURRICS=("competence-0.01-0.9-${STEPS[2]}" "competence-bias-0.01-0.9-${STEPS[2]}")
	fi

	for curriculum in "${CURRICS[@]}"; do
		script="run_${dataset}.sh"

		bash ${script} \
		${curriculum} \
		${model} \
		${seed} \
		${lr} \
		${epochs} \
		"${dataset}_${model}_${seed}_LR${lr}_LEN${max_len}_BS${bs}_E${epochs}" \
		"confidence" \
		"variability"
	done
done
