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


# HEURISTICS
for metric in "length" "rarity" "ppl"; do
	for seed in 123 456 789; do

		if [[ ${metric} == "length" ]]; then
			metric_file="${dataset}_heuristics"
		elif [[ ${metric} == "rarity" ]]; then
			metric_file="${dataset}_heuristics"
		else
			metric_file="${dataset}_${model}_ppl"
		fi

		if [[ ${seed} == 123 ]]; then
			curriculum="competence-0.01-0.9-${STEPS[0]}"
		elif [[ ${seed} == 456 ]]; then
			curriculum="competence-0.01-0.9-${STEPS[1]}"
		else
			curriculum="competence-0.01-0.9-${STEPS[2]}"
		fi

		bash "run_${dataset}.sh" \
			${curriculum} \
			${model} \
			${seed} \
			${lr} \
			${epochs} \
			${metric_file} \
			${metric}

	done
done