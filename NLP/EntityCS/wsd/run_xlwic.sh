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

saving_dir='../xlwic_saved_models'

# WEP
#main_model="v7_16_2_wep_rndFalse_sameTrue_EntProb1.0_wikiann_langs_nodp"
#STEPS=(
#"step_100000_ppl_230.0281"
#)

main_model='xlm-roberta-base'
STEPS=('baseline')

for step in "${STEPS[@]}"; do
	for SEED in 11 22 33 42 55; do

		BS=8
		WARMUP=0.0
		LR=1e-5
		EPS=10

		save_model_dir_name="xlwic_${main_model}/${step}_seed${SEED}"

		if [[ "${main_model}" != "xlm-roberta-base" ]]; then
			model_dir="/mnt/data/fenia/CodeSwitching/cs_pretrain_saved_models/${main_model}/${step}"
		else
			model_dir="xlm-roberta-base"
		fi

		python main_xlwic.py  \
				--model_name_or_path "${model_dir}" \
				--dataset_name "../data/xl_wic" \
				--eval_languages "bg,da,et,fa,hr,ja,ko,nl,zh,de,fr,it" \
				--metric_for_best_model "accuracy" \
				--do_train \
				--do_eval \
				--do_predict \
				--num_train_epochs ${EPS} \
				--learning_rate ${LR} \
				--max_seq_length 128 \
				--max_grad_norm 1.0 \
				--per_device_train_batch_size ${BS} \
				--per_device_eval_batch_size 32 \
				--warmup_ratio "${WARMUP}" \
				--output_dir "${saving_dir}/${save_model_dir_name}" \
				--save_total_limit 1 \
				--load_best_model_at_end True \
				--save_strategy "steps" \
				--logging_strategy "steps" \
				--evaluation_strategy "steps" \
				--evals_per_epoch 5 \
				--disable_tqdm True \
				--seed ${SEED} \
				--overwrite_output_dir

		python collect_predictions.py --seeds ${SEED} --model_dir "${saving_dir}/${save_model_dir_name}"
	done
	python collect_predictions.py --model_dir "${saving_dir}/${save_model_dir_name}"
done
