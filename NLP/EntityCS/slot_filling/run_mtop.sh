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

saving_dir="../mtop_saved_models"


# PEP-MS + MLM
#main_model="v7_16_2_pep-mlm_rndFalse_sameTrue_EntProb0.5_wikiann_langs_nodp"
#STEPS=(
#"step_700000_ppl_5.9913"
#)

main_model='xlm-roberta-base'
STEPS=('baseline')

for step in "${STEPS[@]}"; do
	for SEED in 11 22 33 42 55; do
		BS=8
		LR=2e-5
		WARMUP=0.1
		TAG=False
		EVAL_LANGS="en,fr,de,hi,es,th"

		save_model_dir_name="mtop_${main_model}/${step}_seed${SEED}"

		if [[ "${main_model}" != "xlm-roberta-base" ]]; then
			model_dir="/mnt/data/fenia/CodeSwitching/cs_pretrain_saved_models/${main_model}/${step}"
		else
			model_dir="xlm-roberta-base"
		fi

		python main_sf.py  \
			--all_val_langs False \
			--model_name_or_path "${model_dir}" \
			--dataset_name "../data/mtop" \
			--pretraining_languages "../cs_pretrain/languages/wikiann_langs" \
			--xlmr_langs "../cs_pretrain/languages/xlmr_langs" \
			--eval_languages ${EVAL_LANGS} \
			--metric_for_best_model "overall_f1" \
			--do_train \
			--do_eval \
			--do_predict \
			--num_train_epochs 10 \
			--task_name "slot" \
			--learning_rate ${LR} \
			--max_seq_length 128 \
			--max_grad_norm 1.0 \
			--per_device_train_batch_size ${BS} \
			--per_device_eval_batch_size ${BS} \
			--warmup_ratio "${WARMUP}" \
			--output_dir "${saving_dir}/${save_model_dir_name}" \
			--return_entity_level_metrics \
			--save_total_limit 1 \
			--load_best_model_at_end True \
			--save_strategy "steps" \
			--logging_strategy "steps" \
			--evaluation_strategy "steps" \
			--evals_per_epoch 5 \
			--disable_tqdm True \
			--seed ${SEED}

		python collect_predictions.py --model_dir "${saving_dir}/${save_model_dir_name}" --seeds ${SEED}  --languages ${EVAL_LANGS}
	done
	python collect_predictions.py --model_dir "${saving_dir}/${save_model_dir_name}"  --languages ${EVAL_LANGS}
done
