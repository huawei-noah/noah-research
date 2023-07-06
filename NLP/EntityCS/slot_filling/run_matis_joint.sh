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

saving_dir="../multiatis_joint_saved_models"

# PEP-MS
#main_model="v7_16_2_pep_rndFalse_sameTrue_EntProb1.0_wikiann_langs_nodp"
#STEPS=(
#"step_200000_ppl_39.0267"
#)


STEPS=("baseline")
main_model="xlm-roberta-base"

for step in "${STEPS[@]}"; do
  	for SEED in 11 22 42 33 55; do

		BS=8
		LR=3e-5
		WARMUP=0.0
		TAG=False
		EVAL_LANGS="en,es,de,hi,fr,pt,zh,ja,tr"

		save_model_dir_name="multiatis_${main_model}/${step}_seed${SEED}"

		if [[ "${main_model}" != "xlm-roberta-base" ]]; then
			model_dir="/mnt/data/fenia/CodeSwitching/cs_pretrain_saved_models/${main_model}/${step}"
		else
			model_dir="xlm-roberta-base"
		fi

		python main_joint_intent_sf.py  \
			--all_val_langs False \
			--model_name_or_path "${model_dir}" \
			--dataset_name "../data/multiatis" \
			--xlmr_langs "../cs_pretrain/languages/xlmr_langs" \
			--eval_languages ${EVAL_LANGS} \
			--num_train_epochs 10 \
			--do_predict \
			--warmup_ratio ${WARMUP} \
			--learning_rate ${LR} \
			--max_length 128 \
			--per_device_train_batch_size ${BS} \
			--output_dir "${saving_dir}/${save_model_dir_name}" \
			--return_entity_level_metrics \
			--seed ${SEED} \
			--add_language_tag ${TAG} \
			--max_clip_grad_norm 1.0 \
			--evals_per_epoch 5 \

		python collect_joint_predictions.py --model_dir "${saving_dir}/${save_model_dir_name}" --seeds ${SEED}  --languages ${EVAL_LANGS}
  	done
  	python collect_joint_predictions.py --model_dir "${saving_dir}/${save_model_dir_name}"  --languages ${EVAL_LANGS}
done