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

saving_dir="../wikiann_saved_models"

# WEP
#main_model="v7_16_2_wep_rndFalse_sameTrue_EntProb1.0_wikiann_langs_nodp"
#STEPS=(
#"step_750000_ppl_106.7362"
#)

# MLM
#main_model="v7_16_2_mlm_rndTrue_sameTrue_EntProb0.0_wikiann_langs_nodp"
#STEPS=(
#"step_600000_ppl_4.1798"
#)

# PEP-MS + MLM
#main_model="v7_16_2_pep-mlm_rndFalse_sameTrue_EntProb0.5_wikiann_langs_nodp"
#STEPS=(
#"step_770000_ppl_5.7259"
#)

# PEP-MS
main_model="v7_16_2_pep_rndFalse_sameTrue_EntProb1.0_wikiann_langs_nodp"
STEPS=(
"step_750000_ppl_28.6346"
)

# ------------------------------------ #

# PEP-MRS
#main_model="v7_16_2_pep_rndTrue_sameTrue_EntProb1.0_wikiann_langs_nodp"
#STEPS=(
#"step_750000_ppl_105.1040"
#)

# PEP-M
#main_model="v7_16_2_pep_rndFalse_sameFalse_EntProb1.0_wikiann_langs_nodp"
#STEPS=(
#"step_780000_ppl_66.2938"
#)

# PEP-MRS + MLM
#main_model="v7_16_2_pep-mlm_rndTrue_sameTrue_EntProb0.5_wikiann_langs_nodp"
#STEPS=(
#"step_770000_ppl_7.2147"
#)

# PEP-M + MLM
#main_model="v7_16_2_pep-mlm_rndFalse_sameFalse_EntProb0.5_wikiann_langs_nodp"
#STEPS=(
#"step_790000_ppl_6.8606"
#)

# WEP + MLM
#main_model="v7_16_2_wep-mlm_rndTrue_sameTrue_EntProb0.5_wikiann_langs_nodp"
#STEPS=(
#"step_720000_ppl_19.9380"
#)

# MLM english
#main_model="v7_16_2_mlm_rndTrue_sameTrue_EntProb0.0_en_nodp"
#STEPS=(
#"step_210000_ppl_3.3819"
#)

# MLM xlmr
#main_model="v7_16_2_mlm_rndTrue_sameTrue_EntProb0.0_xlmr_langs_nodp"
#STEPS=(
#"step_750000_ppl_4.3863"
#)

# WEP english
#main_model="v7_16_2_wep_rndFalse_sameTrue_EntProb1.0_en_nodp"
#STEPS=(
#"step_110000_ppl_39.7504"
#)

# WEP xlmr
#main_model="v7_16_2_wep_rndFalse_sameTrue_EntProb1.0_xlmr_langs_nodp"
#STEPS=(
#"step_1100000_ppl_122.0869"
#)

# PEP-MS + MLM english
#main_model="v7_16_2_pep-mlm_rndFalse_sameTrue_EntProb0.5_en_nodp"
#STEPS=(
#"step_140000_ppl_4.7765"
#)

# PEP-MS + MLM xlmr_langs
#main_model="v7_16_2_pep-mlm_rndFalse_sameTrue_EntProb0.5_xlmr_langs_nodp"
#STEPS=(
#"step_1090000_ppl_5.9205"
#)

# PEP-MS + MLM 0.8
#main_model="v7_16_2_pep-mlm_rndFalse_sameTrue_EntProb0.8_wikiann_langs_nodp"
#STEPS=(
#"step_770000_ppl_10.3077"
#)

# PEP-MS + MLM 1.0
#main_model="v7_16_2_pep-mlm_rndFalse_sameTrue_EntProb1.0_wikiann_langs_nodp"
#STEPS=(
#"step_800000_ppl_18.3643"
#)

#main_model='xlm-roberta-base'
#STEPS=('baseline')

for step in "${STEPS[@]}"; do
	for SEED in 11 22 33 42 55; do

		LR=1e-5
		WARMUP=0.1
		BS=8
		TAG=False

		save_model_dir_name="wikiann_${main_model}/${step}_seed${SEED}"

		if [[ "${main_model}" != "xlm-roberta-base" ]]; then
			model_dir="/mnt/data/fenia/CodeSwitching/cs_pretrain_saved_models/${main_model}/${step}"
		else
			model_dir="xlm-roberta-base"
		fi

		python main_ner.py  \
		--all_val_langs False \
		--model_name_or_path "${model_dir}" \
		--dataset_name "../data/wikiann" \
        --pretraining_languages "../cs_pretrain/languages/wikiann_langs" \
        --xlmr_langs "../cs_pretrain/languages/xlmr_langs" \
        --eval_languages "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu" \
		--metric_for_best_model "overall_f1" \
		--do_train \
		--do_eval \
		--do_predict \
		--num_train_epochs 10 \
		--task_name "ner" \
		--learning_rate ${LR} \
		--max_seq_length 128 \
		--max_grad_norm 1.0 \
		--per_device_train_batch_size ${BS} \
		--warmup_ratio ${WARMUP} \
		--output_dir "${saving_dir}/${save_model_dir_name}" \
		--return_entity_level_metrics \
		--save_total_limit 1 \
		--load_best_model_at_end True \
		--save_strategy "steps" \
		--logging_strategy "steps" \
		--evaluation_strategy "steps" \
		--evals_per_epoch 5 \
		--disable_tqdm True \
		--seed ${SEED} \
		--add_language_tag ${TAG} \
        --overwrite_output_dir

		python collect_predictions.py --model_dir "${saving_dir}/${save_model_dir_name}" --seeds ${SEED}
	done

	# Collect results from all seeds
	python collect_predictions.py --model_dir "${saving_dir}/${save_model_dir_name}"
done
