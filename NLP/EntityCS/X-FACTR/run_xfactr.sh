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

LANGUAGES=("en" "fr" "nl" "es" "ru" "zh" "he" "tr" "ko" "vi" "el" "mr" "ja" "hu" "bn" "ceb" "war" "tl" "sw" "pa" "mg" "ilo")

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
#main_model="v7_16_2_pep_rndFalse_sameTrue_EntProb1.0_wikiann_langs_nodp"
#STEPS=(
#"step_750000_ppl_28.6346"
#)


main_model="xlm-roberta-base"
STEPS=("baseline")

for step in "${STEPS[@]}"; do
	echo "${main_model}"

	model_dir="${main_model}/"
	saving_dir="xfactr_predictions/${main_model}"

	for LANG in "${LANGUAGES[@]}"; do
		echo "${LANG}"
		echo "Using INDEPENDENT predictions for ${LANG}..."
		python scripts/probe.py \
			   --model "${model_dir}" \
			   --lang "${LANG}" \
			   --pred_dir "${saving_dir}/${LANG}_ind" \
			   --num_mask 5 \
			   --batch_size 32
	done
done

echo "========== Running collect predictions NOW ==========="
python collect_predictions.py "${model_dir}" "${saving_dir}" "ind"



for step in "${STEPS[@]}"; do
	echo "${main_model}"

	model_dir="${main_model}/${step}"
	saving_dir="xfactr_predictions/${main_model}"

	for LANG in "${LANGUAGES[@]}"; do
		echo "${LANG}"
		echo "Using CONFIDENCE-based predictions for ${LANG} ..."
		python scripts/probe.py \
				--model "${model_dir}" \
				--lang "${LANG}" \
				--pred_dir "${saving_dir}/${LANG}_conf" \
				--init_method "confidence" \
				--iter_method "confidence" \
				--max_iter 10 \
				--num_mask 5 \
				--batch_size 32
	done
done

echo "======== Running collect predictions NOW ========"
python collect_predictions.py "${model_dir}" "${saving_dir}" "conf"

