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

NPROC_PER_NODE=1

BS=2
ACC=2
MAX_SEQ_LEN=128
LANG_TAG=False

MASKING="ep-mlm" # ep, mlm, ep-mlm
ENTITY_PROB=0.5
PARTIAL_MASKING=True
KEEP_RANDOM=False
KEEP_SAME=True
LANGS="wikiann_langs"

if [[ "${MASKING}" == "mlm" ]]; then
	save_dir_name="v7_${BS}_${ACC}_${MASKING}_rnd${KEEP_RANDOM}_same${KEEP_SAME}_EntProb${ENTITY_PROB}_${LANGS}_nodp"
elif [[ "${PARTIAL_MASKING}" == "True" ]]; then
	save_dir_name="v7_${BS}_${ACC}_p${MASKING}_rnd${KEEP_RANDOM}_same${KEEP_SAME}_EntProb${ENTITY_PROB}_${LANGS}_nodp"
else
	save_dir_name="v7_${BS}_${ACC}_w${MASKING}_rnd${KEEP_RANDOM}_same${KEEP_SAME}_EntProb${ENTITY_PROB}_${LANGS}_nodp"
fi

python -m torch.distributed.launch --nproc_per_node ${NPROC_PER_NODE} --use_env \
main_lm_no_trainer.py \
--model_name_or_path "xlm-roberta-base" \
--hf_datasets_folder_no_en "/mnt/data/fenia/CodeSwitching/entity_cs_no_en" \
--hf_datasets_folder_en "/mnt/data/fenia/CodeSwitching/entity_cs_en" \
--languages "languages/${LANGS}" \
--seed 42 \
--max_seq_length ${MAX_SEQ_LEN} \
--validation_steps 10000 \
--validation_examples_per_lang 100 \
--fp16 \
--per_device_train_batch_size ${BS} \
--per_device_eval_batch_size ${BS} \
--gradient_accumulation_steps ${ACC} \
--learning_rate 5e-5 \
--weight_decay 0.0 \
--max_clip_grad_norm 1.0 \
--num_train_epochs 1 \
--warmup_ratio 0.0 \
--insert_lang_tag ${LANG_TAG} \
--output_dir "cs_pretrain_saved_models/${save_dir_name}" \
--partial_masking ${PARTIAL_MASKING} \
--keep_random ${KEEP_RANDOM} \
--keep_same ${KEEP_SAME} \
--masking ${MASKING}  \
--entity_probability ${ENTITY_PROB} \
--update_partial_layers "embeddings" "10" "11" "lm_head"
