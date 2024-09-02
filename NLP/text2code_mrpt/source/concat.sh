# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

python sample_concatenation.py \
--dataset_dir=path_to_data/python_data_from_ast_1024docstr_toksAgree_v2 \
--max_seq_length=1024 \
--tokenizer=pycodegpt \
--model_name_or_path=/nfs/aiml2/nlp_team/fenia/MRPT/stage1_trained_models_100M/pycodegpt \
--save_name=pycodegpt_partial_sep \
--separate_some_embeds="python_tokens.txt"
#--separate_embeds=True
#--separate_some_embeds="python_tokens.txt"