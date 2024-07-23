#!/usr/bin/env bash

# Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under MIT License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://opensource.org/license/mit
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

NUM_GPUs=4
DATA_PATH="/path/to/HumanRankEvalData/"
MODEL_DIR="/path/to/models/"

#---------------------------------------------------------------

deepspeed --num_gpus ${NUM_GPUs} main.py \
          --model auto_hf \
          --tasks human_rank_eval_* \
          --model_args pretrained=${MODEL_DIR}Vicuna-7B \
          --data_path ${DATA_PATH} \
          --batch_size 4 \
          --world_size ${NUM_GPUs}

#deepspeed --include localhost:2 main.py \
#          --model auto_hf \
#          --tasks human_rank_eval_* \
#          --model_args pretrained=${MODEL_DIR}Pythia-410M \
#          --batch_size 8 \
#          --data_path ${DATA_PATH}

#python main.py --model mindspore \
#               --tasks human_rank_eval_math \
#               --data_path ${DATA_PATH} \
#               --model_args pretrained=opt-350m \
#               --batch_size 4