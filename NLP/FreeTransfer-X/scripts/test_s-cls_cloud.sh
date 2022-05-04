#!/bin/bash
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE



if [ -d "/cache" ]; then
  REPO="/cache/test_s-cls_cloud"
  WORK_DIR="$PWD"
else
  REPO="$PWD/cache/test_s-cls_cloud"
  WORK_DIR="$PWD"
fi


# parse commandline args
ARGS=`getopt -o x \
  -l gpu:,model:,train_langs:,test_langs:,data_ratio:,data_dir:,save_tag:,load_dir:,rand_init:,seed:,data_url:,vocab_size:,test_split:,task:,data_ver:  \
  -- "$@"`
while [ -n "$1" ]
do
  case "$1" in
  --gpu) GPU="$2"; shift ;;
  --model) MODEL="$2"; shift ;;
  --train_langs) TRAIN_LANGS="$2"; shift ;;
  --test_langs) TEST_LANGS="$2"; shift ;;
  --data_ratio) DATA_RATIO="$2"; shift ;;
  --data_dir) DATA_DIR="$2"; shift ;;
  --save_tag) SAVE_TAG="$2"; shift ;;
  --load_dir) LOAD_DIR="$2"; shift ;;
  --rand_init) if [ "$2" = "True" ]; then RAND_INIT="--rand_init"; fi ;;
  --seed) SEED="$2"; shift ;;
  --data_url) echo "data_url = $2"; shift ;;
  --vocab_size) VOCAB_SIZE="$2"; shift ;;
  --test_split) TEST_SPLIT="$2"; shift ;;
  --task) TASK="$2"; shift ;;
  --data_ver) DATA_VER="$2"; shift ;;
  --) shift; break ;;
   *) echo "[parse_args] $1 is not option";;
  esac
  shift
done

count=1
for param in "$@"
do
  echo "Unknown parameter #$count: $param"
  count=$[ $count + 1 ]
done

# set default values
MODEL=${MODEL:-bert-base-multilingual-cased}
GPU=${GPU:-0}
TRAIN_LANGS=${TRAIN_LANGS:-"en"}
TEST_LANGS=${TEST_LANGS:-"en"}
DATA_RATIO=${DATA_RATIO:-1}
DATA_DIR=${DATA_DIR:-"$REPO/$SEED/download"}
SAVE_TAG=${SAVE_TAG:-""}
LOAD_DIR=${LOAD_DIR:-""}
RAND_INIT=${RAND_INIT:-""}
SEED=${SEED:-42}
VOCAB_SIZE=${VOCAB_SIZE:-10000}
TEST_SPLIT=${TEST_SPLIT:-"trans-test"}
TASK=${TASK:-""}
DATA_VER=${DATA_VER:-""}

TASK_DATA_DIR=$DATA_DIR/$TASK

export CUDA_VISIBLE_DEVICES=$GPU
OUT_DIR=$REPO/$SEED/outputs


echo "======== Configs ========"
echo REPO=$REPO
echo MODEL=$MODEL
echo GPU=$GPU
echo OUT_DIR=$OUT_DIR
echo TRAIN_LANGS=$TRAIN_LANGS
echo TEST_LANGS=$TEST_LANGS
echo DATA_RATIO=$DATA_RATIO
echo DATA_DIR=$DATA_DIR
echo SAVE_TAG=$SAVE_TAG
echo LOAD_DIR=$LOAD_DIR
echo RAND_INIT="${RAND_INIT}"
echo SEED="${SEED}"
echo VOCAB_SIZE="${VOCAB_SIZE}"
echo TEST_SPLIT="${TEST_SPLIT}"
echo TASK="${TASK}"
echo DATA_VER="$DATA_VER"
echo "====== Configs End ======"


LC=""
if [[ "$MODEL" == bert-* ]]; then
  MODEL_TYPE="bert"
elif [ "$MODEL" == "xlm-mlm-100-1280" ] || [ "$MODEL" == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE="xlm"
  LC=" --do_lower_case"
elif [ "$MODEL" == "xlm-roberta-large" ] || [ "$MODEL" == "xlm-roberta-base" ]; then
  MODEL_TYPE="xlmr"
elif [[ "$MODEL" == cnn* ]]; then
  MODEL_TYPE="cnn"
elif [[ "$MODEL" == bilstm* ]]; then
  MODEL_TYPE="bilstm"
else
  MODEL_TYPE="$MODEL"
fi


LR=2e-5
EPOCH=5
MAXL=128
BATCH_SIZE=64
GRAD_ACC=1


RUNTIME=$(date '+%y%m%d-%H%M%S')
if [ "${SAVE_TAG}" = "" ]; then
  SAVE_TAG="${SEED}-${RUNTIME}-${DATA_RATIO}:${MODEL}:LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-${TRAIN_LANGS}-${TEST_LANGS}"
fi
SAVE_DIR="$OUT_DIR/$TASK/$SAVE_TAG"
if [ "${LOAD_DIR}" = "" ]; then
  LOAD_DIR="$DATA_DIR/models/$MODEL"
fi
echo SAVE_TAG=${SAVE_TAG}
echo SAVE_DIR=${SAVE_DIR}
echo LOAD_DIR=${LOAD_DIR}


mkdir -p $SAVE_DIR


cmd="python ${WORK_DIR}/third_party/run_task.py \
  --model_type $MODEL_TYPE \
  --model_name_or_path ${LOAD_DIR} \
  --init_checkpoint ${LOAD_DIR} \
  --task_name $TASK \
  --do_predict \
  --data_dir ${TASK_DATA_DIR} \
  --per_gpu_eval_batch_size $(($BATCH_SIZE*$GRAD_ACC)) \
  --max_seq_length $MAXL \
  --output_dir $SAVE_DIR/ \
  --seed ${SEED} \
  --test_split $TEST_SPLIT \
  --log_file 'train' \
  --predict_languages $TEST_LANGS \
  --sent_cls \
  --overwrite_output_dir $LC"
echo $cmd
eval $cmd

