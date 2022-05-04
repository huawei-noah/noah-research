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
  REPO="/cache/train_distill_cloud"
  REPO_TASK="/cache/"
  WORK_DIR="$PWD"
else
  REPO="$PWD/cache/train_distill_cloud"
  REPO_TASK="$PWD/cache/"
  WORK_DIR="$PWD"
fi


# parse commandline args
ARGS=`getopt -o x \
  -l gpu:,model_src:,model_via:,model_tgt:,train_langs_src:,train_langs:,test_langs:,data_ratio:,task:,loss_alpha:,loss_kl_temp:,loss_kl_red:,rand_init:,seed:,data_url:,teacher_dist_margin:,vocab_size:,vocab_size_src:,tgt_lang_for_dev:,tgt_lang_for_test:,data_ver:,task_type:,sent_cls:,vocabs_dir:,src_lr:,dist_int_lr:,dist_tgt_lr:,src_ep:,dist_int_ep:,dist_tgt_ep:,dist_tgt_lr_decay:,run_steps:,int_lr_find_params:,tgt_lr_find_params:,int_trans_train_dist:,tgt_train_aug:,mono_src_dir:,multi_dir: \
  -- "$@"`
while [ -n "$1" ]
do
  case "$1" in
  --gpu) GPU="$2"; shift ;;
  --model_src) MODEL_SRC="$2"; shift ;;
  --model_via) MODEL_VIA="$2"; shift ;;
  --model_tgt) MODEL="$2"; shift ;;
  --train_langs_src) TRAIN_LANGS_SRC="$2"; shift ;;
  --train_langs) TRAIN_LANGS="$2"; shift ;;
  --test_langs) TEST_LANGS="$2"; shift ;;
  --data_ratio) DATA_RATIO="$2"; shift ;;
  --task) TASK="$2"; shift ;;
  --task_type) TASK_TYPE="$2"; shift ;;
  --loss_alpha) LOSS_ALPHA="$2"; shift ;;
  --loss_kl_temp) LOSS_KL_TEMP="$2"; shift ;;
  --loss_kl_red) LOSS_KL_RED="$2"; shift ;;
  --rand_init) if [ "${2}" = "src" ]; then SRC_RAND_INIT="True"; elif [ "${2}" = "tgt" ]; then TGT_RAND_INIT="--rand_init"; elif [ "${2}" = "all" ]; then SRC_RAND_INIT="True"; TGT_RAND_INIT="--rand_init"; fi; shift ;;
  --seed) SEED="$2"; shift ;;
  --teacher_dist_margin) TEACHER_DIST_MARGIN="$2"; shift ;;
  --vocab_size) VOCAB_SIZE="$2"; shift ;;
  --vocab_size_src) VOCAB_SIZE_SRC="$2"; shift ;;
  --vocabs_dir) VOCABS_DIR="$2"; shift ;;
  --tgt_lang_for_dev) if [ "${2}" == "True" ]; then TGT_LANG_FOR_DEV="--tgt_lang_for_dev"; fi; shift ;;
  --tgt_lang_for_test) if [ "${2}" == "True" ]; then TGT_LANG_FOR_TEST="--tgt_lang_for_test"; fi; shift ;;
  --data_ver) DATA_VER="$2"; shift ;;
  --run_steps) RUN_STEPS="$2"; shift ;;
  --src_lr) LR="$2"; shift ;;
  --dist_int_lr) DIST_INT_LR="$2"; shift ;;
  --dist_tgt_lr) DIST_TGT_LR="$2"; shift ;;
  --src_ep) EPOCH="$2"; shift ;;
  --dist_int_ep) DIST_INT_EP="$2"; shift ;;
  --dist_tgt_ep) DIST_TGT_EP="$2"; shift ;;
  --dist_tgt_lr_decay) DIST_TGT_LR_DECAY="$2"; shift ;;
  --sent_cls) if [ "${2}" == "True" ]; then SENT_CLS="--sent_cls"; fi; shift ;;
  --int_lr_find_params) INT_LR_FIND_PARAMS="$2"; shift ;;
  --tgt_lr_find_params) TGT_LR_FIND_PARAMS="$2"; shift ;;
  --int_trans_train_dist) if [ "${2}" == "True" ]; then INT_TRANS_TRAIN_DIST="--trans_train_dist"; fi; shift ;;
  --tgt_train_aug) if [ "${2}" == "True" ]; then TGT_TRAIN_AUG="--train_aug"; fi; shift ;;
  --data_url) echo "data_url = $2"; shift ;;
  --mono_src_dir) MONO_SRC_DIR="$2"; shift ;;
  --multi_dir) MULTI_DIR="$2"; shift ;;
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
GPU=${GPU:-0}
MODEL_SRC=${MODEL_SRC:-bert-base-cased}
MODEL_VIA=${MODEL_VIA:-bert-base-multilingual-cased}
MODEL=${MODEL:-bert-base-chinese} # MODEL_TGT
TRAIN_LANGS_SRC=${TRAIN_LANGS_SRC:-"en"}
TRAIN_LANGS=${TRAIN_LANGS:-"zh"}
TEST_LANGS=${TEST_LANGS:-$TRAIN_LANGS}
DATA_RATIO=${DATA_RATIO:-1}
TASK=${TASK:-"mtop"}
SEED=${SEED:-42}
TEACHER_DIST_MARGIN=${TEACHER_DIST_MARGIN:--1}
VOCAB_SIZE=${VOCAB_SIZE:-10000} # 210913: create, 210925: cmd-arg
VOCAB_SIZE_SRC=${VOCAB_SIZE_SRC:-10000} # 211011: specify src vocab
VOCABS_DIR=${VOCABS_DIR:-"vocabs"}
TGT_LANG_FOR_DEV=${TGT_LANG_FOR_DEV:-""}
TGT_LANG_FOR_TEST=${TGT_LANG_FOR_TEST:-""}
DATA_VER=${DATA_VER:-""}
TASK_TYPE=${TASK_TYPE:-"s-cls"}
SENT_CLS=${SENT_CLS:-""}
RUN_STEPS=${RUN_STEPS:-"0,1,2"}
INT_TRANS_TRAIN_DIST=${INT_TRANS_TRAIN_DIST:-""}
TGT_TRAIN_AUG=${TGT_TRAIN_AUG:-""}
MULTI_DIR=${MULTI_DIR:-""}
MONO_SRC_DIR=${MONO_SRC_DIR:-""}


export CUDA_VISIBLE_DEVICES=$GPU
DATA_DIR=$REPO/$SEED/download
TASK_DATA_DIR=$DATA_DIR/$TASK
OUT_DIR=$REPO/$SEED/outputs
REPO_TASK="${REPO_TASK}/train_${TASK_TYPE}_cloud/${SEED}/outputs"
SCRIPT_TASK="$WORK_DIR/scripts/train_${TASK_TYPE}_cloud.sh"

echo "======== Configs ========"
echo REPO=$REPO
echo GPU=$GPU
echo MODEL_SRC=$MODEL_SRC
echo MODEL_VIA=$MODEL_VIA
echo MODEL=$MODEL
echo DATA_DIR=$DATA_DIR
echo OUT_DIR=$OUT_DIR
echo TRAIN_LANGS_SRC=$TRAIN_LANGS_SRC
echo TRAIN_LANGS=$TRAIN_LANGS
echo TEST_LANGS=$TEST_LANGS
echo DATA_RATIO=$DATA_RATIO
echo TASK=$TASK
echo SEED="$SEED"
echo TEACHER_DIST_MARGIN="$TEACHER_DIST_MARGIN"
echo VOCAB_SIZE="$VOCAB_SIZE"
echo VOCAB_SIZE_SRC="$VOCAB_SIZE_SRC"
echo VOCABS_DIR="$VOCABS_DIR"
echo TGT_LANG_FOR_DEV="$TGT_LANG_FOR_DEV"
echo TGT_LANG_FOR_TEST="$TGT_LANG_FOR_TEST"
echo DATA_VER="$DATA_VER"
echo TASK_TYPE="$TASK_TYPE"
echo SENT_CLS="$SENT_CLS"
echo RUN_STEPS="$RUN_STEPS"
echo INT_TRANS_TRAIN_DIST="$INT_TRANS_TRAIN_DIST"
echo TGT_TRAIN_AUG="$TGT_TRAIN_AUG"
echo MULTI_DIR=$MULTI_DIR
echo "====== Configs End ======"
echo ""


LC_SRC=""
if [[ "$MODEL_SRC" == bert-* ]]; then
  MODEL_TYPE_SRC="bert"
elif [ "$MODEL_SRC" == "xlm-mlm-100-1280" ] || [ "$MODEL_SRC" == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE_SRC="xlm"
  LC_SRC=" --do_lower_case"
elif [ "$MODEL_SRC" == "xlm-roberta-large" ] || [ "$MODEL_SRC" == "xlm-roberta-base" ]; then
  MODEL_TYPE_SRC="xlmr"
elif [[ "$MODEL_SRC" == cnn* ]]; then
  MODEL_TYPE_SRC="cnn"
elif [[ "$MODEL_SRC" == bilstm* ]]; then
  MODEL_TYPE_SRC="bilstm"
else
  # MODEL_TYPE_SRC="bert"
  MODEL_TYPE_SRC="$MODEL_SRC"
fi

LC_VIA=""
if [[ "$MODEL_VIA" == bert-* ]]; then
  MODEL_TYPE_VIA="bert"
elif [ "$MODEL_VIA" == "xlm-mlm-100-1280" ] || [ "$MODEL_VIA" == "xlm-mlm-tlm-xnli15-1024" ]; then
  MODEL_TYPE_VIA="xlm"
  LC_VIA=" --do_lower_case"
elif [ "$MODEL_VIA" == "xlm-roberta-large" ] || [ "$MODEL_VIA" == "xlm-roberta-base" ]; then
  MODEL_TYPE_VIA="xlmr"
elif [[ "$MODEL_VIA" == cnn* ]]; then
  MODEL_TYPE_VIA="cnn"
elif [[ "$MODEL_VIA" == bilstm* ]]; then
  MODEL_TYPE_VIA="bilstm"
else
  MODEL_TYPE_VIA="$MODEL_VIA"
fi

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


LOSS_ALPHA=${LOSS_ALPHA:-1.0} # fully KD training
LOSS_KL_TEMP=${LOSS_KL_TEMP:-1.0}
LOSS_KL_RED=${LOSS_KL_RED:-"batchmean"}
SRC_RAND_INIT=${SRC_RAND_INIT:-"False"}
TGT_RAND_INIT=${TGT_RAND_INIT:-""}
LR=${LR:-"1e-4"} # 211115: tune hyper-params of distillation
DIST_INT_LR=${DIST_INT_LR:-"1e-5"} # 211115: tune hyper-params of distillation
DIST_TGT_LR=${DIST_TGT_LR:-"1e-4"} # 211115: tune hyper-params of distillation
EPOCH=${EPOCH:-10} # 211115: tune hyper-params of distillation
DIST_INT_EP=${DIST_INT_EP:-50} # 211115: tune hyper-params of distillation
DIST_TGT_EP=${DIST_TGT_EP:-100} # 211115: tune hyper-params of distillation
DIST_TGT_LR_DECAY=${DIST_TGT_LR_DECAY:-"lin"} # 211115: tune hyper-params of distillation
INT_LR_FIND_PARAMS=${INT_LR_FIND_PARAMS:-""} # 211121: lr-finder
TGT_LR_FIND_PARAMS=${TGT_LR_FIND_PARAMS:-"100,1e-3"} # 211121: lr-finder
if [[ "$INT_LR_FIND_PARAMS" != "" ]]; then
  INT_LR_FIND_PARAMS="--lr_find_params $INT_LR_FIND_PARAMS"
fi
if [[ "$TGT_LR_FIND_PARAMS" != "" ]]; then
  TGT_LR_FIND_PARAMS="--lr_find_params $TGT_LR_FIND_PARAMS"
fi


MAXL=128
BATCH_SIZE=64
GRAD_ACC=1
if [ "$MODEL_TYPE" == "xlm" ] || [ "$MODEL_TYPE" == "xlmr" ]; then
  BATCH_SIZE=32
  GRAD_ACC=2
elif [ "$MODEL_TYPE" == "cnn" ]; then
  LR=$LR
elif [ "$MODEL_TYPE" == "bilstm" ]; then
  LR=$LR
elif [ "$MODEL_TYPE" == "mlp" ]; then
  LR=$LR
else
  BATCH_SIZE=64
  GRAD_ACC=1
fi

echo "======== HyperParams ========"
echo LOSS_ALPHA=$LOSS_ALPHA
echo LOSS_KL_TEMP=$LOSS_KL_TEMP
echo LOSS_KL_RED=$LOSS_KL_RED
echo SRC_RAND_INIT="$SRC_RAND_INIT"
echo TGT_RAND_INIT="$TGT_RAND_INIT"
echo LR=$LR
echo DIST_INT_LR=$DIST_INT_LR
echo DIST_TGT_LR=$DIST_TGT_LR
echo EPOCH=$EPOCH
echo DIST_INT_EP=$DIST_INT_EP
echo DIST_TGT_EP=$DIST_TGT_EP
echo DIST_TGT_LR_DECAY=$DIST_TGT_LR_DECAY
echo INT_LR_FIND_PARAMS=$INT_LR_FIND_PARAMS
echo TGT_LR_FIND_PARAMS=$TGT_LR_FIND_PARAMS
echo MAXL=$MAXL
echo LC_SRC=$LC_SRC
echo LC_VIA=$LC_VIA
echo LC=$LC
echo MODEL_TYPE_SRC=$MODEL_TYPE_SRC
echo MODEL_TYPE_VIA=$MODEL_TYPE_VIA
echo MODEL_TYPE=$MODEL_TYPE
echo BATCH_SIZE=$BATCH_SIZE
echo GRAD_ACC=$GRAD_ACC
echo "====== HyperParams End ======"
echo ""

RUNTIME=$(date '+%y%m%d-%H%M%S')


# NOTE: 2 pass distillation
if [ "${MULTI_DIR}" == "" ]; then
  if [[ "$RUN_STEPS" =~ .*"0".* ]]; then
    ############################
    ########## Step 0 ##########
    ############################
    # train / load src
    TASK_SAVE_TAG="src:${SEED}-${RUNTIME}-${DATA_RATIO}:${MODEL_SRC}:LR${LR}-epoch${EPOCH}-MaxLen${MAXL}-${TRAIN_LANGS_SRC}-${TRAIN_LANGS_SRC}"
    TASK_SAVE_DIR="$REPO_TASK/$TASK/$TASK_SAVE_TAG"
    mkdir -p $TASK_SAVE_DIR
    if [ "$DATA_VER" == "" ]; then
      DATA_VER_ARG=""
    else
      DATA_VER_ARG="--data_ver $DATA_VER"
    fi
    if [ "${MONO_SRC_DIR}" == "" ]; then
      # train mono-src 
      echo "[train_distill_cloud.sh] train mono-src: ${TASK_SAVE_TAG}"
      cmd="bash $SCRIPT_TASK \
        --model $MODEL_SRC \
        --gpu $GPU \
        --train_langs $TRAIN_LANGS_SRC \
        --test_langs $TRAIN_LANGS_SRC \
        --data_ratio $DATA_RATIO \
        --data_dir $DATA_DIR \
        --seed $SEED \
        --rand_init $SRC_RAND_INIT \
        --vocab_size $VOCAB_SIZE_SRC \
        --save_tag $TASK_SAVE_TAG \
        --task $TASK \
        --vocabs_dir $VOCABS_DIR \
        --learning_rate $LR \
        --epochs $EPOCH $DATA_VER_ARG"
      echo $cmd
      eval $cmd
    else
      # NOTE: skip training if mono-src is provided
      echo "[train_distill_cloud.sh] `ls -l ${MONO_SRC_DIR}`"
      echo "[train_distill_cloud.sh] load existing mono-src: ${MONO_SRC_DIR}"
    fi
  fi

  if [[ "$RUN_STEPS" =~ .*"1".* ]]; then
    ############################
    ########## Step 1 ##########
    ############################
    # 210914: mono-src -> multi-src -> mono-tgt
    # distill multi-src
    MODEL_TYPE_T=${MODEL_TYPE_SRC}
    MODEL_TYPE_S=${MODEL_TYPE_VIA}
    MODEL_PATH_T=${MONO_SRC_DIR}
    MODEL_PATH_S=$DATA_DIR/models/$MODEL_VIA
    LANGS_T=${TRAIN_LANGS_SRC}
    LANGS_S=${TRAIN_LANGS_SRC}
    if [ "$TGT_LANG_FOR_DEV" == "--tgt_lang_for_dev" ] || [ "$TGT_LANG_FOR_TEST" == "--tgt_lang_for_test" ]; then
      PRED_LANGS=${TEST_LANGS}
    else
      PRED_LANGS=${LANGS_S}
    fi
    if [ "${LC_SRC}" = " --do_lower_case" ]; then
      LC_T=${LC_SRC}_t
    else
      LC_T=""
    fi
    LC_S=${LC_VIA}
    TASK_SAVE_TAG="via:${SEED}-${RUNTIME}-${DATA_RATIO}:${MODEL_SRC}:${MODEL_VIA}:LR${DIST_INT_LR}-epoch${DIST_INT_EP}-MaxLen${MAXL}-${TRAIN_LANGS_SRC}-${TRAIN_LANGS_SRC}"
    TASK_SAVE_DIR="$OUT_DIR/$TASK/$TASK_SAVE_TAG"
    mkdir -p $TASK_SAVE_DIR
    echo "[train_distill_cloud.sh] distill multi-src: ${TASK_SAVE_TAG}"
    cmd="python $WORK_DIR/third_party/run_distill.py \
      --loss_alpha ${LOSS_ALPHA} \
      --loss_kl_temp ${LOSS_KL_TEMP} \
      --loss_kl_red ${LOSS_KL_RED} \
      --model_type_t $MODEL_TYPE_T \
      --model_type $MODEL_TYPE_S \
      --model_name_or_path_t $MODEL_PATH_T \
      --model_name_or_path $MODEL_PATH_S \
      --vocab_path_t ${TASK_DATA_DIR}/${VOCABS_DIR}/vocab.${LANGS_T}.${VOCAB_SIZE_SRC}.txt \
      --train_language_t $LANGS_T \
      --train_language $LANGS_S \
      --trans_train_languages $TRAIN_LANGS \
      --predict_languages $PRED_LANGS \
      --task_name $TASK \
      --data_dir ${TASK_DATA_DIR} \
      --output_dir $TASK_SAVE_DIR/ \
      --ratio_train_examples $DATA_RATIO \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --per_gpu_eval_batch_size $(($BATCH_SIZE*$GRAD_ACC)) \
      --learning_rate $DIST_INT_LR \
      --num_train_epochs $DIST_INT_EP \
      --max_seq_length_t $MAXL \
      --max_seq_length $MAXL \
      --seed ${SEED} \
      --teacher_dist_margin ${TEACHER_DIST_MARGIN} \
      --save_steps 100 \
      --log_file train \
      --save_only_best_checkpoint \
      --overwrite_output_dir \
      --eval_all_checkpoints \
      --do_train \
      --do_eval \
      --do_predict $TGT_LANG_FOR_DEV $TGT_LANG_FOR_TEST $SENT_CLS $INT_LR_FIND_PARAMS $INT_TRANS_TRAIN_DIST \
      $LC_T $LC_S"
    echo $cmd
    eval $cmd
  fi

else
  # NOTE: skip Step 0, 1 if multi is provided
  TASK_SAVE_TAG="via:${SEED}-${RUNTIME}-${DATA_RATIO}:${MODEL_SRC}:${MODEL_VIA}:LR${DIST_INT_LR}-epoch${DIST_INT_EP}-MaxLen${MAXL}-${TRAIN_LANGS_SRC}-${TRAIN_LANGS_SRC}"
  TASK_SAVE_DIR="$OUT_DIR/$TASK/$TASK_SAVE_TAG"
  mkdir -p $TASK_SAVE_DIR

  MODEL_TYPE_T=${MODEL_TYPE_VIA}
  MODEL_TYPE_S=${MODEL_TYPE}
  MODEL_PATH_T=${TASK_SAVE_DIR}/checkpoint-best
  MODEL_PATH_S=$DATA_DIR/models/$MODEL

  echo "[train_distill_cloud.sh] copy existing multi: ${MULTI_DIR} -> ${MODEL_PATH_T}"
  cp -r ${MULTI_DIR} ${MODEL_PATH_T}
  echo "[train_distill_cloud.sh] copied `ls -l ${MODEL_PATH_T}`"
  echo "[train_distill_cloud.sh] load existing multi: ${MODEL_PATH_T}"
fi

if [[ "$RUN_STEPS" =~ .*"2".* ]]; then
  ############################
  ########## Step 2 ##########
  ############################
  # distll mono-tgt / reversely distill mono-src
  MODEL_TYPE_T=${MODEL_TYPE_VIA}
  MODEL_TYPE_S=${MODEL_TYPE}
  MODEL_PATH_T=${TASK_SAVE_DIR}/checkpoint-best
  MODEL_PATH_S=$DATA_DIR/models/$MODEL
  if [ "${LC_VIA}" = " --do_lower_case" ]; then
    LC_T=${LC_VIA}_t
  else
    LC_T=""
  fi
  LC_S=${LC}

  for TRAIN_LANG in `echo $TRAIN_LANGS | tr ',' ' '`
  do 
    SAVE_TAG="${SEED}-${RUNTIME}-${DATA_RATIO}:${MODEL_SRC}:${MODEL_VIA}:${MODEL}:LR${DIST_TGT_LR}-epoch${DIST_TGT_EP}-MaxLen${MAXL}-${TRAIN_LANGS_SRC}-${TRAIN_LANG}-${TRAIN_LANG}"
    SAVE_DIR="$OUT_DIR/$TASK/$SAVE_TAG"
    mkdir -p $SAVE_DIR
    echo "[train_distill_cloud.sh] distill mono-tgt: ${SAVE_TAG}"

    LANGS_T=${TRAIN_LANG}
    LANGS_S=${TRAIN_LANG}

    DATA_SPLIT="train"
    cmd="python $WORK_DIR/third_party/run_distill.py \
      --loss_alpha ${LOSS_ALPHA} \
      --loss_kl_temp ${LOSS_KL_TEMP} \
      --loss_kl_red ${LOSS_KL_RED} \
      --model_type_t $MODEL_TYPE_T \
      --model_type $MODEL_TYPE_S \
      --model_name_or_path_t $MODEL_PATH_T \
      --model_name_or_path $MODEL_PATH_S \
      --vocab_path ${TASK_DATA_DIR}/${VOCABS_DIR}/vocab.${LANGS_S}.${VOCAB_SIZE}.txt \
      --train_language_t $LANGS_T \
      --train_language $LANGS_S \
      --predict_languages $TRAIN_LANG \
      --task_name $TASK \
      --data_dir ${TASK_DATA_DIR} \
      --output_dir $SAVE_DIR/ \
      --ratio_train_examples $DATA_RATIO \
      --gradient_accumulation_steps $GRAD_ACC \
      --per_gpu_train_batch_size $BATCH_SIZE \
      --per_gpu_eval_batch_size $(($BATCH_SIZE*$GRAD_ACC)) \
      --learning_rate $DIST_TGT_LR \
      --num_train_epochs $DIST_TGT_EP \
      --max_seq_length_t $MAXL \
      --max_seq_length $MAXL \
      --seed $SEED \
      --teacher_dist_margin ${TEACHER_DIST_MARGIN} \
      --train_split ${DATA_SPLIT} \
      --lr_decay ${DIST_TGT_LR_DECAY} \
      --save_steps 100 \
      --log_file train \
      --save_only_best_checkpoint \
      --overwrite_output_dir \
      --eval_all_checkpoints \
      --do_train \
      --do_eval \
      --do_predict $TGT_RAND_INIT $SENT_CLS $TGT_LR_FIND_PARAMS $TGT_TRAIN_AUG \
      $LC_T $LC_S"
    echo $cmd
    eval $cmd

  done

fi
