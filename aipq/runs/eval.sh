#!/bin/bash


# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      



ROOT_DATA_PATH=/storage/local/db/AIPQ/public

GPU_NO=0
CKPT_DIR=../checkpoints

DL_FRAMEWORK=torch # mindspore | torch

ARCH="DISTS"  # DISTS | LPIPS | L2POOL

# choose one from: 
# 	live, csiq, tid2013, liu, ma, shrq, dibr, pipal (train split)
# bonus datasets: qads, tqd, kadid, pieapp (test split)
DATASET=live

# paper datasets:
if [ "${DATASET}" == "live" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/LIVE"
elif [ "${DATASET}" == "csiq" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/CSIQ"
elif [ "${DATASET}" == "tid2013" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/TID2013"
elif [ "${DATASET}" == "liu" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/Liu2013"
elif [ "${DATASET}" == "ma" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/Ma-1620"
elif [ "${DATASET}" == "shrq" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/SHRQ"
elif [ "${DATASET}" == "dibr" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/DIBR"
elif [ "${DATASET}" == "pipal" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/PIPAL"
# bonus datasets:
elif [ "${DATASET}" == "qads" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/QADS"
elif [ "${DATASET}" == "tqd" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/TQD"
elif [ "${DATASET}" == "kadid" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/KADID"
elif [ "${DATASET}" == "pieapp" ]; then
	EVAL_DATA="${ROOT_DATA_PATH}/PieAPP"
else
	echo "Unknown dataset ${DATASET}"
fi
#--------------------------------

EVAL_SCRIPT=eval.py

echo "DL library: ${DL_FRAMEWORK}"

if [ "${DL_FRAMEWORK}" == "mindspore" ]; then
    DL_SHORT="ms"
    CKPT_FNAME=checkpoint.ckpt
else
    DL_SHORT="pt"
    CKPT_FNAME=checkpoint.pth.tar
fi

#----------
if [ "${ARCH}" == "LPIPS" ]; then
    EXP_NAME=lpips-vgg-bce-ps256-l4prp1-rs1-rk1-bs64-ep10-${DL_SHORT}
    EXP_OPTS="--lpips --backbone vgg"
elif [ "${ARCH}" == "L2POOL" ]; then
    EXP_NAME=lpips-vgg-l2pool-bce-ps256-l4prp1-rs1-rk1-bs64-ep10-${DL_SHORT}
    EXP_OPTS="--lpips --backbone vgg --l2pooling"
elif [ "${ARCH}" == "DISTS" ]; then
    EXP_NAME=dists-bce-ps256-l4prp1-rs1-rk1-bs64-ep10-${DL_SHORT}
    EXP_OPTS="--dists"
else
    echo "Unrecognized model: '${ARCH}'"
fi


echo "Model path: '${CKPT_DIR}/${EXP_NAME}/'"

echo "dataset: ${DATASET}, root_dir: '${EVAL_DATA}'"

CUDA_VISIBLE_DEVICES=$GPU_NO python ../${DL_FRAMEWORK}/${EVAL_SCRIPT} 	\
		--data-dir ${EVAL_DATA} 				\
		--dataset=${DATASET} 					\
		${EXP_OPTS} 						\
		--batch-size=1 						\
		-j=1 							\
		--loader=cv2 						\
		--model-path ${CKPT_DIR}/${EXP_NAME}/${CKPT_FNAME}

