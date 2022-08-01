export CUDA_VISIBLE_DEVICES=0

if false; then
# MPII
IMG_DIR="path to the MPII images"
CLIFF_GT_PATH="cliffGT_v1/mpii_cliffGT.npz"
fi

if true; then
# COCO 2014 part
IMG_DIR="path to the COCO images"
CLIFF_GT_PATH="cliffGT_v1/coco2014part_cliffGT.npz"
fi


#python viz_cliffGT_perImg.py --img_dir ${IMG_DIR} --cliffGT_path ${CLIFF_GT_PATH}
python viz_cliffGT_perSubject.py --img_dir ${IMG_DIR} --cliffGT_path ${CLIFF_GT_PATH}