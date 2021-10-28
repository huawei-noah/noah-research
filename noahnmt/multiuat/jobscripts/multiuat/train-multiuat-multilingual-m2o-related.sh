TEMP=$1
REWARD=$2
K=$3
TAG=$4
SAVE_DIR=$5
DATA_PATH=corpus/multilingual/data-bin/ted_8_related
LANG_PAIRS="aze-eng,tur-eng,bel-eng,rus-eng,glg-eng,por-eng,slk-eng,ces-eng"

echo ${DATA_PATH}
echo ${LANG_PAIRS}
echo ${TEMP}
echo ${REWARD}
echo ${K}
echo ${TAG}
echo ${SAVE_DIR}

mkdir -p ${SAVE_DIR}

python fairseq/fairseq_cli/multiuat_train.py \
    ${DATA_PATH} \
    --log-format tqdm --log-interval 100 \
    --task multiuat_multilingual_translation --lang-pairs ${LANG_PAIRS} \
    --reward-type ${REWARD} --temperature ${TEMP} --max-sample-tokens 1024 --K ${K} \
    --data-actor base --data-actor-optim-step 200 --data-actor-lr 0.0001 --update-sampling-interval 2000 \
    --sample-prob-log ${SAVE_DIR}/probs.csv \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoder-input-output-embed --share-encoders --share-decoders \
    --max-epoch 40 \
    --dropout 0.3 --weight-decay 0.0001 \
    --left-pad-source 'True' --left-pad-target 'False' \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 5e-4 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4800 --update-freq 2 \
    --max-source-positions 150 --max-target-positions 150 \
    --encoder-normalize-before --decoder-normalize-before \
    --skip-invalid-size-inputs-valid-test \
    --encoder-langtok "src" --decoder-langtok \
    --save-dir ${SAVE_DIR} |& tee ${SAVE_DIR}/train-multiuat.sh.${TEMP}.log

rm ${SAVE_DIR}/checkpoint[0-9]*


python evaluate-ckpt-multilingual.py --lang-pairs ${LANG_PAIRS} --setup related --ckpt ${SAVE_DIR}/checkpoint_best.pt --split test |& tee ${SAVE_DIR}/train-multiuat-multilingual.sh.evaluation.${TEMP}.test.log