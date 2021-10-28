

TEMP=$1
TAG=$2
SAVE_DIR=$3
DATA_PATH=corpus/multilingual/data-bin/ted_8_diverse
LANG_PAIRS="eng-bos,eng-mar,eng-hin,eng-mkd,eng-ell,eng-bul,eng-fra,eng-kor"

echo ${DATA_PATH}
echo ${LANG_PAIRS}
echo ${TEMP}
echo ${TAG}
echo ${SAVE_DIR}

mkdir -p ${SAVE_DIR}

python fairseq/fairseq_cli/train.py\
    ${DATA_PATH} \
    --log-format tqdm --log-interval 100 \
    --task temperature_multilingual_translation \
    --temperature ${TEMP} \
    --arch multilingual_transformer_iwslt_de_en \
    --share-decoder-input-output-embed --share-encoders --share-decoders \
    --max-epoch 40 \
    --lang-pairs ${LANG_PAIRS} \
    --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 --weight-decay 0.0 \
    --left-pad-source 'True' --left-pad-target 'False' \
    --optimizer adam --adam-betas '(0.9, 0.98)' --lr-scheduler inverse_sqrt \
    --warmup-init-lr 1e-7 --warmup-updates 4000 --lr 2e-4 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 9600 \
    --max-source-positions 150 --max-target-positions 150 \
    --encoder-normalize-before --decoder-normalize-before \
    --skip-invalid-size-inputs-valid-test \
    --encoder-langtok "src" --decoder-langtok \
    --fp16 \
    --save-dir ${SAVE_DIR} |& tee ${SAVE_DIR}/train-temperature.sh.${TEMP}.log

rm ${SAVE_DIR}/checkpoint[0-9]*


python evaluate-ckpt-multilingual.py --lang-pairs ${LANG_PAIRS} --setup diverse --ckpt ${SAVE_DIR}/checkpoint_best.pt --split test |& tee ${SAVE_DIR}/train-temperature-multilingual.sh.evaluation.${TEMP}.test.log