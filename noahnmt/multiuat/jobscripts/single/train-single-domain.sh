SRC=$1
TGT=$2
DOMAIN_PATH=$3
DATA_PATH=$4
TAG=$5
SAVE_DIR=$6

echo ${SRC}
echo ${TGT}
echo ${DOMAIN_PATH}
echo ${DATA_PATH}
echo ${SAVE_DIR}

mkdir -p ${SAVE_DIR}

python fairseq/fairseq_cli/train.py \
    ${DATA_PATH} \
    --log-format tqdm --log-interval 10 \
    --source-lang ${SRC} --target-lang ${TGT} \
    --task translation \
    --arch transformer --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 \
    --lr 0.0007 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --weight-decay 0.0\
    --max-tokens 8192 --skip-invalid-size-inputs-valid-test \
    --num-workers 16 --max-epoch 30 \
    --keep-last-epochs 1 \
    --fp16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir ${SAVE_DIR} |& tee ${SAVE_DIR}/train-single.sh.log


rm ${SAVE_DIR}/checkpoint[0-9]*


python evaluate-ckpt-multidomain.py --domain-file ${DOMAIN_PATH} --setup iid --ckpt ${SAVE_DIR}/checkpoint_best.pt --srclang ${SRC} --tgtlang ${TGT} --split test |& tee ${SAVE_DIR}/train-single.sh.evaluation.iid.${TEMP}.test.log

python evaluate-ckpt-multidomain.py --domain-file ende-ood-domain.txt --setup ood --ckpt ${SAVE_DIR}/checkpoint_best.pt --srclang ${SRC} --tgtlang ${TGT} --split test |& tee ${SAVE_DIR}/train-single.sh.evaluation.ood.${TEMP}.test.log
