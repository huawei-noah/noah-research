#!/bin/bash

GEN=$1

if ! command -v sacremoses &> /dev/null
then
    echo "sacremoses could not be found, please install with: pip install sacremoses"
    exit
fi

grep ^H $GEN \
| sed 's/^H\-//' \
| sort -n -k 1 \
| cut -f 3 \
| sacremoses detokenize \
> $GEN.sorted.hyp.detok

grep ^T $GEN \
| sed 's/^T\-//' \
| sort -n -k 1 \
| cut -f 2 \
| sacremoses detokenize \
> $GEN.sorted.ref.detok

cat $GEN.sorted.hyp.detok | sacrebleu -w 2 $GEN.sorted.ref.detok | tee $GEN.sacrebleu
cat $GEN.sorted.hyp.detok | sacrebleu -w 2 -b $GEN.sorted.ref.detok | tee $GEN.sacrebleu.scoreonly