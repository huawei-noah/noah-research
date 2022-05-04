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
  REPO="/cache/build_wiki_vocabs_cloud"
  REPO_TASK="/cache/"
  WORK_DIR="$PWD"
else
  REPO="$PWD/cache/build_wiki_vocabs_cloud"
  REPO_TASK="$PWD/cache"
  WORK_DIR="$PWD"
fi

# parse commandline args
ARGS=`getopt -o x \
  -l gpu:,seed:,langs:,data_pat:,vocab_sizes:,preprocess:,build_vocab:,bpe_package:  \
  -- "$@"`

while [ -n "$1" ]
do
  case "$1" in
  --gpu) echo; shift ;;
  --seed) echo; shift ;;
  --langs) LANGS="$2"; shift ;;
  --data_pat) DATA_PAT="$2"; shift ;;
  --vocab_sizes) VOCAB_SIZES="$2"; shift ;;
  --preprocess) if [ $2 == "True" ]; then PREPROCESS="--preprocess"; fi; shift ;;
  --build_vocab) if [ $2 == "True" ]; then BUILD_VOCAB="--build_vocab"; fi; shift ;;
  --bpe_package) BPE_PACKAGE="$2"; shift ;;
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

SCRIPT="$WORK_DIR/third_party/utils/preprocess_wiki_txt.py"

LANGS=${LANGS:-"en,de,es,fr,hi,ja,pt,th,tr,zh"}
DATA_PAT=${DATA_PAT:-""}
VOCAB_SIZES=${VOCAB_SIZES:-10000}
PREPROCESS=${PREPROCESS:-""}
BUILD_VOCAB=${BUILD_VOCAB:-""}
BPE_PACKAGE=${BPE_PACKAGE:-"subword-nmt"}


DATA_DIR=$REPO/download
OUT_DIR=$REPO/outputs

mkdir -p $OUT_DIR

# prepare nltk data
mkdir -p $HOME/nltk_data/tokenizers/
cp -r $WORK_DIR/third_party/utils/punkt $HOME/nltk_data/tokenizers

echo "python $SCRIPT \
  --langs $LANGS \
  --vocab_sizes $VOCAB_SIZES \
  --data_pat $DATA_PAT \
  --data_dir $DATA_DIR \
  --out_dir $OUT_DIR \
  --bpe_package $BPE_PACKAGE \
  $PREPROCESS \
  $BUILD_VOCAB
"

python $SCRIPT \
  --langs $LANGS \
  --vocab_sizes $VOCAB_SIZES \
  --data_pat $DATA_PAT \
  --data_dir $DATA_DIR \
  --out_dir $OUT_DIR \
  --bpe_package $BPE_PACKAGE \
  $PREPROCESS \
  $BUILD_VOCAB
