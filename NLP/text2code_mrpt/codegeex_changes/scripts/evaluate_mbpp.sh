# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
# This script is based on CodeGeeX's original scripts/evaluate_humaneval_x.sh and has been adapted
# for evaluating the functional correctness of the generated codes of MBPP.

INPUT_FILE=$1  # Path to the .jsonl file that contains the generated codes.
LANGUAGE=$2  # Target programming language, currently support one of ["python", "java", "cpp", "js", "go"]
N_WORKERS=$3  # Number of parallel workers.
TIMEOUT=$4  # Timeout in seconds.

SCRIPT_PATH=$(realpath "$0")
SCRIPT_DIR=$(dirname "$SCRIPT_PATH")
MAIN_DIR=$(dirname "$SCRIPT_DIR")

echo "$INPUT_FILE"

if [ -z "$N_WORKERS" ]
then
    N_WORKERS=64
fi

if [ -z "$LANGUAGE" ]
then
    LANGUAGE=python
fi

if [ -z "$TIMEOUT" ]
then
    TIMEOUT=5
fi

DATA_DIR=$MAIN_DIR/mbpp_test.jsonl

if [ $LANGUAGE = go ]; then
  export PATH=$PATH:/usr/local/go/bin
fi

if [ $LANGUAGE = cpp ]; then
  export PATH=$PATH:/usr/bin/openssl
fi

CMD="python $MAIN_DIR/codegeex/benchmark/mbpp/evaluate_mbpp.py \
    --input_file "$INPUT_FILE" \
    --n_workers $N_WORKERS \
    --tmp_dir $MAIN_DIR/codegeex/benchmark/mbpp/ \
    --problem_file $DATA_DIR \
    --timeout $TIMEOUT"

echo "$CMD"
eval "$CMD"