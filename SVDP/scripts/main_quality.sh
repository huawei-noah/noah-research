#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "<STEP> Evaluating quality for ${MODEL_PATH} with ${MODEL_NAME}_baseline implementation on ${BENCHMARKS} at port ${DEPLOY_PORT}!"
export TAG_NAME=${MODEL_NAME}_baseline
echo "Will write results under outputs/quality/${TAG_NAME}"
./UltraEval/evaluate.sh vllm_implementations/${MODEL_NAME}_baseline.py ${MODEL_PATH} ${BENCHMARKS} ${DEPLOY_PORT} ${TAG_NAME}

for s in 0.4 0.5 0.6 0.7
do
    echo "<STEP> Constructing predictors with rank ${RANK} and s ${s}"
    export PREDICTORS_PATH=weights/main_results/${MODEL_NAME}/r${RANK}_s${s}
    export TAG_NAME=${MODEL_NAME}_svd_predictors_r${RANK}_s${s}
    python3 utils/construct_predictors.py \
        --model_path $MODEL_PATH \
        --predictors_output_path $PREDICTORS_PATH \
        --rank $RANK \
        --s $s \
        --sparsity_plot_output_file outputs/tmp.pdf \
        --calibration_prompts_path calibration_prompts.json \
        --torch_dtype float16 \
        --device_map cuda:0

    echo "<STEP> Evaluating quality for ${MODEL_PATH} with ${MODEL_NAME}_svd_predictors implementation on ${BENCHMARKS} at port ${DEPLOY_PORT}!"
    echo "Will write results under outputs/quality/${TAG_NAME}"
    ./UltraEval/evaluate.sh vllm_implementations/${MODEL_NAME}_svd_predictors.py ${MODEL_PATH} ${BENCHMARKS} ${DEPLOY_PORT} ${TAG_NAME}
done