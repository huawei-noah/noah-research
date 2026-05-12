export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python3 utils/construct_predictors.py \
    --model_path ../models/Tiiny/SparseQwen2-7B \
    --calibration_prompts_path calibration_prompts.json\
    --predictors_output_path predictors/weights/sparse_qwen2/r512_s05 \
    --rank 512 \
    --s 0.5 \
    --torch_dtype float16 \
    --device_map cuda:0 \
    --ablation_config_whitening cholesky \
    --ablation_config_bias full \
    --ablation_config_penalty full \
    --sparsity_plot_output_file outputs/sparsity.pdf
