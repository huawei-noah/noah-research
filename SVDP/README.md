# SVDP: Training-Free Contextual Sparsity Predictors for Fast LLM Inference

[![arXiv](https://img.shields.io/badge/arXiv-2603.14110-green.svg)](https://arxiv.org/abs/2603.14110)

Up to 1.8× end-to-end generation speedup with near-lossless accuracy.

## Description

This project exploits contextual sparsity in the FFN blocks of ReGLU-based LLMs via training-free predictors to speed up LLM inference.

## Models

Currently 4 LLMs can be tested:

- [ProSparseLLaMA2-7B](https://huggingface.co/SparseLLM/prosparse-llama-2-7b)
- [TurboSparse-Mistral-Instruct](https://huggingface.co/Tiiny/TurboSparse-Mistral-Instruct)
- [SparseQwen2-7B](https://huggingface.co/Tiiny/SparseQwen2-7B)
- [Bamboo](https://huggingface.co/Tiiny/Bamboo-DPO-v0_1)

> Bamboo and Mistral share one model implementation

## Contents

- `kernels` — custom CUDA kernels for sparse FFN;
- `vllm_implementations` — vLLM model implementations (baseline and SVD-predictor variants);
- `utils` — Python scripts for predictor construction, evaluation, profiling and stats;
- `scripts` — shell drivers that orchestrate the quality / performance / ablation experiments;
- `performance` — performance-evaluation scripts and sample prompt datasets;
- `notebooks` — notebooks for ROC-AUC and calibration figures;
- `npu` — NPU (`torch.npu`) solution and timing-analysis scripts;
- [UltraEval](https://github.com/OpenBMB/UltraEval) — vendored evaluation framework;
- `outputs`, `weights` — created at runtime for experiment results and generated predictors.

## Implementations

For each model we have 2 vLLM implementations:

1. `vllm_implementations/{model}_baseline.py` — the default implementation with dense computations. The code is based on official vLLM implementations. The activation function is replaced with ReLU. The implementation does not utilize sparsity in FFN blocks.
2. `vllm_implementations/{model}_svd_predictors.py` — the **optimized** version. It uses SVD predictors to identify the sparse pattern. Custom CUDA kernels reduce the number of memory reads during generation. The `MLP` class and the `load_weights` method are modified.

## Usage

### Reproduce the Environment

All Python dependencies (with exact versions used for the paper) are pinned in `requirements.txt`. Set up the environment with:

```shell
conda create --name expenv0 python=3.10
conda activate expenv0
pip install -r requirements.txt
```

> The pinned stack targets CUDA 12.4; `requirements.txt` carries the matching `--extra-index-url` for the CUDA builds of `torch`/`vllm`. The optimized CUDA kernels are built separately (see [Compile CUDA Kernels](#compile-cuda-kernels-for-sparse-ffn)).

For UltraEval to work properly, follow the corresponding instructions:

```shell
pip install -r UltraEval/requirements.txt
cd UltraEval
# Download the benchmark data archive (RawData.zip). Mirror provided by UltraEval:
curl -L -o RawData.zip "https://cloud.tsinghua.edu.cn/f/11d562a53e40411fb385/?dl=1"
unzip RawData.zip
python data_process.py
```

> A Google Drive mirror of `RawData.zip` is also available; see `UltraEval/README.md`.

### Construct SVD-Predictors

Obtain calibration prompts with `python3 utils/get_calibration_prompts.py --output_path calibration_prompts.json`.

Weights for SVD predictors are stored in separate `pt` files. The format is in line with the PowerInfer [predictors format](https://huggingface.co/Tiiny/prosparse-llama-2-7b-predictor), so you can use the constructed SVD predictors in the [PowerInfer framework](https://github.com/Tiiny-AI/PowerInfer) as well!

```shell
python3 utils/construct_predictors.py \
    --model_path path/to/model/weights \
    --predictors_output_path weights/predictors \
    --rank 256 \
    --s 0.5 \
    --sparsity_plot_output_file outputs/sparsity.pdf \
    --calibration_prompts_path calibration_prompts.json \
    --torch_dtype float16 \
    --device_map cuda:0
```

#### Troubleshooting

The construction script uses `transformers` to obtain hidden activation statistics for calibration. Therefore, a proper Hugging Face model implementation is required. You might need to handle incompatibilities, e.g. with the `huggingface KV Cache API`. Make sure that the model implementation works correctly (e.g. that the double ReLU is present in the FFN block).

### Compile CUDA Kernels For Sparse FFN

You need to specify [compute capability](https://developer.nvidia.com/cuda/gpus) of your GPU and LLM architecture (you can compile kernels for all models).

```shell
cd kernels
export LLM_ARCH=llama2 # one of "llama2", "qwen2", "mistral"
CUDA_ARCH_LIST=86 python3 setup.py install
```

The performance may be suboptimal on some devices. You might want to play with `NUM_THREADS`, `BLOCK_SIZE_X`,`BLOCK_SIZE_Y` hyperparameters in this case.

#### Troubleshooting

- Make sure that the CUDA version matches the version that was used to compile PyTorch. You can check the CUDA version via the nvcc compiler:

    ```shell
    nvcc --version
    ```

    To use the same CUDA version for compilation (e.g. 12.6):

    ```shell
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-12.6/bin:$PATH"
    ```

- gcc should likely be 9.0 or higher

    ```shell
    export CC=/path/to/gcc
    export CXX=/path/to/g++
    ```

### Check Sanity

Run the model on a single prompt for manual sanity check, e.g.

```shell
python3 utils/sanity.py \
    --model_path path/to/model/weights \
    --vllm_module_path vllm_implementations/sparse_mistral_baseline.py \
    --start_prompt "hello, what year is it today?" \
    --temperature 0.0
```

When running models with predictors, you additionally have to set the `PREDICTORS_PATH` environment variable to specify the path to the predictors' weights, e.g.

```shell
export PREDICTORS_PATH=weights/predictors
```

### Evaluate Quality

We use the UltraEval framework for evaluation, just as ProSparseLLaMA's authors did. We have modified the original source code a bit (updated generation configs, removed randomness in few-shot sampling, etc.). The following shell script takes 5 positional arguments: implementation path, model path, benchmarks, port and tag name. Benchmarks have to be listed with commas and without spaces, e.g. `mbpp,triviaqa,arc-c,arc-e,gsm8k,bbh,humaneval,cmmlu`.

```shell
export MODEL_PATH=path/to/model/weights
export IMPL_NAME=sparse_llama_baseline
export BENCHMARKS=triviaqa,arc-e,arc-c,humaneval,mbpp,gsm8k,bbh,cmmlu
export DEPLOY_PORT=5089
export TAG_NAME=baseline
./UltraEval/evaluate.sh vllm_implementations/${IMPL_NAME}.py ${MODEL_PATH} ${BENCHMARKS} ${DEPLOY_PORT} ${TAG_NAME}
```

This script will deploy the specified vLLM model locally (via Flask) on `DEPLOY_PORT` and evaluate its quality on the specified benchmarks. Logs of the deployed model will be written to a `server-port${DEPLOY_PORT}.log` file in the current directory. The evaluation results will be stored in an `outputs/quality/${TAG_NAME}` folder in the current directory.

> [!WARNING]
> This command creates 2 processes. One of them makes requests and waits for responses, the other one sends responses. In case an error appears or something else goes wrong, make sure that the corresponding processes are killed. You can use `kill -9 PID`, where the PIDs will be noted in the logs.

> [!TIP]
> `triviaqa,arc-c,arc-e,humaneval` are the fastest (several minutes) benchmarks while `gsm8k,bbh,cmmlu,mbpp` are the slowest (can take more than 1 hour).

Note that processing the first sample takes more time due to model deployment. The running status can be observed in the `server-port${DEPLOY_PORT}.log` file.

### Evaluate Performance

We use the PyTorch profiler for performance evaluation.
The `performance/data` folder contains datasets from different domains, with 5 prompts in each dataset. You can use your own files with the same json structure for performance evaluation as well.

Profile the desired model implementation on the desired dataset

```shell
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATASET=arc-e # any of mbpp, triviaqa, arc-c, arc-e, gsm8k, bbh, humaneval, cmmlu
export VLLM_USE_V1=0
export PREDICTORS_PATH=path/to/predictors
export MODEL_PATH=path/to/model/weights
export IMPL_NAME=sparse_llama_baseline
python3 utils/profiling.py \
    --model ${MODEL_PATH} \
    --metrics outputs/performance/${IMPL_NAME}/${DATASET}.json \
    --vllm_model_modulename vllm_implementations/${IMPL_NAME}.py \
    --num_prompts 5 \
    --dataset-path performance/data/${DATASET}.json \
    --prompt-len 1000 \
    --generation_length 200 \
    --trust_remote_code \
    --dtype half \
    --enforce-eager \
    --max_model_len 1201 \
    --gpu_memory_utilization 0.9
```

The results will be written into a json file specified by the `metrics` flag.
Profiling a single prompt should take a few minutes.

View the performance results

```shell
python3 utils/print_stats.py outputs/performance/${IMPL_NAME}/${DATASET}.json
```

### Reproduce the Paper Results

#### Ablation Study

For ROC AUC comparison use `notebooks/roc_auc_figures.ipynb` notebook. Use the `--test` option in `utils/get_calibration_prompts.py` to obtain test prompts.

Step-by-step adaptation:

```shell
export CUDA_VISIBLE_DEVICES=0
export MODEL_PATH=path/to/model/weights
export MODEL_NAME=sparse_llama
export DEPLOY_PORT=5096
export BENCHMARKS=triviaqa,arc-e,arc-c,humaneval,mbpp,gsm8k,bbh,cmmlu
python3 ./utils/run_ablation.py
```

#### Sparsity-Accuracy Trade-off

For quality:

The `MODEL_PATH` values below assume the model weights live under a `models/` directory next to the repository (`../models/...`). Point `MODELS_DIR` at wherever you downloaded the weights (the `scripts/*.sh` drivers read the same variable, default `../models`).

```shell
export CUDA_VISIBLE_DEVICES=0
export DEPLOY_PORT=5089

# Base directory holding the downloaded model weights.
export MODELS_DIR=path/to/models

export RANK=256
export MODEL_PATH=${MODELS_DIR}/SparseLLM/prosparse-llama-2-7b
export MODEL_NAME=sparse_llama
./scripts/main_quality.sh

export RANK=352
export MODEL_PATH=${MODELS_DIR}/Tiiny/TurboSparse-Mistral-Instruct
export MODEL_NAME=sparse_mistral
./scripts/main_quality.sh

export RANK=512
export MODEL_PATH=${MODELS_DIR}/Tiiny/SparseQwen2-7B
export MODEL_NAME=sparse_qwen2
./scripts/main_quality.sh
```

For performance:

```shell
export CUDA_VISIBLE_DEVICES=0

# Base directory holding the downloaded model weights.
export MODELS_DIR=path/to/models

export RANK=256
export MODEL_PATH=${MODELS_DIR}/SparseLLM/prosparse-llama-2-7b
export MODEL_NAME=sparse_llama
./scripts/main_performance.sh

export RANK=352
export MODEL_PATH=${MODELS_DIR}/Tiiny/TurboSparse-Mistral-Instruct
export MODEL_NAME=sparse_mistral
./scripts/main_performance.sh

export RANK=512
export MODEL_PATH=${MODELS_DIR}/Tiiny/SparseQwen2-7B
export MODEL_NAME=sparse_qwen2
./scripts/main_performance.sh
```

#### Baseline Comparisons

- Comparison with PowerInfer predictors
  - Download [PowerInfer predictors](https://huggingface.co/Tiiny/prosparse-llama-2-7b-predictor) and use `vllm_implementations/sparse_llama_trainable.py` as a vLLM implementation, utilizing previously described scripts for evaluation and profiling. Architecturally, SVD predictors differ from PowerInfer by the absence of ReLU and the presence of bias.
- Comparison with Deja Vu and Polar Sparsity
  - Follow their official github repositories to train predictors. Similarly to PowerInfer, use `vllm_implementations/sparse_llama_trainable.py` but make sure that predictor's architecture in the implementation file matches yours.
- Comparison with GRIFFIN
  - Use `vllm_implementations/sparse_llama_griffin.py` implementation
- FFN Speedups at different sparsity ratios

    ```shell
    python3 kernels/tests/test_gpu.py
    ```

## Running on NPU

### Implementations

We have 2 torch.npu implementations in `npu/kernel.py`:

1. `FeedForward` — the default implementation with dense computations. The implementation does not utilize sparsity in FFN blocks.
2. `SparseFeedForward` — the **optimized** version. It uses SVD predictors to identify the sparse pattern.

### Evaluate Quality

We use the PyTorch profiler for performance evaluation.
To run a test of both solutions on the shapes of 3 datasets, call

```shell
cd npu
bash run.sh
```

The results will be written into csv files in folders named `perf_$SOLUTION_$MODEL_$SPARSITY_LEVEL`.
In each folder, the most interesting files are `op_statistics.csv` and `kernel_details.csv`.

To analyze the time performance use the `csv_parse.py` script — it prints the total and mean time of all operators from a specific column.

```shell
python3 csv_parse.py [csv_file] [column_name]
```

## Citation

```
@misc{serbin2026svdcontextualsparsitypredictors,
      title={SVDP: Training-Free Contextual Sparsity Predictors for Fast LLM Inference}, 
      author={Georgii Serbin and Kirill Koshkin and Zhongao Sun and Anastasiya Bistrigova and C. C. Korikov},
      year={2026},
      eprint={2603.14110},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.14110}, 
}
```

## License

This project is released under the [MIT License](LICENSE).

Third-party components are licensed separately: the vendored evaluation framework under `UltraEval/` (OpenBMB, Apache-2.0) and the model implementations under `vllm_implementations/` (adapted from vLLM / HuggingFace Transformers, Apache-2.0).
