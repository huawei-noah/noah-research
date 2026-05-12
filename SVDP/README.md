<h1>&nbsp;SVD Contextual Sparsity Predictors for Fast LLM Inference</h1>
<h1>
<a href="https://arxiv.org/abs/2603.14110">
  <img src="https://img.shields.io/badge/Arxiv-2603.14110-green.svg"></a> 
</h1>
Up to x1.8 End-to-End generation speedup with near-lossless accuracy degradation.
## Description
This project utilizes contextual sparsity in FFN blocks of ReGLU-based LLMs via training-free predictors to speedup LLM inference.

## Models
Currently 4 LLMs can be tested:
 - [ProSparseLLaMA2-7B](https://huggingface.co/SparseLLM/prosparse-llama-2-7b)
 - [TurboSparse-Mistral-Instruct](https://huggingface.co/Tiiny/TurboSparse-Mistral-Instruct)
 - [SparseQwen2-7B](https://huggingface.co/Tiiny/SparseQwen2-7B)
 - [Bamboo](https://huggingface.co/Tiiny/Bamboo-DPO-v0_1)

 > Bamboo and Mistral share one model implementation

## Contents
 - `kernels` --- custom CUDA kernels for sparse FFN;
 - `outputs` folder is for outputs;
 - `vllm_implementations` --- vLLM model implementations;
 - `performance` folder contains scripts for performance evaluation and some test data samples;
 - `utils` --- python scripts;
 - [UltraEval](https://github.com/OpenBMB/UltraEval) --- framework for evaluation;
 - `npu` --- npu kernels and python scripts of tests and parsing time statistics;


## Implementations
For each model we have 2 vLLM implementations:
1. `vllm_implementations/..._baseline.py` --- the default implementation with dense computations. The code is based on official vLLM implementations. The activation function is replaced with ReLU. The implementation does not utilize sparsity in FFN blocks.
2. `vllm_implementations/..._svd_predictors.py` --- the **optimized** version. It utilizes SVD predictors to identify sparse-pattern. A custom CUDA kernels reduce the number of memory reads during the generation. `MLP` class and `load_weights` method are modified. 


## Usage
### Reproduce the Environment
```shell
conda create --name expenv0 python=3.10
conda activate expenv0
pip install vllm==0.8.5.post1 --extra-index-url https://download.pytorch.org/whl/cu124
```
For UltraEval proper work follow the correspoding instructions
```shell
pip install -r UltraEval/requirements.txt
cd UltraEval
unzip RawData.zip
python data_process.py
```

### Construct SVD-Predictors
Obtain calibration prompts with `python3 utils/get_calibration_prompts.py --output_path calibration_prompts.json`.

Weights for SVD predictors are stored in a separate `pt` files. The format is in line with PowerInfer [predictors format](https://huggingface.co/Tiiny/prosparse-llama-2-7b-predictor), so you can use constructed SVD-predictors in [PowerInfer framework](https://github.com/Tiiny-AI/PowerInfer) as well!
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
The construction script uses `transformers` to obtain hidden activation statistics for calibration. Therefore, proper `hugging face` model implementation is required. You might need to handle incompatabilities e.g. with `huggingface KV Cache API`. Make sure that the model implementation works correctly (e.g. double ReLU is present in FFN block).

### Compile CUDA Kernels For Sparse FFN
You need to specify [compute capability](https://developer.nvidia.com/cuda/gpus) of your GPU and LLM architecture (you can compile kernels for all models).
```shell
cd kernels
export LLM_ARCH=llama2 # one of "llama2", "qwen2", "mistral"
CUDA_ARCH_LIST=86 python3 setup.py install
```
The performance may be suboptimal on some devices. You might want to play with `NUM_THREADS`, `BLOCK_SIZE_X`,`BLOCK_SIZE_Y` hyperparameters in this case.
#### Troubleshooting
 - Make sure that the CUDA version matches version that was used to compile pytorch. You can check CUDA version via nvcc compiler
    ```shell
    nvcc --version
    ```
    To use the same CUDA version for compilation (e.g. 12.6)
    ```shell
    export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
    export PATH="/usr/local/cuda-12.6/bin:$PATH"
    ```
 - Likely gcc should be 9.0 or higher
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
When running models with predictors additionaly you have to set `PREDICTORS_PATH` environment variable to specify path to predictors' weights, e.g.
```shell
export PREDICTORS_PATH=weights/predictors
```

### Evaluate Quality
We use UltraEval framework for evaluation just as ProSparseLLaMA's authors. We have modified the original source code a bit (updated generation configs, removed randomness in fewshot sampling, etc.). The following shell script has 5 positional arguments: implementation path, model path, benchmarks and port. Benchmarks have to be listed with commas without spacing e.g. `mbpp,triviaqa,arc-c,arc-e,gsm8k,bbh,humaneval,cmmlu`

```shell
export MODEL_PATH=path/to/model/weights
export IMPL_NAME=sparse_llama_baseline
export BENCHMARKS=triviaqa,arc-e,arc-c,humaneval,mbpp,gsm8k,bbh,cmmlu
export DEPLOY_PORT=5089
export TAG_NAME=baseline
./UltraEval/evaluate.sh vllm_implementations/${IMPL_NAME}.py ${MODEL_PATH} ${BENCHMARKS} ${DEPLOY_PORT} ${TAG_NAME}
```
This script will deploy specified vLLM model locally (via Flask) on `DEPLOY_PORT` and evaluate its quality on specified benchmarks. Logs of deployed model will be written in a `server-port${DEPLOY_PORT}.log` file of the current directory. The evaluation results will be stored in a `outputs/quality/${TAG_NAME}` folder of the current directory.

> [!WARNING]
> This command creates 2 proccesses. One of them makes requests and waits for responses, the other one send responses. In case error appears or something else went wrong, make sure that the corresponding processes are killed. One can use `kill -9 PID` where PIDs will be noted in logs. 

> [!TIP]
> `triviaqa,arc-c,arc-e,humaneval` are the fastest (several minutes) benchmarks while `gsm8k,bbh,cmmlu,mbpp` are the slowest (can take more than 1 hour).

Note that proccessing of first sample takes more time due to model deploying. The running status can be observed in `server-port${DEPLOY_PORT}.log` file

### Evaluate Performance
We use pytorch profiler for performance evaluation. 
The `performance/data` folder contains datasets of different domains with 5 prompts in each dataset. You can use your own files with the same json structure for performance evaluation as well.

Profile the desired model implementation on the desired dataset
```shell
export CUDA_VISIBLE_DEVICES=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export DATASET=arc-e # any of mbpp, triviaqa, arc-c, arc-e, gsm8k, bbh, humaneval, cmmlu
export VLLM_USE_V1=0
export PREDICTORS_PATH=path/to/predictors
export MODEL_PATH=path/to/model/weights
export IMPL_NAME=sparse_llama_baseline
python3 performance/profiling.py \
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
The results will be writen into json file specified by `metrics` flag.
The profiling on a single prompt should take few minutes.

View the performance results
```shell
python3 utils/print_stats.py outputs/performance/${IMPL_NAME}/${DATASET}.json
```
### Reproduce the Paper Results
 - Ablation Study
    - For ROC AUC comparison use `notebooks/roc_auc_figures.ipynb` notebook. Use `--test` option in `utils/get_calibration_prompts.py` to obtain test prompts.
    - Step-by-step adaptation
        ```shell
        export CUDA_VISIBLE_DEVICES=0
        export MODEL_PATH=path/to/model/weights
        export MODEL_NAME=sparse_llama
        export DEPLOY_PORT=5096
        export BENCHMARKS=triviaqa,arc-e,arc-c,humaneval,mbpp,gsm8k,bbh,cmmlu
        python3 ./utils/run_ablation.py
        ```
 - Sparsity-Accuracy Trade-off
    - For quality
        ```shell
        export CUDA_VISIBLE_DEVICES=0
        export DEPLOY_PORT=5089

        export RANK=256
        export MODEL_PATH=../models/SparseLLM/prosparse-llama-2-7b
        export MODEL_NAME=sparse_llama
        ./scripts/main_quality.sh

        export RANK=352
        export MODEL_PATH=../models/Tiiny/TurboSparse-Mistral-Instruct
        export MODEL_NAME=sparse_mistral
        ./scripts/main_quality.sh

        export RANK=512
        export MODEL_PATH=../models/Tiiny/SparseQwen2-7B
        export MODEL_NAME=sparse_qwen2
        ./scripts/main_quality.sh
        ```
    - For performance
        ```shell
        export CUDA_VISIBLE_DEVICES=0

        export RANK=256
        export MODEL_PATH=../models/SparseLLM/prosparse-llama-2-7b
        export MODEL_NAME=sparse_llama
        ./scripts/main_performance.sh

        export RANK=352
        export MODEL_PATH=../models/Tiiny/TurboSparse-Mistral-Instruct
        export MODEL_NAME=sparse_mistral
        ./scripts/main_performance.sh

        export RANK=512
        export MODEL_PATH=../models/Tiiny/SparseQwen2-7B
        export MODEL_NAME=sparse_qwen2
        ./scripts/main_performance.sh

        ```
 - Comparison with PowerInfer predictors
    - Download [PowerInfer predictors](https://huggingface.co/Tiiny/prosparse-llama-2-7b-predictor) and use `vllm_implementations/sparse_llama_trainable.py` as a vLLM implementation, utilizing previously described scripts for evaluation and profiling. Architecturally, SVD predictors differ from PowerInfer by the absence of ReLU and the presence of bias.
 - Comparison with Deja Vu and Polar Sparisty
    - Follow their official github repositories to train predictors. Similarly to PowerInfer, use `vllm_implementations/sparse_llama_trainable.py` but make sure that predictor's architecture in the implementation file matches yours.
 - Comparison with GRIFFIN
    - Use `vllm_implementations/sparse_llama_griffin.py` implementation
 - FFN Speedups at different sparsity ratios
    ```shell
    python3 kernels/tests/test_gpu.py
    ```

# Running on NPU
## Implementations
We have 2 kernels using torch.npu in `npu/kernels.py`:
1. `FeedForward` --- the default implementation with dense computations. The implementation does not utilize sparsity in FFN blocks.
2. `SparseFeedForward` --- the **optimized** version. It utilizes SVD predictors to identify sparse-pattern. 

## Evaluate Quality
We use pytorch profiler for performance evaluation. 
To run a test of both solutions on shapes of 3 datasets, call

```shell
bash run.sh
```

The results will be writen into csv files into folders with name `perf_$SOLUTION_$MODEL_$SPARSITY_LEVEL`. 
In each folder the most interesting files are `op_statistics.csv` and `kernel_details.csv`. 

To analyze the time performanсe use script:
 `csv_parse.py` --- prints the total and mean time of all operators from a specific one column.

 ```shell
python3 csv_parse.py [csv_file] [column_name]
```
## Citation
```
@misc{serbin2026svdcontextualsparsitypredictors,
      title={SVD Contextual Sparsity Predictors for Fast LLM Inference}, 
      author={Georgii Serbin and Kirill Koshkin and Zhongao Sun and Anastasiya Bistrigova and C. C. Korikov},
      year={2026},
      eprint={2603.14110},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.14110}, 
}
```