## HumanRankEval: Automatic Evaluation of LMs as Conversational Assistants

#### The repository is based on [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness), big thanks!

This project provides a framework to evaluate generative language models (Seq2seq also supported by AutoHF) on HumanRankEval (HRE).
If you find it helpful, please cite the **HumanRankEval** [paper](https://aclanthology.org/2024.naacl-long.456/).

- Supported Topics: **python, java, unix, cpp, html, english, physics, latex, soft_eng, stats, cs_db, languages_sciences, apple_android, math**
- Supported Models: **AutoHF (single and multi-gpu runs implemented, see below)**
- Supported Deepspeed Inference: **Tensor Parallel, Kernel Injection and/or DS ZeRO3**

### Installation (PyTorch)

Create an environment with conda or virtualenv and then run the following command:

```bash
pip install -r requirements.txt
```

### Installation (MindSpore)

You **additionally** need to install [MindSpore](https://www.mindspore.cn/install/en) and [MindNLP](https://github.com/mindspore-lab/mindnlp). 
We provide an example in ```lm_eval.models.mindspore``` for OPT (facebook) models that can be extended to additional LLMs.

### Dataset

The HRE dataset is hosted on [HuggingFace Datasets](https://huggingface.co/datasets/huawei-noah/human_rank_eval). 
Download with: ```load_dataset("huawei-noah/human_rank_eval")```, then save it to disk for the next step, please.

### Running HumanRankEval

Set the **MODEL_DIR=/your/path/to/models/**

Set the **DATA_PATH=/your/path/to/HumanRankEval/**

> 💡 Check out ```evaluate.sh``` for full details 💡
> 
The following command runs Pythia-410M on HRE on gpu:2 (see **evaluate.sh**):
```
deepspeed --include localhost:2 main.py \
          --model auto_hf \
          --tasks human_rank_eval_* \
          --model_args pretrained=${MODEL_DIR}Pythia-410M \
          --batch_size 8 \
          --data_path ${DATA_PATH}
```

The output should look like this:

|               Task               |   Metric    |Value |
|----------------------------------|-------------|-----:|
|human_rank_eval_apple_android     |pearson_corr |0.0860|
|human_rank_eval_cpp               |pearson_corr |0.1351|
|human_rank_eval_cs_db             |pearson_corr |0.0646|
|human_rank_eval_english           |pearson_corr |0.1193|
|human_rank_eval_html              |pearson_corr |0.1055|
|human_rank_eval_java              |pearson_corr |0.1044|
|human_rank_eval_languages_sciences|pearson_corr |0.1201|
|human_rank_eval_latex             |pearson_corr |0.1648|
|human_rank_eval_math              |pearson_corr |0.1405|
|human_rank_eval_physics           |pearson_corr |0.1118|
|human_rank_eval_python            |pearson_corr |0.0778|
|human_rank_eval_soft_eng          |pearson_corr |0.0769|
|human_rank_eval_stats             |pearson_corr |0.1100|
|human_rank_eval_unix              |pearson_corr |0.0967|
|=== HumanRankEval Score ===       |Micro Average|0.1081|

The following command runs Vicuna-7B on HRE on all gpus with tensor parallel (default).
```bash
deepspeed --num_gpus ${NUM_GPUs} main.py \
          --model auto_hf \
          --tasks human_rank_eval_* \
          --model_args pretrained=${MODEL_DIR}Vicuna-7B \
          --data_path ${DATA_PATH} \
          --batch_size 4 \
          --world_size ${NUM_GPUs}
```
The output should look like this:

|               Task               |   Metric    |Value |
|----------------------------------|-------------|-----:|
|human_rank_eval_apple_android     |pearson_corr |0.1310|
|human_rank_eval_cpp               |pearson_corr |0.1657|
|human_rank_eval_cs_db             |pearson_corr |0.1043|
|human_rank_eval_english           |pearson_corr |0.1468|
|human_rank_eval_html              |pearson_corr |0.1430|
|human_rank_eval_java              |pearson_corr |0.1670|
|human_rank_eval_languages_sciences|pearson_corr |0.1571|
|human_rank_eval_latex             |pearson_corr |0.1743|
|human_rank_eval_math              |pearson_corr |0.1257|
|human_rank_eval_physics           |pearson_corr |0.1114|
|human_rank_eval_python            |pearson_corr |0.1402|
|human_rank_eval_soft_eng          |pearson_corr |0.0962|
|human_rank_eval_stats             |pearson_corr |0.1629|
|human_rank_eval_unix              |pearson_corr |0.1289|
|=== HumanRankEval Score ===       |Micro Average|0.1396|

Evaluating a MindSpore model on a single topic can be done as follows:

```bash
python main.py --model mindspore \
               --tasks human_rank_eval_math \
               --data_path ${DATA_PATH} \
               --model_args pretrained=opt-350m \
               --batch_size 4
```

You should see the following output:

|           Task            |   Metric    |Value|
|---------------------------|-------------|----:|
|human_rank_eval_math       |pearson_corr |0.078|
|=== HumanRankEval Score ===|Micro Average|0.078|

## License

We follow MIT license. Please see the [License](./LICENSE) file for more information.

Disclaimer: This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.
