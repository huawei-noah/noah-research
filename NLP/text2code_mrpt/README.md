# Text-to-Code Generation with Modality-related Pre-training


Official implementation for the EACL 2024 paper 
["Text-to-Code Generation with Modality-related Pre-training"](https://aclanthology.org/2024.eacl-long.72/).

If you like this work or plan to use it, please cite the publication as follows:
```html
@inproceedings{christopoulou-etal-2024-text,
    title = "Text-to-Code Generation with Modality-relative Pre-training",
    author = "Christopoulou, Fenia  and
      Zhang, Guchun  and
      Lampouras, Gerasimos",
    editor = "Graham, Yvette  and
      Purver, Matthew",
    booktitle = "Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2024",
    address = "St. Julian{'}s, Malta",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.eacl-long.72",
    pages = "1194--1208"
}
```


## Environment Setup

We provide a docker image with the specifics of the environment used for the experiments.
```bash
docker load < text2code_mrpt_img.tar.gz

docker run \
--name mrpt \
-e TERM=xterm-256color \
--gpus="all" \
-ti --rm -d \
--mount type=bind,src="/path_to_the_folder/text2code_mrpt/",target=/workspace/text2code_mrpt/ \
text2code_mrpt:latest
```

## Training Data

We provide the data from Github that we use to train PyCodeGPT and PanGu-Coder. They are available on the 
HuggingFace Hub:
[https://huggingface.co/datasets/huawei-noah/python_github_text2code](https://huggingface.co/datasets/huawei-noah/python_github_text2code)

In order to optimize training, we concatenated training instances, resulting in a meta-instance that was fed into 
the model. So that consecutive training instances do not contaminate one another, we edit attention masks and 
absolute positions accordingly.
The data are pre-tokenized first and then concatenated.
To convert the training data into a concatenated format, run the following selecting (or use `concat.sh`) the separation strategy that you want (look at the comments):

```bash
python sample_concatenation.py \
--main_dir=directory_where_the_project_resides/ \
--dataset_dir=path_to_python_data/python_github_text2code \
--max_seq_length=1024 \
--tokenizer=pycodegpt \  # or pangu
--model_name_or_path=/nfs/aiml2/nlp_team/fenia/MRPT/stage1_trained_models_100M/pycodegpt \
--save_name=pycodegpt_partial_sep \
--separate_some_embeds="python_tokens.txt"
#--separate_embeds=True
```


## Modality-relative Pre-training
There are the following arguments, that control the modality-relative pre-training objectives, summarised in the 
table below:

| Objective        | Separation | Additional Arguments                                                                      | 
|------------------|------------|-------------------------------------------------------------------------------------------|
| Text-Code CLM    | -          | -                                                                                         |
|                  | partial    | --separate_some_embeds="python_tokens.txt"                                                |
|                  | full       | --separate_embeds=True                                                                    |
| Code CLM         | -          | --predict_code=True                                                                       |
|                  | partial    | --predict_code=True, --separate_some_embeds="python_tokens.txt"                           |
|                  | full       | --predict_code=True, --separate_embeds=True                                               |
| Corrupt Code CLM | -          | --predict_code=True, --corrupt_docstring=True                                             |
|                  | partial    | --predict_code=True, --corrupt_docstring=True, --separate_some_embeds="python_tokens.txt" |
|                  | full       | --predict_code=True, --corrupt_docstring=True, --separate_embeds=True                     |
| Prefix Code CLM  | -          | --predict_code=True, --prefix_lm=True                                                     |
|                  | partial    | --predict_code=True, --prefix_lm=True, --separate_some_embeds="python_tokens.txt"         |
|                  | full       | --predict_code=True, --prefix_lm=True, --separate_embeds=True                             |

You can run the `run_model.sh` script by appending the necessary additional arguments for each objective described in the above table.


## Evaluation

In the paper, we evaluated models on [HumanEval](https://github.com/openai/human-eval) and 
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp) based on functional correctness, using the [CodeGeeX](https://github.com/THUDM/CodeGeeX) framework.  
We provide simpler code (compared to the framework) for generation via the `run_generation_greedy.sh` and `run_generation_samples.sh` scripts.
For further details on evaluation, check the [evaluation_guide](evaluation_guide.md) guide.

For measuring **incremental pass@k** we provide the augmented HumanEval and BPP datasets.


## Released Models

We release 6 models, trained on the text-to-code paired data, based on the CodeCLM objective:
- PyCodeGPT:
  - https://huggingface.co/huawei-noah/pycodegpt-CodeCLM-100m
  - https://huggingface.co/huawei-noah/pycodegpt-CodeCLM-partial-100m
  - https://huggingface.co/huawei-noah/pycodegpt-CodeCLM-full-100m
- PanGu:
  - https://huggingface.co/huawei-noah/pangu-CodeCLM-300M
  - https://huggingface.co/huawei-noah/pangu-CodeCLM-partial-300m
  - https://huggingface.co/huawei-noah/pangu-CodeCLM-full-300m


## License

We follows Apache License Version 2.0. Please see the [License](./LICENSE) file for more information.

Disclaimer: This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.
