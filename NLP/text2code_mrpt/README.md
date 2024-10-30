# Text-to-Code Generation with Modality-relative Pre-training


Official implementation for the EACL 2024 paper 
"[Text-to-Code Generation with Modality-relative Pre-training](https://aclanthology.org/2024.eacl-long.72/)".

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
Click [here](https://drive.google.com/file/d/1PtskeJAKor8sbXVXEQtQ9JVB2W2XxqhX/view?usp=drive_link) to download it.
Alternative, `setup.py` includes all the necessary dependencies.
To install the directory as a package, run
```bash
pip install -e .
```

## Training Data

We make available the data from Github that we use to train PyCodeGPT and PanGu-Coder (approximately 23M text-to-code pairs) on the 
HuggingFace Hub:  
[https://huggingface.co/datasets/huawei-noah/python_text2code](https://huggingface.co/datasets/huawei-noah/python_text2code)

In order to optimize training, we concatenated training instances, resulting in a meta-instance that was fed into 
the model. So that consecutive training instances do not contaminate one another, we edit attention masks and 
absolute positions accordingly.
The data are pre-tokenized first and then concatenated.
To convert the training data into a concatenated format, run the following (or use `concat.sh`) with the separation strategy that you want:

```bash
python sample_concatenation.py \
--main_dir=directory_where_the_project_resides/ \
--dataset_dir=path_to_python_data/python_github_text2code \
--max_seq_length=1024 \
--tokenizer=pycodegpt \
--model_name_or_path=path_to_pretrained_model \
--save_name=pycodegpt_partial_sep \
--separate_some_embeds="python_tokens.txt" # or --separate_embeds=True
```


## Modality-relative Pre-training
The following arguments control the modality-relative pre-training objectives:

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

You can run `run_model.sh` by appending the necessary arguments for each objective described in the above table.


## Released Models

We release 6 models, trained on the text-to-code paired data, based on the CodeCLM objective:
- PyCodeGPT:
  - https://huggingface.co/huawei-noah/pycodegpt-CodeCLM-100m
  - https://huggingface.co/huawei-noah/pycodegpt-CodeCLM-partial-100m
  - https://huggingface.co/huawei-noah/pycodegpt-CodeCLM-full-100m
- PanGu:
  - https://huggingface.co/huawei-noah/pangu-CodeCLM-300m
  - https://huggingface.co/huawei-noah/pangu-CodeCLM-partial-300m
  - https://huggingface.co/huawei-noah/pangu-CodeCLM-full-300m


## Evaluation

In the paper, we evaluated models on [HumanEval](https://github.com/openai/human-eval) and 
[MBPP](https://github.com/google-research/google-research/tree/master/mbpp) based on functional correctness.
Generations are obtained with the `geneval.sh` script and executed with the [CodeGeeX](https://github.com/THUDM/CodeGeeX) framework.  

```bash
# Get CodeGeeX
git clone https://github.com/THUDM/CodeGeeX.git
cd CodeGeeX && pip install -e .
```

```python
# Download the MBPP test set (make sure it gets saved inside the CodeGeeX directory)
from datasets import load_dataset
ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
ds.to_json("mbpp_test.jsonl")
```

In addition, place the files inside `codegeex_changes` into the official CodeGeeX folder:
- `cp codegeex_changes/codegeex/benchmark/humaneval-x/evaluate_humaneval_x.py path_to_codegeex/codegeex/benchmark/humaneval-x/`
- `cp -r codegeex_changes/codegeex/benchmark/mbpp/ path_to_codegeex/codegeex/benchmark/`
- `cp codegeex_changes/scripts/evaluate_mbpp.sh path_to_codegeex/scripts/`

Example runs:
```bash
cd source

# greedy decoding
bash geneval.sh -cgxp path_to_codegeex -mf pycodegpt -mp path_to_model/pycodegpt-CodeCLM-partial-100m/ -dat humaneval -greedy True 

# sampling
bash geneval.sh -cgxp path_to_codegeex -mf pycodegpt -mp path_to_model/pycodegpt-CodeCLM-partial-100m/ -dat mbpp             

# for incremental pass@k
bash geneval.sh -cgxp path_to_codegeex -mf pycodegpt -mp path_to_model/pycodegpt-CodeCLM-partial-100m/ -dat humaneval -greedy True -incr True  
```


## License

We follows Apache License Version 2.0. Please see the [License](./LICENSE) file for more information.

Disclaimer: This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.
