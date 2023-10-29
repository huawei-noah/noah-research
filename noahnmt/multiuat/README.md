# Uncertainty-Aware Balancing for Multilingual and Multi-Domain Neural Machine Translation Training

This is the official repo for the EMNLP 2021 paper "Uncertainty-Aware Balancing for Multilingual and Multi-Domain Neural Machine Translation Training".

## Installation

```
# clone this repo to your local directory
conda create -n multiuat python=3.6
conda activate multiuat
cd multiuat/fairseq
pip install --editable ./
```

## Data Preprocessing

### Multi-Domain Preprocessing

1. Download, tokenize and BPEize the corpora

   ```
   mkdir corpus/multidomain
   # cp all the scripts in multidomain-preprocess to corpus/multidomain
   bash prepare-corpora.sh
   ```

2. Fairseq-preprocessing

   ```
   # pwd = corpus/multidomain
   # preprocessing the concatenated corpus to get the dictionaries
   fairseq-preprocess \
   	--source-lang en --target-lang de \
   	--joined-dictionary --workers 16 \
   	--trainpref concat/train \
   	--validpref concat/valid \
   	--testpref concat/test \
   	--destdir concat-bin
   # preprocessing the dataset for each domain. WMT as an example.
   fairseq-preprocess \
   	--source-lang en --target-lang de \
   	--srcdict concat-bin/dict.en.txt \
   	--tgtdict concat-bin/dict.de.txt \
   	--destdir wmt14_en_de-bin
   ```

### Multilingual Preprocessing

You can download the preprocessed data from [here](https://drive.google.com/file/d/1xNlfgLK55SbNocQh7YpDcFUYymfVNEii/view?usp=sharing)

## Training

All the scripts used for training are in the directory `jobscripts`. Some paths are hard-coded, please update the paths before running those scripts.

For example,

```
bash jobscripts/multiuat/train-multiuat-multidomain.sh [SRC] [TGT] [DOMAIN_PATH] [DATA_PATH] [TEMP] [K] [REWARD] [TAG] [SAVE_DIR]
# [SRC] -- source language
# [TGT] -- target language
# [DOMAIN_PATH] -- in-domain config file ptah, e.g. ende-iid-domain.txt 
# [DATA_PATH] -- directory path of Multi-Domain corpora
# [TEMP] -- temperature
# [K] -- the number of inference pass in Monte Carlo Dropout
# [REWARD] -- type of reward, {enttp,enteos,pretp,exptp,vartp,comtp}
# [TAG] -- unique identifier for your experiment
# [SAVE_DIR] -- directory path to save checkpoints
```

## Evaluation

You can use `evaluate-ckpt-multidomain.py` and `evaluate-ckpt-multilingual.py` to evaluate the checkpoint on multiple datasets with a single command. 

【This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.】