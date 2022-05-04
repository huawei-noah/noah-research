# FreeTransfer-X: Safe and Label-Free Cross-Lingual Transfer from Off-the-Shelf Models

[**Abstract**](#abstract) | [**Datasets**](#datasets) |
[**Two-Step Knowledge Distillation**](#two-step-knowledge-distillation) |

This repository contains the implementation of the NAACL 2022 paper [FreeTransfer-X: Safe and Label-Free Cross-Lingual Transfer from Off-the-Shelf Models](https://2022.naacl.org/).

# Abstract

Cross-lingual transfer (CLT) applications in private scenarios face three limitations. 1) Data labels in the scenarios like medical, law and business is private and sensitive. It's difficult to safely leverage the private data to widely benefit the community and industry. 2) It requires expensive annotations in the source language (e.g. English). Previous CLT methods tackle this via learning a language-agnostic representation or generating language-specific corpus via machine translation (MT). In this paper, we define the above limitations as a novel CLT problem named *FreeTransfer-X*: transfer off-the-shelf models into other languages without labels. To address the problem, a 2-step knowledge distillation (KD, Hinton et al.,2015) framework is proposed. Significant improvement over the baselines demonstrates the effectiveness of the proposed framework. Various analyses further indicate the great potential of the proposed method. In addition, the proposed method is easy to be applied to various network structures. 

# Datasets

## [MultiATIS++](https://aclanthology.org/2020.emnlp-main.410.pdf)
MultiATIS++ (Xu et al., 2020) extends the Multilingual ATIS corpus (Upadhyay et al., 2018) to 9 languages across 4 language families, including Indo-European (English, Spanish, German, French, Portuguese and Hindi), Sino-Tibetan (Chinese), Japonic (Japanese) and Altaic (Turkish). It provides annotations for intent recognition (sentence classification) and slot filling (sequence tagging) for each languages. The utterances are professionally translated from English and manually annotated. MultiATIS++ includes 37,084 training examples and 7,859 testing examples.

## [MTOP](https://aclanthology.org/2021.eacl-main.257.pdf)
MTOP (Li et al., 2021) is a recently released multilingual NLU dataset covering 6 languages: English, German, French, Spanish, Hindi, Thai. It’s also manually annotated for intent recognition (sentence classification) and slot filling (sequence tagging). MTOP provides a larger corpus consisting of 104,445 examples, of which 10% is validation set and 20% is testing set.

# Quick Start

The entrance script is `scripts/train_distill_cloud.sh`.

## Two-Step Knowledge Distillation

### Directory Structure

Please prepare data, vocabularies and model checkpoints following:
```
$REPO_TASK: the off-the-shelf monolingual model.
$REPO: working dir for the distillation.
|─ $SEED/download
|  |─ $TASK
|  |  |─ train/dev/test/trans-train/...: training/validation/testing/... data.
|  |  └─ $VOCABS_DIR: vocabulary files.
|  └─ models: HuggingFace-like configuration files of models, also the Pytorch checkpoints of the intermediate mPLM models.
|     └─ $MODELNAME_(SOURCE|INTERMEDIATE|TARGET): HuggingFace-like configuration files of models, also the Pytorch checkpoints of the intermediate mPLM models.
└─ $SEED/outputs: output model files of the distillations.
```
`$MODELNAME_(SOURCE|INTERMEDIATE|TARGET)` are the dirnames of the model configurations, e.g. xlm-roberta-large in terms of HuggingFace model hub. These files should be manually prepared in advance.

### Step 1: Off-the-Shelf Monolingual Source Model -> mPLM Intermediate Model


Take the sentence classification of MTOP for example,
```
bash scripts/train_distill_cloud.sh \
--model_src $MODELNAME_SOURCE \
--model_tgt $MODELNAME_TARGET \
--model_via $MODELNAME_INTERMEDIATE \
--mono_src_dir $MONO_SOURCE_CHKPT \
--rand_init all \
--run_steps 0,1 \
--sent_cls True \
--task mtop-s_cls \
--task_type s-cls \
--train_langs_src en \
--train_langs en,de,es,fr,hi,th \
--tgt_lang_for_test True \
--test_langs en,de,es,fr,hi,th \
--vocab_size $VOCAB_SIZE_TARGET \
--vocab_size_src $VOCAB_SIZE_SOURCE \
--vocabs_dir $VOCABS_DIR
```
Arguments `--task`, `--sent_cls` and `--task_type` specify the task name and task type, i.e. sentence classification and sequence tagging. `--task` is in the format of `${TASK_NAME}-${TASK_TYPE}`.

### Step 2: mPLM Intermediate Model -> Monolingual Target Model

Take the sentence classification of MTOP for example,
```
bash scripts/train_distill_cloud.sh \
--model_src $MODELNAME_SOURCE \
--model_tgt $MODELNAME_TARGET \
--model_via $MODELNAME_INTERMEDIATE \
--multi_dir $MULTI_INTERMEDIATE_CHKPT \
--rand_init all \
--run_steps 3 \
--sent_cls True \
--task mtop-s_cls \
--task_type s-cls \
--train_langs_src en \
--train_langs en,de,es,fr,hi,th \
--tgt_lang_for_test True \
--test_langs en,de,es,fr,hi,th \
--vocab_size $VOCAB_SIZE_TARGET \
--vocab_size_src $VOCAB_SIZE_SOURCE \
--vocabs_dir $VOCABS_DIR
```
The main differences between the args and those of Step 1 are: `--multi_dir` and `--run_steps`.
