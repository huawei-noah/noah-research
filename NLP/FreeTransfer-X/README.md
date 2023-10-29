# FreeTransfer-X: Safe and Label-Free Cross-Lingual Transfer from Off-the-Shelf Models

[**Abstract**](#abstract) | [**Datasets**](#datasets) |
[**Two-Step Knowledge Distillation**](#two-step-knowledge-distillation) |

This repository contains the implementation of the NAACL 2022 paper [FreeTransfer-X: Safe and Label-Free Cross-Lingual Transfer from Off-the-Shelf Models](https://arxiv.org/pdf/2206.06586.pdf).

# Abstract

Cross-lingual transfer (CLT) is of various applications. However, labeled cross-lingual corpus is expensive or even inaccessible, especially in the fields where labels are private, such as diagnostic results of symptoms in medicine and user profiles in business. Nevertheless, there are off-the-shelf models in these sensitive fields. Instead of pursuing the original labels, a workaround for CLT is to transfer knowledge from the off-the-shelf models without labels. To this end, we define a novel CLT problem named FreeTransfer-X that aims to achieve knowledge transfer from the off-the-shelf models in rich-resource languages. To address the problem, we propose a 2-step knowledge distillation (KD, Hinton et al., 2015) framework based on multilingual pre-trained language models (mPLM). The significant improvement over strong neural machine translation (NMT) baselines demonstrates the effectiveness of the proposed method. In addition to reducing annotation cost and protecting private labels, the proposed method is compatible with different networks and easy to be deployed. Finally, a range of analyses indicate the great potential of the proposed method.

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

【This open source project is not an official Huawei product, Huawei is not expected to provide support for this project.】