# Dual Transfer for Low-Resource Neural Machine Translation

This repository contains code for the following paper:

>Meng Zhang, Liangyou Li, Qun Liu. **Two Parents, One Child: Dual Transfer for Low-Resource Neural Machine Translation**. In Findings of ACL 2021.

This is a reimplementation based on [XLM](https://github.com/facebookresearch/XLM) that should reproduce reasonable results. It only supports shared target transfer with word embeddings as transfer parameters, but extending to other variants in the paper should be straightforward.

## Usage
Please refer to [XLM](https://github.com/facebookresearch/XLM) for general usage like data preparation. Note that dual transfer uses separate vocabularies for source and target languages. This functionality has been supported in this codebase.

Here we describe how to transfer from German->English to Estonian->English as an example. Variables in the scripts should be set accordingly.

1. Pretrain a German BERT.

```
bash train-bert.sh
```

2. Pretrain an Estonian BERT. Its embeddings are trainable, but other parameters are initialized by the German BERT and frozen.

```
bash train-child-bert.sh
```

3. Train the German->English NMT model. The encoder is initialized from the German BERT, and the source side word embeddings are frozen.

```
bash train-parent-mt.sh
```

4. Train the Estonian->English NMT model. The source side word embeddings are initialized from the Estonian BERT, and other parameters are initialized from the German->English NMT model.

```
bash train-child-mt.sh
```
