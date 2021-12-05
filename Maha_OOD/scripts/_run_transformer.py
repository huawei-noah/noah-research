# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import inspect
from os import environ

from pytorch_lightning.callbacks import EarlyStopping

import torch
from pytorch_lightning import Trainer
from transformers import BertTokenizer, RobertaTokenizer, DistilBertTokenizer, RobertaModel, DistilBertModel, BertModel
from pytorch_lightning.loggers.mlflow import MLFlowLogger

from configs.config import ExpConfig
from lib.dataclass_utils import datacli
from lib.datasets import datasets
from lib.modules.transformer_classifier import TransformerModule, TransformerClassifier, DistilBertWrapper
import lib.score_functions as score_functions


def bert_collate(bert_tok):
    def collate_fn(batch):
        idxs = [b[0] for b in batch]
        intents = [b[1] for b in batch]
        is_ood = [b[2] for b in batch]
        encoded = bert_tok.batch_encode_plus(idxs,
                                             add_special_tokens=False,
                                             pad_to_max_length=True,
                                             return_tensors='pt', return_token_type_ids=False)
        padded_seq = encoded["input_ids"]
        att_idxs = encoded["attention_mask"]
        return {
            'seq': padded_seq,
            'attention_mask': att_idxs,
            'labels': torch.tensor(intents),
            'is_ood': torch.tensor(is_ood)
        }

    return collate_fn


def get_kwargs_for_mahalanobis_score(score_cls, classifier, config):
    kwargs = {}
    for arg in inspect.signature(score_cls.__init__).parameters:
        if arg == 'dim':
            kwargs[arg] = classifier.transformer.config.hidden_size
        elif arg == 'num_labels':
            kwargs[arg] = config.n_labels
        elif arg == 'start_elem':
            kwargs[arg] = config.start_elem if 'start_elem' in config.__dict__ else config.n_labels
    return kwargs


def get_tokenizer_and_model(model_name):
    if 'roberta' in model_name:
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name)
    elif 'distilbert' in model_name:
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertWrapper(distil_bert_model=DistilBertModel.from_pretrained(model_name))
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    return tokenizer, model


def run_experiment(config):
    tokenizer, transformer = get_tokenizer_and_model(config.bert_type)

    train_dataset, val_dataset, test_dataset = datasets.get_dataset_transformers(
                                                           tokenizer=tokenizer,
                                                           dataset_name=config.data_name,
                                                           data_path=config.data_path,
                                                           version=config.version,
                                                           ood_type=config.ood_type)
    config.n_labels = len(train_dataset.label_vocab)
    transformer_classifier = TransformerClassifier(
        transformer, config.hidden_dropout_prob, config.n_labels
    )

    collate_fn = bert_collate(tokenizer)
    score_cls = score_functions.SCORE_FUNCTION_REGISTRY[config.score_type]
    if issubclass(score_cls, score_functions.LogitsScoreFunction):
        score_function = score_cls(temperature=config.temperature)
    elif issubclass(score_cls, score_functions.AbstractMahalanobisScore):
        kwargs = get_kwargs_for_mahalanobis_score(score_cls, transformer_classifier, config)
        score_function = score_cls(**kwargs)
    else:
        raise ValueError(f'Unknown score class {score_cls}')

    module = TransformerModule(config, transformer_classifier,
                               score_function, collate_fn, tokenizer,
                               train_dataset, val_dataset, test_dataset)
    logger = MLFlowLogger(config.experiment_name)
    trainer = Trainer(
        gpus=1,
        logger=logger,
        max_epochs=config.n_epochs,
        overfit_pct=config.overfit_pct,
        train_percent_check=config.train_percent_check,
        val_percent_check=config.val_percent_check,
        num_sanity_val_steps=0,
        accumulate_grad_batches=config.accumulate_grad_batches,
        checkpoint_callback=config.use_checkpoint,
        early_stop_callback=EarlyStopping(monitor='val_loss', mode='min'),
        gradient_clip_val=config.gradient_clip_val
    )
    trainer.fit(module)
    eval_trainer = Trainer(gpus=1,
                           logger=logger,
                           )
    eval_trainer.test(module)


if __name__ == '__main__':
    config = datacli(ExpConfig)
    environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    run_experiment(config)
