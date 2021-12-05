# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""
Training and evaluation script for classifier based on the CNN or LSTM.
"""

from functools import partial
import inspect

import hydra
from nltk.tokenize import word_tokenize as tokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping

from lib.data_utils import Vocab
from lib.datasets.datasets import get_dataset_simple, collate_fn_simple, collate_fn_bow
from lib.modules.classification import ClassificationModule
from lib.modules.classifiers import LSTMClassifier, CNNClassifier, CBOWClassifier, BOWClassifier
from lib.utils import get_device_from_config
import lib.score_functions as score_functions


def get_kwargs_for_mahalanobis_score(score_cls, classifier):
    kwargs = {}
    for arg in inspect.signature(score_cls.__init__).parameters:
        if arg == 'dim':
            kwargs[arg] = classifier.penultimate_size
        elif arg == 'num_labels':
            kwargs[arg] = classifier.config.num_classes
        elif arg == 'start_elem':
            kwargs[arg] = classifier.config.get('start_elem', classifier.config.num_classes)
    return kwargs


def get_classifier(config, vocab: Vocab):
    if config.classifier.name == 'lstm':
        return LSTMClassifier(config=config, vocab=vocab)
    elif config.classifier.name == 'cnn':
        return CNNClassifier(config=config, vocab=vocab)
    elif config.classifier.name == 'cbow':
        return CBOWClassifier(config=config, vocab=vocab)
    elif config.classifier.name == 'bow':
        return BOWClassifier(config=config, vocab=vocab)
    else:
        raise NotImplementedError


@hydra.main(config_path='../configs/config_msp_cnn.yaml', strict=False)
def main(config):
    device = get_device_from_config(config=config)

    training_config = config.training
    datasets, vocab = get_dataset_simple(config.dataset.pop('name').value(),
                                         add_valid_to_vocab=config.add_valid_to_vocab,
                                         add_test_to_vocab=config.add_test_to_vocab,
                                         tok_fn=tokenizer,
                                         **config.dataset
                                         )
    if config.classifier.name == 'bow':
        collate_fn = partial(collate_fn_bow,
                             vocab_size=len(vocab))
    else:
        collate_fn = partial(collate_fn_simple,
                             pad_idx=vocab.pad_idx,
                             bos_idx=vocab.bos_idx,
                             eos_idx=vocab.eos_idx
                             )

    classifier = get_classifier(config, vocab)

    score_cls = score_functions.SCORE_FUNCTION_REGISTRY[config.score_type]
    if issubclass(score_cls, score_functions.LogitsScoreFunction):
        score_function = score_cls(temperature=config.temperature)
    elif issubclass(score_cls, score_functions.AbstractMahalanobisScore):
        kwargs = get_kwargs_for_mahalanobis_score(score_cls, classifier)
        score_function = score_cls(**kwargs)
    else:
        raise ValueError(f'Unknown score class {score_cls}')

    module = ClassificationModule(config=config,
                                  classifier=classifier,
                                  score_fn=score_function,
                                  collate_fn=collate_fn,
                                  train_dataset=datasets['train'],
                                  val_dataset=datasets['val'],
                                  test_dataset=datasets['test'])

    mlflow_logger = MLFlowLogger(config.experiment_name)
    early_stop = EarlyStopping(**training_config.early_stop) if training_config.use_early_stop else False
    trainer = Trainer(
        gpus=device,
        logger=mlflow_logger,
        early_stop_callback=early_stop,
        max_epochs=training_config.max_epochs,
        checkpoint_callback=config.use_checkpoint,
        gradient_clip_val=config.training.gradient_clip_val
    )
    trainer.fit(module)
    trainer.test()


if __name__ == '__main__':
    main()
