# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""
Training and evaluation script for LikeLihood Ratio model.
"""

from functools import partial

import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import MLFlowLogger
from pytorch_lightning.callbacks import EarlyStopping

from lib.datasets.datasets import get_dataset_simple, collate_fn_simple as collate_fn
from lib.modules.likelihood_ratio import LikelihoodratioModule
from lib.utils import get_device_from_config


@hydra.main(config_path='../configs/config_likelihood_clinc.yaml', strict=False)
def main(config):
    device = get_device_from_config(config=config)

    training_config = config.training
    datasets, vocab = get_dataset_simple(config.dataset.pop('name').value(),
                                         add_valid_to_vocab=config.add_valid_to_vocab,
                                         add_test_to_vocab=config.add_test_to_vocab,
                                         **config.dataset
                                         )

    module = LikelihoodratioModule(config, vocab=vocab,
                                   collate_fn=partial(collate_fn,
                                                      pad_idx=vocab.pad_idx,
                                                      bos_idx=vocab.bos_idx,
                                                      eos_idx=vocab.eos_idx
                                                      ),
                                   train_dataset=datasets['train'],
                                   val_dataset=datasets['val'],
                                   test_dataset=datasets['test'])
    mlflow_logger = MLFlowLogger(config.experiment_name, tracking_uri=None)

    trainer = Trainer(
        gpus=device,
        logger=mlflow_logger,
        early_stop_callback=EarlyStopping(**training_config.early_stop),
        max_epochs=training_config.max_epochs,
        gradient_clip_val=training_config.get('gradient_clip_val', 0),
        checkpoint_callback=False
    )
    trainer.fit(module)
    trainer.test()


if __name__ == '__main__':
    main()
