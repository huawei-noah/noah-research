# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from dataclasses import dataclass


@dataclass
class ExpConfig:
    experiment_name: str = "debug"
    data_name: str = None
    n_labels: int = None
    hidden_dropout_prob: float = 0.1
    bert_type: str = "bert-base-uncased"
    lr: float = 1e-5
    batch_size: int = 32
    n_workers: int = 10
    n_epochs: int = 5
    overfit_pct: float = 0.0
    train_percent_check: float = 1.0
    val_percent_check: float = 1.0
    temperature: float = 1.0
    accumulate_grad_batches: int = 1
    score_type: str = 'max'
    data_path: str = 'data/clinc/data_full.json'
    device: str = ''
    ood_type: str = None
    version: int = None
    balance_classes: bool = False
    gradient_clip_val: float = 0.
    use_checkpoint: bool = False
