# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from argparse import Namespace
from dataclasses import fields
from tempfile import gettempdir
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import WeightedRandomSampler

from lib.metrics import fpr_at_x_tpr, roc_aupr, roc_auc


def compute_l2_penalty(module: torch.nn.Module, l2_coeff: float = 1e-6):
    penalty = 0.0
    for name, param in module.named_parameters():
        if 'weight' in name:
            penalty += param.norm(2).sum()
    return l2_coeff * penalty


def make_hparams(config, delimiter: str = '/'):
    from omegaconf import OmegaConf
    nested_dict = OmegaConf.to_container(config)

    def process_dict(d, res, prefix=''):
        for key, value in d.items():
            if key == 'experiment_name':
                continue
            if isinstance(value, dict):
                process_dict(value, res, prefix=prefix + key + delimiter)
            elif isinstance(value, list):
                res[prefix + key] = str(value)
            else:
                res[prefix + key] = value

    res = {}
    process_dict(nested_dict, res)
    return Namespace(**res)


def config_from_hparams(hparams, delimiter: str = '/'):
    res = {}
    for param, value in hparams.items():
        cur_dict = res
        param_split = param.split(delimiter)
        for split in param_split[:-1]:
            if split in cur_dict:
                next_dict = cur_dict[split]
            else:
                next_dict = {}
                cur_dict[split] = next_dict
            cur_dict = next_dict
        if isinstance(value, str) and value[0] == '[' and value[-1] == ']':
            value = eval(value)
        cur_dict[param_split[-1]] = value
    return OmegaConf.create(res)


def compute_ood_metrics(ood_scores, is_ood, prefix: str = ''):
    assert (not prefix) or prefix.endswith("/")
    return {
        f'{prefix}fpr90_ood': fpr_at_x_tpr(ood_scores, is_ood, 90),
        f'{prefix}fpr95_ood': fpr_at_x_tpr(ood_scores, is_ood, 95),
        f'{prefix}fpr90_in': fpr_at_x_tpr(ood_scores, is_ood, 90, swap_labels=True),
        f'{prefix}fpr95_in': fpr_at_x_tpr(ood_scores, is_ood, 95, swap_labels=True),
        f'{prefix}aupr_ood': roc_aupr(ood_scores, is_ood),
        f'{prefix}aupr_in': roc_aupr(ood_scores, is_ood, swap_labels=True),
        f'{prefix}auroc': roc_auc(ood_scores, is_ood)
    }


def get_device_from_config(config):
    device = config.get('device', None)
    if device is None:
        return device
    if isinstance(device, int):
        return [device]
    if isinstance(device, (str, list)):
        return device
    raise ValueError('Wrong type of device instance.')


class TmpFile:
    def __init__(self, file_name: str, tmp_dir: Path = None, suffix: str = ''):
        if tmp_dir is None:
            tmp_dir = Path(gettempdir())
        if '.' not in suffix:
            suffix = '.' + suffix
        self.file_path = tmp_dir / f"{file_name}{suffix}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file_path.unlink()


def get_dataclass_params(datacls):
    param_fields = fields(datacls)
    return {f.name: getattr(datacls, f.name) for f in param_fields}


def get_weighted_sampler(dataset):
    target = dataset.vectorized_labels
    class_sample_count = np.unique(target, return_counts=True)[1]
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight).double()
    return WeightedRandomSampler(samples_weight, len(samples_weight))
