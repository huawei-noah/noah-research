import sys
import os
from mmcv.runner import obj_from_dict
from torch import nn
import pdb

from . import (roi_extractors)
from . import (bbox_heads)

__all__ = [
    'build_roi_extractor', 'build_bbox_head'
]


def _build_module(cfg, parrent=None, default_args=None):
    return cfg if isinstance(cfg, nn.Module) else obj_from_dict(
        cfg, parrent, default_args)


def build(cfg, parrent=None, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, parrent, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, parrent, default_args)


def build_roi_extractor(cfg):
    return build(cfg, roi_extractors)


def build_bbox_head(cfg):
    return build(cfg, bbox_heads)

