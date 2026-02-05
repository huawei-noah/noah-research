# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
import torch.nn as nn

from .base import BaseObserver


class PercentileObserver(BaseObserver):

    def __init__(self,
                 module_type,
                 bit_type,
                 calibration_mode,
                 shape=None,
                 percentile_sigma=0.01,
                 percentile_alpha=0.99999):
        super(PercentileObserver, self).__init__(module_type, bit_type,
                                                 calibration_mode)
        self.percentile_sigma = 0.01
        self.percentile_alpha = 0.99999
        self.symmetric = self.bit_type.signed

    def update(self, v):
        # channel-wise needs too much time.
        assert self.calibration_mode == 'layer_wise'
        v = self.reshape_tensor(v)
        try:
            cur_max = torch.quantile(v.reshape(-1), self.percentile_alpha)
            cur_min = torch.quantile(v.reshape(-1),
                                     1.0 - self.percentile_alpha)
        except:
            cur_max = torch.tensor(np.percentile(
                v.reshape(-1).cpu(), self.percentile_alpha * 100),
                                   device=v.device,
                                   dtype=torch.float32)
            cur_min = torch.tensor(np.percentile(
                v.reshape(-1).cpu(), (1 - self.percentile_alpha) * 100),
                                   device=v.device,
                                   dtype=torch.float32)
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = self.max_val + \
                self.percentile_sigma * (cur_max - self.max_val)
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = self.min_val + \
                self.percentile_sigma * (cur_min - self.min_val)

    def get_quantization_params(self, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        if self.symmetric:
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            scale.clamp_(self.eps)
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
        else:
            scale = (max_val - min_val) / float(qmax - qmin)
            scale.clamp_(self.eps)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
        return scale, zero_point
