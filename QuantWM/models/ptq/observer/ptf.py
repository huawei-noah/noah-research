# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss


class PtfObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode,shape=None):
        super(PtfObserver, self).__init__(module_type, bit_type,
                                          calibration_mode)

    def update(self, v):
        v = self.reshape_tensor(v)
        cur_max = v.max(axis=1).values
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()

    def get_quantization_params(self, inputs, *args, **kwargs):
        max_val = self.max_val
        min_val = self.min_val

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        best_score = 1e+10
        max_val_t = max_val.max()
        min_val_t = min_val.min()
        scale8 = (max_val_t - min_val_t) / float(qmax - qmin)
        scale8.clamp_(self.eps)
        scale4 = scale8 / 2
        scale2 = scale4 / 2
        scale1 = scale2 / 2
        zero_point = qmin - torch.round(min_val_t / scale8)
        zero_point.clamp_(qmin, qmax)
        scale_mask = torch.ones_like(max_val)
        for j in range(inputs.shape[2]):
            data = inputs[..., j].unsqueeze(-1)
            data_q1 = ((data / scale1 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale1
            data_q2 = ((data / scale2 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale2
            data_q4 = ((data / scale4 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale4
            data_q8 = ((data / scale8 + zero_point).round().clamp(qmin, qmax) -
                       zero_point) * scale8
            score1 = lp_loss(data, data_q1, p=2.0, reduction='all')
            score2 = lp_loss(data, data_q2, p=2.0, reduction='all')
            score4 = lp_loss(data, data_q4, p=2.0, reduction='all')
            score8 = lp_loss(data, data_q8, p=2.0, reduction='all')
            score = [score1, score2, score4, score8]
            scale_mask[j] *= 2**score.index(min(score))
        scale = scale1 * scale_mask
        return scale, zero_point
