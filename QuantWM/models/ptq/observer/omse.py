# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss
CLIPMIN = 1e-5 # 1e-5
CLIPMAX = 1e4 # 1e4

class OmseObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode,shape=None):
        super(OmseObserver, self).__init__(module_type, bit_type,
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


    def lp_loss(self, pred, tgt, p=2.0, reduction='none'):
        if reduction == 'none':
            return (pred - tgt).abs().pow(p).sum(1).mean()

        elif reduction =='channel1':
            return (pred - tgt).abs().pow(p).sum(1) # weight (cout, cin)
        elif reduction =='channel0':
            return (pred - tgt).abs().pow(p).sum(0) # output (n_token, Cout)
        else:
            return (pred - tgt).abs().pow(p).mean()


    def get_quantization_params(self, inputs):
        if self.calibration_mode == 'layer_wise': # for activation
            max_val = self.max_val
            min_val = self.min_val
            qmax = self.bit_type.upper_bound
            qmin = self.bit_type.lower_bound

            best_score = 1e+10
            for i in range(90):
                new_max = max_val * (1.0 - (i * 0.01))
                new_min = min_val * (1.0 - (i * 0.01))
                new_scale = (new_max - new_min) / float(qmax - qmin)
                new_scale.clamp_(self.eps)
                new_zero_point = qmin - torch.round(new_min / new_scale)
                new_zero_point.clamp_(qmin, qmax)
                inputs_q = ((inputs / new_scale + new_zero_point).round().clamp(
                    qmin, qmax) - new_zero_point) * new_scale
                # L_p norm minimization as described in LAPQ
                # https://arxiv.org/abs/1911.07190
                score = lp_loss(inputs, inputs_q, p=2.0, reduction='all')
                if score < best_score:
                    best_score = score
                    self.max_val = new_max
                    self.min_val = new_min
                    scale = new_scale
                    zero_point = new_zero_point
            return scale, zero_point

        elif self.calibration_mode == 'channel_wise': # for weight
            inputs = self.reshape_tensor(inputs)  # w   [co, ci] == [co, n_group * group_size] => [co*n_group, group_size]
            symmetric = True
            reduce_shape = [-1]
            xmin_ori = inputs.amin(reduce_shape, keepdim=True).to(inputs.device)
            xmax_ori =  inputs.amax(reduce_shape, keepdim=True).to(inputs.device)
            qmax = self.bit_type.upper_bound
            qmin = self.bit_type.lower_bound

            # loop_times = 100; bottom_bound = 0.65
            loop_times = 100; bottom_bound = 0.35 # better
            ratio_list = torch.ones_like(xmax_ori)
            scale_list = torch.ones_like(xmax_ori)
            best_score_list = torch.ones_like(xmax_ori) * 100000
            
            for i in range(loop_times + 1):
                temp_ratio = (bottom_bound + i * (1 - bottom_bound) / loop_times)
                xmax = temp_ratio*xmax_ori
                xmin = temp_ratio*xmin_ori

                if symmetric:
                    abs_max = torch.max(xmax.abs(),xmin.abs())
                    scale = abs_max / qmax
                    scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                    round_zero_point = torch.zeros_like(scale)
                else:
                    scale = (xmax - xmin) / qmax
                    scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                    zero_point = -(xmin) / (scale)
                    round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round().clamp(qmin, qmax)

                inputs_q = (inputs/scale.to(inputs.device)).round().add(round_zero_point.to(inputs.device)).clamp(qmin, qmax).sub(round_zero_point.to(inputs.device)).mul(scale.to(inputs.device))
                scores = self.lp_loss(inputs, inputs_q, p=2.0, reduction='channel1').reshape(xmax.shape)

                better_index = scores < best_score_list # find channel_wise best scale
                ratio_list[better_index] = temp_ratio
                best_score_list = torch.min(best_score_list, scores)

            xmax = ratio_list*xmax_ori
            xmin = ratio_list*xmin_ori
            if symmetric:
                abs_max = torch.max(xmax.abs(),xmin.abs())
                scale = abs_max / qmax
                scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                round_zero_point = torch.zeros_like(scale)
            else:
                scale = (xmax - xmin) / qmax
                scale = scale.clamp(min=CLIPMIN, max=CLIPMAX)
                zero_point = -(xmin) / (scale)
                round_zero_point = zero_point.clamp(min=-CLIPMAX, max=CLIPMAX).round().clamp(qmin, qmax)
            return scale, round_zero_point
