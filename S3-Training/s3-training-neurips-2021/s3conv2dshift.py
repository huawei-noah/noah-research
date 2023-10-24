'''
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of BSD 3-Clause License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
BSD 3-Clause License for more details.
'''

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
# from torch import Tensor
from torch.autograd import Function

class STEBinarize01F(Function):
    @staticmethod
    def forward(ctx, inputs):
        return (inputs.sign() - (inputs == 0).float() + 1) * 0.5
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

ste_binarize01 = STEBinarize01F.apply

class STBinarizeF(Function):
    @staticmethod
    def forward(ctx, inputs):
        return inputs.sign() + (inputs == 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

ste_binarize = STBinarizeF.apply

class S3Conv2dShift3bit(nn.Conv2d):
    """
        S3 reparameterized Conv2d module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(S3Conv2dShift3bit, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             bias, padding_mode)

        self.weight_sign = self.weight
        self.weight_val = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_shift = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_shift2 = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        shift_bits = ste_binarize01(self.weight_shift) * (ste_binarize01(self.weight_shift2) + 1)
        base = torch.ones_like(self.weight_val) * 2
        shift3bit_weight = ste_binarize(self.weight_sign) * ste_binarize01(self.weight_val) * torch.pow(base, shift_bits)
        return self._conv_forward(input, shift3bit_weight)


def add_reg_sparse_to_loss(model, loss, alpha=1e-5):
    reg = 0.
    for name, param in model.named_parameters():
        if 'weight_val' in name:
            reg += torch.sum(torch.max(-1 * param, torch.zeros_like(param)))
    return loss + alpha * reg
