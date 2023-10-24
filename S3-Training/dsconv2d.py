'''
Copyright (C) 2023. Huawei Technologies Co., Ltd. All rights reserved.
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

class DenseShiftConv2d3bit(nn.Conv2d):
    """
        3bit DenseShift Conv2d module.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DenseShiftConv2d3bit, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                             bias, padding_mode)

        self.weight_sign = self.weight
        self.weight_t1 = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_t2 = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.weight_t3 = Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        shift_bits = ste_binarize01(self.weight_t1)
        shift_bits = (shift_bits + 1) * ste_binarize01(self.weight_t2)
        shift_bits = (shift_bits + 1) * ste_binarize01(self.weight_t3)
        base = torch.ones_like(self.weight_sign) * 2

        w_sign = ste_binarize(self.weight_sign)

        bw_w_shift_3bit = w_sign * torch.sqrt(shift_bits + 1)
        w_shift_3bit = bw_w_shift_3bit
        with torch.no_grad():
            fw_w_shift_3bit = w_sign * torch.pow(base, shift_bits)
            w_shift_3bit += fw_w_shift_3bit - bw_w_shift_3bit

        return self._conv_forward(input, w_shift_3bit)