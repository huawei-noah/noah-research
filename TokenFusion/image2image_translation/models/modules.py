#Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.
#
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import torch.nn as nn
import cfg
import torch


class TokenExchange(nn.Module):
    def __init__(self):
        super(TokenExchange, self).__init__()

    def forward(self, x, mask, mask_threshold):
        # x: [B, N, C], mask: [B, N, 2]
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x0[mask[0] >= mask_threshold] = x[0][mask[0] >= mask_threshold]
        x0[mask[0] < mask_threshold] = x[1][mask[0] < mask_threshold]
        x1[mask[1] >= mask_threshold] = x[1][mask[1] >= mask_threshold]
        x1[mask[1] < mask_threshold] = x[0][mask[1] < mask_threshold]
        return [x0, x1]


class ChannelExchange(nn.Module):
    def __init__(self):
        super(ChannelExchange, self).__init__()

    def forward(self, x, lrnorm, lrnorm_threshold):
        lrnorm0, lrnorm1 = lrnorm[0].weight.abs(), lrnorm[1].weight.abs()
        x0, x1 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x0[:, lrnorm0 >= lrnorm_threshold] = x[0][:, lrnorm0 >= lrnorm_threshold]
        x0[:, lrnorm0 < lrnorm_threshold] = x[1][:, lrnorm0 < lrnorm_threshold]
        x1[:, lrnorm1 >= lrnorm_threshold] = x[1][:, lrnorm1 >= lrnorm_threshold]
        x1[:, lrnorm1 < lrnorm_threshold] = x[0][:, lrnorm1 < lrnorm_threshold]
        return [x0, x1]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class LayerNormParallel(nn.Module):
    def __init__(self, num_features):
        super(LayerNormParallel, self).__init__()
        for i in range(cfg.num_parallel):
            setattr(self, 'lrnorm_' + str(i), nn.LayerNorm(num_features, eps=1e-6))

    def forward(self, x_parallel):
        return [getattr(self, 'lrnorm_' + str(i))(x) for i, x in enumerate(x_parallel)]
