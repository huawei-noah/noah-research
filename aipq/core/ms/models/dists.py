# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

# MIT License
# 
# Copyright (c) 2020 Keyan Ding
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included in all
#     copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#     SOFTWARE.

import os
import sys
from collections import namedtuple
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops

root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)
from mindspore.common.initializer import Normal, initializer

import numpy as np
from .vgg import vgg16 as Vgg16



class L2pooling(nn.Cell):
    """from https://github.com/dingkeyan93/DISTS/blob/master/DISTS_pytorch/DISTS_pt.py"""

    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = ms.numpy.hanning(filter_size)[1:-1]
        g = ms.Tensor(a[:, None] * a[None, :])
        g = g / g.sum()
        self.filter = ms.numpy.tile(g[None, None, :, :], (self.channels, 1, 1, 1)).astype("float32")

    def construct(self, x):
        x = x**2
        conv2d = ops.Conv2D(
            out_channel=self.filter.shape[0],
            kernel_size=self.filter.shape[2],
            stride=self.stride,
            pad_mode="pad",
            pad=self.padding,
            group=x.shape[1],
        )
        out = conv2d(x, self.filter)
        return (out + 1e-12).sqrt()


class vgg16(nn.Cell):
    def __init__(self, requires_grad=False, model_path=None, l2pooling=False):
        super(vgg16, self).__init__()
        v16 = Vgg16()
        if model_path and os.path.exists(model_path):
            param_dict = ms.load_checkpoint(model_path)
            ms.load_param_into_net(v16, param_dict, strict_load=True)
        vgg_pretrained_features = v16.features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.N_slices = 5

        if l2pooling:
            for x in range(0, 4):
                self.slice1.append(vgg_pretrained_features[x])
            self.slice2.append(L2pooling(channels=64))
            for x in range(5, 9):
                self.slice2.append(vgg_pretrained_features[x])
            self.slice3.append(L2pooling(channels=128))
            for x in range(10, 16):
                self.slice3.append(vgg_pretrained_features[x])
            self.slice4.append(L2pooling(channels=256))
            for x in range(17, 23):
                self.slice4.append(vgg_pretrained_features[x])
            self.slice5.append(L2pooling(channels=512))
            for x in range(24, 30):
                self.slice5.append(vgg_pretrained_features[x])
        else:
            for x in range(4):
                self.slice1.append(vgg_pretrained_features[x])
            for x in range(4, 9):
                self.slice2.append(vgg_pretrained_features[x])
            for x in range(9, 16):
                self.slice3.append(vgg_pretrained_features[x])
            for x in range(16, 23):
                self.slice4.append(vgg_pretrained_features[x])
            for x in range(23, 30):
                self.slice5.append(vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


class DISTSsinglewRef(nn.Cell):
    def __init__(self, pretrained=True, pnet_tune=False):
        super(DISTSsinglewRef, self).__init__()

        self.MS_major_ver = int(ms.__version__.split('.')[0])

        self.pnet_tune = pnet_tune

        model_path = None
        self.net = vgg16(model_path=model_path, requires_grad=self.pnet_tune, l2pooling=True)

        self.mean = ms.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = ms.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        self.chns = [3, 64, 128, 256, 512, 512]

        self.alpha = ms.Parameter(
            initializer(
                Normal(sigma=0.01, mean=0.1),
                shape=(1, sum(self.chns), 1, 1),
                dtype=ms.float32,
            )
        )
        self.beta = ms.Parameter(
            initializer(
                Normal(sigma=0.01, mean=0.1),
                shape=(1, sum(self.chns), 1, 1),
                dtype=ms.float32,
            )
        )

    def custom_net_construct(self, x):
        h = (x - self.mean) / self.std
        h = self.net.slice1(h)
        h_relu1_2 = h
        h = self.net.slice2(h)
        h_relu2_2 = h
        h = self.net.slice3(h)
        h_relu3_3 = h
        h = self.net.slice4(h)
        h_relu4_3 = h
        h = self.net.slice5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def construct(self, ref, x, normalize=False):
        feats0 = self.custom_net_construct(ref)
        feats1 = self.custom_net_construct(x)

        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()

        if self.MS_major_ver > 1:
            alpha = ops.split(tensor=self.alpha/w_sum, split_size_or_sections=self.chns, axis=1)
            beta = ops.split(tensor=self.beta/w_sum, split_size_or_sections=self.chns, axis=1)
        else:
            alpha = []
            beta = []
            start_ = 0
            for i in self.chns:
                end = start_ + i
                alpha.append((self.alpha / w_sum)[:, start_:end, :, :])
                beta.append((self.beta / w_sum)[:, start_:end, :, :])
                start_ = end

        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keep_dims=True)
            y_mean = feats1[k].mean([2, 3], keep_dims=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdims=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keep_dims=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keep_dims=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keep_dims=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdims=True)

        score = 1 - (dist1 + dist2)
        return score.squeeze(-1).squeeze(-1), None


class DISTSpairwRef(nn.Cell):
    def __init__(self, model_path=None, pnet_tune=False):
        super(DISTSpairwRef, self).__init__()

        self.pnet_tune = pnet_tune

        self.net = vgg16(model_path=model_path, requires_grad=self.pnet_tune, l2pooling=True)

        self.mean = ms.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = ms.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
        self.chns = [3, 64, 128, 256, 512, 512]
        self.alpha = ms.Parameter(
            initializer(
                Normal(sigma=0.01, mean=0.1),
                shape=(1, sum(self.chns), 1, 1),
                dtype=ms.float32,
            )
        )
        self.beta = ms.Parameter(
            initializer(
                Normal(sigma=0.01, mean=0.1),
                shape=(1, sum(self.chns), 1, 1),
                dtype=ms.float32,
            )
        )

    def custom_net_construct(self, x):
        h = (x - self.mean) / self.std
        h = self.net.slice1(h)
        h_relu1_2 = h
        h = self.net.slice2(h)
        h_relu2_2 = h
        h = self.net.slice3(h)
        h_relu3_3 = h
        h = self.net.slice4(h)
        h_relu4_3 = h
        h = self.net.slice5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def inner_construct(self, feats_ref, feats_x):
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()

        alpha = []
        beta = []
        start_ = 0
        for i in self.chns:
            end = start_ + i
            alpha.append((self.alpha / w_sum)[:, start_:end, :, :])
            beta.append((self.beta / w_sum)[:, start_:end, :, :])
            start_ = end

        for k in range(len(self.chns)):
            x_mean = feats_ref[k].mean([2, 3], keepdim=True)
            y_mean = feats_x[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean**2 + y_mean**2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats_ref[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats_x[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats_ref[k] * feats_x[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2)

        return score.squeeze(-1).squeeze(-1)

    def construct(self, ref, x0, x1, normalize=False):
        feats_ref = self.custom_net_construct(ref)
        feats0 = self.custom_net_construct(x0)
        feats1 = self.custom_net_construct(x1)

        val0 = self.inner_construct(feats_ref, feats0)
        val1 = self.inner_construct(feats_ref, feats1)

        return (val0, None), (val1, None)


