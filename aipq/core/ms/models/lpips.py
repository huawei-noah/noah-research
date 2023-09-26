# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
# All rights reserved.                                                                         
#                                                                                              
# Redistribution and use in source and binary forms, with or without                           
# modification, are permitted provided that the following conditions are met:                  
#                                                                                              
#     * Redistributions of source code must retain the above copyright notice, this            
#       list of conditions and the following disclaimer.                                       
#                                                                                              
#       * Redistributions in binary form must reproduce the above copyright notice,            
#         this list of conditions and the following disclaimer in the documentation            
#           and/or other materials provided with the distribution.                             
#                                                                                              
#           THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"        
#           AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE          
#           IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE     
#           DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE       
#           FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL         
#           DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR         
#                   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER 
#           CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,      
#           OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE      
#           OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               

import os
import sys
from collections import namedtuple
from pathlib import Path

import mindspore as ms
from mindspore import nn, ops


root_path = str(Path(__file__).parent.parent.parent)
sys.path.append(root_path)
from .vgg import vgg16 as models_vgg16

default_model_urls = {
    # key "url" is the default
    "0.0_alex": "https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.0_alex-18720f55.pth",
    "0.0_vgg": "https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.0_vgg-b9e42362.pth",
    "0.0_squeeze": "https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.0_squeeze-c27abd3a.pth",
    "0.1_alex": "https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.1_alex-df73285e.pth",
    "0.1_vgg": "https://github.com/chaofengc/IQA-Toolbox-Python/releases/download/v0.1-weights/LPIPS_v0.1_vgg-a78928a0.pth",
    "0.1_squeeze": "https://github.com/chaofengc/IQA-PyTorch/releases/download/v0.1-weights/LPIPS_v0.1_squeeze-4a5350f2.pth",
}


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



class squeezenet(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=True):
        super(squeezenet, self).__init__()
        pretrained_features = SqueezeNet().features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.slice6 = nn.SequentialCell()
        self.slice7 = nn.SequentialCell()
        self.N_slices = 7
        for x in range(2):
            self.slice1.append(pretrained_features[x])
        for x in range(2, 5):
            self.slice2.append(pretrained_features[x])
        for x in range(5, 8):
            self.slice3.append(pretrained_features[x])
        for x in range(8, 10):
            self.slice4.append(pretrained_features[x])
        for x in range(10, 11):
            self.slice5.append(pretrained_features[x])
        for x in range(11, 12):
            self.slice6.append(pretrained_features[x])
        for x in range(12, 13):
            self.slice7.append(pretrained_features[x])
        if not requires_grad:
            for param in self.get_parameters():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        h = self.slice6(h)
        h_relu6 = h
        h = self.slice7(h)
        h_relu7 = h
        vgg_outputs = namedtuple(
            "SqueezeOutputs",
            ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"],
        )
        out = vgg_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)
        return out


class alexnet(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = AlexNet().features
        self.slice1 = nn.SequentialCell()
        self.slice2 = nn.SequentialCell()
        self.slice3 = nn.SequentialCell()
        self.slice4 = nn.SequentialCell()
        self.slice5 = nn.SequentialCell()
        self.N_slices = 5
        for x in range(2):
            self.slice1.append(alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.append(alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.append(alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.append(alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.append(alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.get_parameters():
                param.requires_grad = False

    def construct(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple("AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"])
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)
        return out


class vgg16(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=True, l2pooling=False):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models_vgg16().features
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
            for param in self.get_parameters():
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


class resnet(nn.Cell):
    def __init__(self, requires_grad=False, pretrained=True, num=18):
        super(resnet, self).__init__()
        if num == 18:
            self.net = resnet18(pretrained=pretrained)
        elif num == 34:
            self.net = resnet34(pretrained=pretrained)
        elif num == 50:
            self.net = resnet50(pretrained=pretrained)
        elif num == 101:
            self.net = resnet101(pretrained=pretrained)
        elif num == 152:
            self.net = resnet152(pretrained=pretrained)
        self.N_slices = 5

        self.conv1 = self.net.conv1
        self.bn1 = self.net.bn1
        self.relu = self.net.relu
        self.maxpool = self.net.maxpool
        self.layer1 = self.net.layer1
        self.layer2 = self.net.layer2
        self.layer3 = self.net.layer3
        self.layer4 = self.net.layer4

    def construct(self, X):
        h = self.conv1(X)
        h = self.bn1(h)
        h = self.relu(h)
        h_relu1 = h
        h = self.maxpool(h)
        h = self.layer1(h)
        h_conv2 = h
        h = self.layer2(h)
        h_conv3 = h
        h = self.layer3(h)
        h_conv4 = h
        h = self.layer4(h)
        h_conv5 = h

        outputs = namedtuple("Outputs", ["relu1", "conv2", "conv3", "conv4", "conv5"])
        out = outputs(h_relu1, h_conv2, h_conv3, h_conv4, h_conv5)
        return out


def upsample(in_tens, out_HW=(64, 64)):  # assumes scale factor is same for H and W
    return nn.Upsample(size=out_HW, mode="bilinear", align_corners=False)(in_tens)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keep_dims=keepdim)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = ops.Sqrt()((in_feat**2).sum(axis=1, keepdims=True))
    return in_feat / (norm_factor + eps)



class ScalingLayer(nn.Cell):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.shift = ms.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        self.scale = ms.Tensor([0.458, 0.448, 0.450])[None, :, None, None]

    def construct(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Cell):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, has_bias=False),
        ]
        self.model = nn.SequentialCell(*layers)

    def construct(self, x):
        return self.model(x)


class LPIPSModel(nn.Cell):
    """LPIPS model.
    Args:
        lpips (Boolean) : Whether to use linear layers on top of base/trunk network.
        pretrained (Boolean): Whether means linear layers are calibrated with human
            perceptual judgments.
        pnet_rand (Boolean): Whether to randomly initialized trunk.
        net (String): ['alex','vgg','squeeze'] are the base/trunk networks available.
        version (String): choose the version ['v0.1'] is the default and latest;
            ['v0.0'] contained a normalization bug.
        pretrained_model_path (String): Petrained model path.

        The following parameters should only be changed if training the network:

        eval_mode (Boolean): choose the mode; True is for test mode (default).
        pnet_tune (Boolean): Whether to tune the base/trunk network.
        use_dropout (Boolean): Whether to use dropout when training linear layers.

    Reference:
        Zhang, Richard, et al. "The unreasonable effectiveness of deep features as
        a perceptual metric." Proceedings of the IEEE conference on computer vision
        and pattern recognition. 2018.

    """

    def __init__(
        self,
        pretrained=True,
        net="alex",
        version="0.1",
        lpips=True,
        spatial=False,
        pnet_rand=False,
        pnet_tune=False,
        use_dropout=True,
        l2pooling=False,
        pretrained_model_path=None,
        eval_mode=True,
        **kwargs
    ):
        super(LPIPSModel, self).__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips  # false means baseline of just averaging all layers
        self.version = version
        self.scaling_layer = ScalingLayer()

        if self.pnet_type in ["vgg", "vgg16"]:
            net_type = vgg16
            self.chns = [64, 128, 256, 512, 512]
        elif self.pnet_type == "alex":
            net_type = alexnet
            self.chns = [64, 192, 384, 256, 256]
        elif self.pnet_type == "squeeze":
            net_type = squeezenet
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        self.L = len(self.chns)

        if self.pnet_type == 'resnet':                                       
            self.net = net_type(pretrained=not self.pnet_rand,
                                requires_grad=self.pnet_tune,                
                                num=50)                                      
        elif self.pnet_type == 'vgg' and l2pooling:  # l2pooling only for vgg
            self.net = net_type(pretrained=not self.pnet_rand,
                                requires_grad=self.pnet_tune,                
                                l2pooling=l2pooling)
        else:                                                                
            self.net = net_type(pretrained=pretrained,                       
                                requires_grad=self.pnet_tune)                


        if lpips:
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if self.pnet_type == "squeeze":  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]
            self.lins = nn.CellList(self.lins)

        if eval_mode:
            self.set_train(False)

    def construct(self, in1, in0, retPerLayer=False, normalize=True):
        r"""Computation IQA using LPIPS.
        Args:
            in1: An input tensor. Shape :math:`(N, C, H, W)`.
            in0: A reference tensor. Shape :math:`(N, C, H, W)`.
            retPerLayer (Boolean): return result contains ressult of
                each layer or not. Default: False.
            normalize (Boolean): Whether to normalize image data range
                in [0,1] to [-1,1]. Default: True.

        Returns:
            Quality score.

        """
        if normalize:  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version == "0.1" else (in0, in1)
        )

        outs0, outs1 = self.net.construct(in0_input), self.net.construct(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
        else:
            if self.spatial:
                res = [upsample(diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk].sum(dim=1, keepdim=True), keepdim=True) for kk in range(self.L)]

        val = 0
        for i in range(self.L):
            val += res[i]

        if retPerLayer:
            return (val, res)
        else:
            return val.squeeze()




class LPIPSsinglewRef(LPIPSModel):
    def __init__(self, net='alex', lpips=True,
                 pretrained=True, pnet_tune=False,
                 use_dropout=True, l2pooling=False):
        super(LPIPSsinglewRef, self).__init__(
            net=net, lpips=lpips,
            pretrained=pretrained, pnet_tune=pnet_tune,
            use_dropout=use_dropout, l2pooling=l2pooling)


    def construct(self, ref, x, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            ref = 2 * ref - 1
            x = 2 * x - 1

        outs_ref = self.net.construct(ref)
        outs_x = self.net.construct(x)
        feats_ref, feats_x, diffs = {}, {}, {}

        for kk in range(self.L):
            feats_ref[kk] = normalize_tensor(outs_ref[kk])
            feats_x[kk] = normalize_tensor(outs_x[kk])
            diffs[kk] = (feats_ref[kk]-feats_x[kk])**2

        if(self.lpips):
            res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)]
        else:
            res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True)
                    for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        val = val.squeeze(-1).squeeze(-1)
        return val, res
