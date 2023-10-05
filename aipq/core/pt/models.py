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

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

import lpips
import lpips.pretrained_networks as pn
from collections import namedtuple


###########
# Modules #
###########

def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.norm(in_feat, dim=1, keepdim=True)
    return in_feat/(norm_factor+eps)


def L2norm(x, eps=1e-6):
    norm = torch.norm(x, dim=1)
    return x / (norm[:, None] + eps)


class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        x = x**2
        out = F.conv2d(x, self.filter,
                       stride=self.stride, padding=self.padding,
                       groups=x.shape[1])
        return (out + 1e-12).sqrt()


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(),] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


############
# Backbone #
############

class ResNet50(nn.Module):    
    def __init__(self, n_classes=1000, pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=pretrained)

        if (n_classes == 1000) and pretrained:
            # preserve last layer
            last_layer = list(self.resnet.children())[-1:]
            self.fc = nn.Sequential(*last_layer)
        else:
            # remove last layer
            self.fc = nn.Linear(2048, n_classes)

        # body of the network
        layers = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*layers)


    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.resnet(x)

        features = torch.flatten(features, 1)
        outputs = self.fc(features)

        return outputs, features


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True, l2pooling=False):
        super(vgg16, self).__init__()
        # ----
        # DEPRECATED since torchvision 0.13 (will be removed in 0.15), please use 'weights' instead.
        major_ver, minor_ver, _ = torchvision.__version__.split('.')  # ie ['0', '13', '1+cu102']
        if int(minor_ver) >= 13:
            if pretrained:
                torchvision_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
                vgg_pretrained_features = torchvision.models.vgg16(weights=torchvision_weights).features
            else:
                vgg_pretrained_features = torchvision.models.vgg16(weights=None).features
        else:
            vgg_pretrained_features = torchvision.models.vgg16(pretrained=pretrained).features
        # ----
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5

        if l2pooling:
            for x in range(0, 4):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            self.slice2.add_module(str(4), L2pooling(channels=64))
            for x in range(5, 9):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            self.slice3.add_module(str(9), L2pooling(channels=128))
            for x in range(10, 16):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            self.slice4.add_module(str(16), L2pooling(channels=256))
            for x in range(17, 23):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            self.slice5.add_module(str(23), L2pooling(channels=512))
            for x in range(24, 30):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])
        else:
            for x in range(4):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(4, 9):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(9, 16):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(16, 23):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            for x in range(23, 30):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
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
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class AlexNet(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(AlexNet, self).__init__()
        model = torchvision.models.alexnet(pretrained=pretrained)

        self.features = model.features
        self.avgpool = model.avgpool

        layers = list(model.classifier.children())[:-1]
        self.classifier = nn.Sequential(*layers)
        self.fc = nn.Linear(4096, n_classes)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        # from 224x224 to 4096, to nclasses
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.fc(out)
        return out, features


class VGG16(nn.Module):
    def __init__(self, n_classes, pretrained=True):
        super(VGG16, self).__init__()
        model = torchvision.models.vgg16(pretrained=pretrained)

        self.features = model.features
        self.avgpool = model.avgpool

        layers = list(model.classifier.children())[:-1]
        self.classifier = nn.Sequential(*layers)
        self.fc = nn.Linear(4096, n_classes)

    def require_all_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        # from 224x224 to 4096, to nclasses
        features = self.features(x)
        out = self.avgpool(features)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.fc(out)
        return out, features


class ResNetFCN(nn.Module):
    def __init__(self, original_model, kernel=7, stride=1):
        super().__init__()

        # all layers up until the avgpool (non-included)
        layers = list(original_model.children())[:-1][0][:-1]
        self.backbone = nn.Sequential(*layers)

        d_in = original_model.fc.in_features
        d_out = original_model.fc.out_features

        pool = nn.AvgPool2d((kernel, kernel), stride)
        conv = nn.Conv2d(d_in, d_out, 1, 1)

        conv.weight.data = original_model.fc.weight.view(d_out, d_in, 1, 1)
        conv.bias = original_model.fc.bias

        layers = [pool, conv]
        self.head = nn.Sequential(*layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x, flatten=True):
        features = self.backbone(x)
        outputs = self.head(features)
        if flatten:
            outputs = self.avgpool(outputs)
            outputs = torch.flatten(outputs, 1)
        return outputs, features


################
# Architecture #
################

class LPIPS(nn.Module):
    def __init__(self, net='alex', lpips=True,
                 pretrained=True, pnet_tune=False,
                 use_dropout=True, l2pooling=False):
        # lpips - [True] means with linear calibration on top of base network
        # pretrained - [True] means load linear weights

        super(LPIPS, self).__init__()

        self.pnet_type = net
        self.pnet_tune = pnet_tune
        self.lpips = lpips  # false means baseline of just averaging all layers

        if(self.pnet_type in ['vgg', 'vgg16']):
            net_type = vgg16
            self.chns = [64,128,256,512,512]
        elif(self.pnet_type=='alex'):
            net_type = pn.alexnet
            self.chns = [64,192,384,256,256]
        elif(self.pnet_type=='squeeze'):
            net_type = pn.squeezenet
            self.chns = [64,128,256,384,384,512,512]
        elif(self.pnet_type=='resnet'):
            net_type = pn.resnet
            self.chns = [64, 256, 512, 1024, 2048]
        self.L = len(self.chns)

        if self.pnet_type == 'resnet':
            self.net = net_type(pretrained=pretrained,
                                requires_grad=self.pnet_tune,
                                num=50)
        elif self.pnet_type == 'vgg' and l2pooling:  # l2pooling only for vgg
            self.net = net_type(pretrained=pretrained,
                                requires_grad=self.pnet_tune,
                                l2pooling=l2pooling)
        else:
            self.net = net_type(pretrained=pretrained,
                                requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0,self.lin1,self.lin2,self.lin3,self.lin4]
            if(self.pnet_type=='squeeze'): # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins+=[self.lin5,self.lin6]
            self.lins = nn.ModuleList(self.lins)

    def forward(self, x, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            x = 2 * x  - 1

        output = self.net.forward(x)
        feats = {}

        for kk in range(self.L):
            feats[kk] = normalize_tensor(output[kk])

        if(self.lpips):
            res = [spatial_average(self.lins[kk](feats[kk]), keepdim=True)
                   for kk in range(self.L)]
        else:
            res = [spatial_average(feats[kk].sum(dim=1,keepdim=True), keepdim=True)
                   for kk in range(self.L)]

        val = res[0]
        for l in range(1,self.L):
            val += res[l]
        
        return val.squeeze().unsqueeze(1), res

    def clamping(self):
        '''Clamp linear modules to positive values like LPIPS'''
        for module in self.lins.modules():
            if(hasattr(module, 'weight') and module.kernel_size==(1,1)):
                module.weight.data = torch.clamp(module.weight.data,min=0)


class LPIPSsinglewRef(LPIPS):
    def __init__(self, net='alex', lpips=True,
                 pretrained=True, pnet_tune=False,
                 use_dropout=True, l2pooling=False):
        super(LPIPSsinglewRef, self).__init__(
            net=net, lpips=lpips,
            pretrained=pretrained, pnet_tune=pnet_tune,
            use_dropout=use_dropout, l2pooling=l2pooling)

    def forward(self, ref, x, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            ref = 2 * ref - 1
            x = 2 * x - 1

        outs_ref = self.net.forward(ref)
        outs_x = self.net.forward(x)
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


class LPIPSpairwRef(LPIPS):
    def __init__(self, net='alex', lpips=True,
                 pretrained=True, pnet_tune=False,
                 use_dropout=True, l2pooling=False):
        super(LPIPSpairwRef, self).__init__(
            net=net, lpips=lpips,
            pretrained=pretrained, pnet_tune=pnet_tune,
            use_dropout=use_dropout, l2pooling=l2pooling)

    def forward(self, ref, x0, x1, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            ref = 2 * ref - 1
            x0 = 2 * x0 - 1
            x1 = 2 * x1 - 1

        outs_ref = self.net.forward(ref)
        outs0 = self.net.forward(x0)
        outs1 = self.net.forward(x1)

        feats_ref, feats0, feats1 = {}, {}, {}
        diffs0, diffs1 = {}, {}

        for kk in range(self.L):
            feats_ref[kk] = normalize_tensor(outs_ref[kk])
            feats0[kk] = normalize_tensor(outs0[kk])
            feats1[kk] = normalize_tensor(outs1[kk])
            diffs0[kk] = (feats_ref[kk]-feats0[kk])**2
            diffs1[kk] = (feats_ref[kk]-feats1[kk])**2

        if(self.lpips):
            res0 = [spatial_average(self.lins[kk](diffs0[kk]), keepdim=True)
                    for kk in range(self.L)]
            res1 = [spatial_average(self.lins[kk](diffs1[kk]), keepdim=True)
                    for kk in range(self.L)]
        else:
            res0 = [spatial_average(diffs0[kk].sum(dim=1,keepdim=True), keepdim=True)
                    for kk in range(self.L)]
            res1 = [spatial_average(diffs1[kk].sum(dim=1,keepdim=True), keepdim=True)
                    for kk in range(self.L)]

        val0 = res0[0]
        val1 = res1[0]
        for l in range(1,self.L):
            val0 += res0[l]
            val1 += res1[l]

        val0 = val0.squeeze(-1).squeeze(-1)
        val1 = val1.squeeze(-1).squeeze(-1)

        return (val0, res0), (val1, res1)

    def inference(self, ref, x0, normalize=False):
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            ref = 2 * ref - 1
            x0 = 2 * x0 - 1

        outs_ref = self.net.forward(ref)
        outs0 = self.net.forward(x0)
        feats_ref, feats0 = {}, {}
        diffs0 = {}

        for kk in range(self.L):
            feats_ref[kk] = normalize_tensor(outs_ref[kk])
            feats0[kk] = normalize_tensor(outs0[kk])
            diffs0[kk] = (feats_ref[kk]-feats0[kk])**2

        if(self.lpips):
            res0 = [spatial_average(self.lins[kk](diffs0[kk]), keepdim=True)
                    for kk in range(self.L)]
        else:
            res0 = [spatial_average(diffs0[kk].sum(dim=1,keepdim=True), keepdim=True)
                    for kk in range(self.L)]

        val0 = res0[0]
        for l in range(1,self.L):
            val0 += res0[l]

        return val0.squeeze(-1).squeeze(1), res0


class DISTSsinglewRef(nn.Module):
    def __init__(self, pretrained=True, pnet_tune=False):

        super(DISTSsinglewRef, self).__init__()

        self.pnet_tune = pnet_tune

        self.net = vgg16(pretrained=pretrained,
                         requires_grad=self.pnet_tune,
                         l2pooling=True)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter(
            "alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter(
            "beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

    def custom_net_forward(self, x):
        h = (x-self.mean)/self.std
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

    def forward(self, ref, x, normalize=False):

        feats0 = self.custom_net_forward(ref)
        feats1 = self.custom_net_forward(x)

        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1, keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3], keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1 + dist2)
        return score.squeeze(-1).squeeze(-1), None


class DISTSpairwRef(nn.Module):
    def __init__(self, pretrained=True, pnet_tune=False):

        super(DISTSpairwRef, self).__init__()

        self.pnet_tune = pnet_tune

        self.net = vgg16(pretrained=pretrained,
                         requires_grad=self.pnet_tune,
                         l2pooling=True)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter(
            "alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter(
            "beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)

    def custom_net_forward(self, x):
        h = (x-self.mean)/self.std
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

    def inner_forward(self, feats_ref, feats_x):
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats_ref[k].mean([2, 3], keepdim=True)
            y_mean = feats_x[k].mean([2, 3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1, keepdim=True)

            x_var = ((feats_ref[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats_x[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats_ref[k]*feats_x[k]).mean([2,3], keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1 + dist2)

        return score.squeeze(-1).squeeze(-1)

    def forward(self, ref, x0, x1, normalize=False):

        feats_ref = self.custom_net_forward(ref)
        feats0 = self.custom_net_forward(x0)
        feats1 = self.custom_net_forward(x1)

        val0 = self.inner_forward(feats_ref, feats0)
        val1 = self.inner_forward(feats_ref, feats1)

        return (val0, None), (val1, None)



# Eperimental code
# IQT (unofficial) -------------------------------------------------------------------------------------------
# source: https://github.com/anse3832/IQT
#
class IQT(nn.Module):
    def __init__(self, train=True):

        super(IQT, self).__init__()

        
        # IQT config options
        from core.pt.iqt.config import Config
        self.config = Config({
            # device
            "GPU_ID": "0",
            "num_workers": 8,
        
            ## model for LIVE/CSIQ/TID
            #"n_enc_seq": 29*29,    # feat map dims (H x W) from backbone, this size is related to crop_size
            #"n_dec_seq": 29*29,               # feature map dimension (H x W) from backbone
            #"n_layer": 2,                     # number of encoder/decoder layers
            #"d_hidn": 256,                # input channel (C) of encoder / decoder (input: C x N)
            #"i_pad": 0,
            #"d_ff": 1024,                     # feed forward hidden layer dimension
            #"d_MLP_head": 512,                # hidden layer of final MLP
            #"n_head": 4,                      # number of head (in multi-head attention)
            #"d_head": 256,                # input channel (C) of each head (input: C x N) -> same as d_hidn
            #"dropout": 0.1,                   # dropout ratio of transformer
            #"emb_dropout": 0.1,               # dropout ratio of input embedding
            #"layer_norm_epsilon": 1e-12,
            #"n_output": 1,                    # dimension of final prediction
            #"crop_size": 256,                 # input image crop size
        
            # model for PIPAL (NTIRE2021 Challenge)
            "n_enc_seq": 29*29,   # feat map dims (H x W) from backbone, this size is related to crop_size
            "n_dec_seq": 29*29,                 # feature map dimension (H x W) from backbone   (21*21)
            "n_layer": 1,                       # number of encoder/decoder layers
            "d_hidn": 128,                  # input channel (C) of encoder / decoder (input: C x N)
            "i_pad": 0,
            "d_ff": 1024,                       # feed forward hidden layer dimension
            "d_MLP_head": 128,                  # hidden layer of final MLP
            "n_head": 4,                        # number of head (in multi-head attention)
            "d_head": 128,                  # input channel (C) of each head (input: C x N) -> same as d_hidn
            "dropout": 0.1,                     # dropout ratio of transformer
            "emb_dropout": 0.1,                 # dropout ratio of input embedding
            "layer_norm_epsilon": 1e-12,
            "n_output": 1,                      # dimension of final prediction
            "crop_size": 256,                   # input image crop size (used to be 192)
        
            # data
            "db_name": "PIPAL",                 # database name [ PIPAL | LIVE | CSIQ | TID2013 ]
            "db_path": "/storage/local/db/AIPQ/wth/PIPAL/123",          # root of dataset
            "snap_path": "./weights/local",                             # path for saving weights
            "txt_file_name": "./IQA_list/PIPAL.txt",                    # image list file (.txt)
            "train_size": 0.95,
            "scenes": "all",
        
            # ensemble in validation phase
            "test_ensemble": True,
            "n_ensemble": 5,
        
            # optimization
            "batch_size": 16,                   # apparently this is no longer being used (jose)
            "learning_rate": 2e-4,
            "weight_decay": 1e-5,
            "n_epoch": 300,
            "val_freq": 1,
            "save_freq": 5,
            "checkpoint": None,                 # load pretrained weights
            "T_max": 50,                        # cosine learning rate period (iteration)
            "eta_min": 0                        # mininum learning rate
        })

        # device setting
        self.device = torch.device("cuda:%s" %self.config.GPU_ID if torch.cuda.is_available() else "cpu")

        # create model
        from core.pt.iqt.model_main import IQARegression
        from core.pt.iqt.backbone import inceptionresnetv2, Mixed_5b, Block35, SaveOutput
        self.model_transformer = IQARegression(self.config).to(self.device)
        self.model_backbone = inceptionresnetv2(num_classes=1001, pretrained='imagenet+background').to(self.device)
        print("IQT object created (transformer/backbone).")

        # save intermediate layers
        self.save_output = SaveOutput()
        hook_handles = []
        for layer in self.model_backbone.modules():
            if isinstance(layer, Mixed_5b):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)
            elif isinstance(layer, Block35):
                handle = layer.register_forward_hook(self.save_output)
                hook_handles.append(handle)

        # forward() method needs to be aware of: 
        # 'optimizer'
        self.optimizer = None
        self.is_train = train # if this is 'True' self.optimizer needs to be redefined




    def custom_net_forward(self, x):
        y = self.model_backbone(x)
        feat_map = torch.cat(
            (self.save_output.outputs[0],
            self.save_output.outputs[2],
            self.save_output.outputs[4],
            self.save_output.outputs[6],
            self.save_output.outputs[8],
            self.save_output.outputs[10]),
            dim=1
        ) # feat_map: n_batch x (320*6) x 29 x 29
        # clear list (for saving next feature map)
        self.save_output.outputs.clear()
        return feat_map



    def forward(self, ref, x, normalize=False, debug=False):

        if debug:
            print("ref.size()=",ref.size(),"          x.size()=",x.size())
            BS_mismatch=False
            if self.config.batch_size != ref.size()[0]:
                print("BS mismatch (ref):",self.config.batch_size,ref.size()[0])
                BS_mismatch=True
            if self.config.batch_size != x.size()[0]:
                print("BS mismatch (x):",self.config.batch_size,x.size()[0])
                BS_mismatch=True

        # value is not changed
        enc_inputs = torch.ones(ref.size()[0], self.config.n_enc_seq+1).to(self.device)
        dec_inputs = torch.ones(ref.size()[0], self.config.n_dec_seq+1).to(self.device)

        # extract features from backbone network
        feats0 = self.custom_net_forward(ref)
        feats1 = self.custom_net_forward(x)

        # this value should be extracted from backbone network
        # enc_inputs_embed: batch x len_seq x n_feat
        # dec_inputs_embed: batch x len_seq x n_feat
        feat_diff = feats0 - feats1
        enc_inputs_embed = feat_diff
        dec_inputs_embed = feats0

        ## weight update (if training)
        if self.is_train:
            self.optimizer.zero_grad()

        if debug:
            print("enc_inputs:",enc_inputs.size())
            print("          _embed:",enc_inputs_embed.size())
            print("dec_inputs:",dec_inputs.size())
            print("          _embed:",dec_inputs_embed.size())

        pred = self.model_transformer(enc_inputs, enc_inputs_embed, dec_inputs, dec_inputs_embed)
        
        return pred, None



