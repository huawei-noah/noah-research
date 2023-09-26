# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

import numpy as np
import torch
import torch.nn as nn
import torchvision
import sys

sys.path.append('..')
from core.common.utils import logistic4parameters


class PairwiseRankingwRefBatchAll(nn.Module):
    def __init__(self, margin=0.2, invert=True):
        super(PairwiseRankingwRefBatchAll, self).__init__()

        self.margin = margin
        self.relu = nn.ReLU()
        self.invert = invert
        self.debug = True

    def forward(self, inputs, labels):
        # input: [N, 1]
        # labels: [N]
        if self.debug:
            print("-------------------------------------------------------------------")
            print("[PW-Rank-wRef-all] preds:",inputs.shape, \
                    "\n                         ",inputs[:3].t())      # [1,N]
            print("[PW-Rank-wRef-all]   mos:",labels.shape, \
                    "\n                         ",labels[:3])     # [N]
        n_samples = len(inputs)

        diff = inputs - inputs.t()
        if self.debug:
            print("--------------")
            print("[PW-Rank-wRef-all] diff (preds-preds^t):",diff.shape, \
                    "\n",diff[:3,:3]) # [N,N]

        y = labels[None, :] - labels[:, None]
        if self.debug:
            print("--------------")
            print("[PW-Rank-wRef-all] y (labels-labels^t):",y.shape, \
                    "\n",y[:3,:3]) # [N,N]
        y = torch.sign(y)

        if self.invert:
            y *= -1
        if self.debug:
            print("--------------")
            print("[PW-Rank-wRef-all] y (sign):",y.shape, \
                    "\n",y[:3,:3]) # [N,N]

        loss = self.relu(-y * diff + self.margin)
        loss = torch.triu(loss, diagonal=1)

        cond = loss != 0.
        idx_x, idx_y = torch.where(cond)
        loss = loss[idx_x, idx_y]

        num = torch.triu(torch.sign(diff) == y, diagonal=1).sum()
        den = (n_samples ** 2 - n_samples) / 2
        acc = num / den
        return loss.mean(), acc


class PairwiseSigmoidwRefBatchAll(nn.Module):
    def __init__(self, temperature=0.05, invert=True):
        super().__init__()

        self.temperature = temperature
        self.criterion = nn.MSELoss()
        self.invert = invert

    def forward(self, inputs, labels):
        n_samples = len(inputs)
        n_pairs = (n_samples ** 2 - n_samples) / 2

        diff = inputs - inputs.t()
        p_diff = torch.sigmoid(diff / self.temperature)

        y = labels[None, :] - labels[:, None]
        M = 400.
        p_y = 1 / (1 + 10 ** (y / M))

        if self.invert:
            p_y = 1 - p_y

        cond = torch.triu(p_diff, diagonal=1) != 0.
        idx_x, idx_y = torch.where(cond)
        loss = self.criterion(p_diff[idx_x, idx_y], p_y[idx_x, idx_y])

        num = torch.triu((p_diff > 0.5) == (p_y > 0.5), diagonal=1).sum()
        acc = num / n_pairs

        return loss, acc


class PairwiseBCEwRefBatchAll(nn.Module):
    def __init__(self, temperature=0.05, invert=True):
        super(PairwiseBCEwRefBatchAll, self).__init__()

        self.temperature = temperature
        self.criterion = nn.BCELoss()
        self.invert = invert

    def forward(self, inputs, labels):

        n_samples = len(inputs)
        n_pairs = (n_samples ** 2 - n_samples) / 2

        diff = inputs - inputs.t()
        p_diff = torch.sigmoid(diff / self.temperature)

        y = (labels[None, :] - labels[:, None]) < 0.

        if self.invert:
            y = torch.logical_not(y)

        labels = (1. * y)

        cond = torch.triu(p_diff, diagonal=1) != 0.
        idx_x, idx_y = torch.where(cond)
        loss = self.criterion(p_diff[idx_x, idx_y], labels[idx_x, idx_y])

        num = torch.triu((p_diff > 0.5) == (y > 0.), diagonal=1).sum()
        acc = num / n_pairs

        return loss, acc


class PairwiseRankingwRef(nn.Module):
    def __init__(self, margin=0.2, invert=True):
        super(PairwiseRankingwRef, self).__init__()

        self.margin = margin
        self.criterion = nn.MarginRankingLoss(margin=margin)
        self.invert = invert

    def forward(self, inputs0, inputs1):
        s0, y0 = inputs0
        s1, y1 = inputs1 

        y = torch.sign(y0 - y1)

        if self.invert:
            y *= -1

        loss = self.criterion(s0, s1, y)
        acc = (1. * (torch.sign(s0 - s1).squeeze() == y)).mean()
        return loss, acc


class PairwiseSigmoidwRef(nn.Module):
    def __init__(self, temperature=0.05, invert=True):
        super(PairwiseSigmoidwRef, self).__init__()

        self.temperature = temperature
        self.criterion = nn.MSELoss(reduction='none')

        self.invert = invert

    def forward(self, inputs0, inputs1):

        s0, y0 = inputs0
        s1, y1 = inputs1 

        diff = s0 - s1
        p_diff = torch.sigmoid(diff / self.temperature).squeeze()

        y = y0 - y1
        M = 400.
        p_y = 1 / (1 + 10 ** (- y / M))

        if self.invert:
            p_y = 1 - p_y

        loss = self.criterion(p_diff, p_y)

        if torch.isnan(loss).sum() > 0:
            ipdb.set_trace()

        loss = loss.mean()

        acc = (1. * ((p_diff > 0.5) == (p_y > 0.5))).mean()

        return loss, acc


class PairwiseBCEwRef(nn.Module):
    def __init__(self, temperature=0.05, invert=True):
        super(PairwiseBCEwRef, self).__init__()

        self.temperature = temperature
        self.criterion = nn.BCELoss()

        self.invert = invert

    def forward(self, inputs0, inputs1):

        s0, y0 = inputs0
        s1, y1 = inputs1

        diff = s0 - s1
        p_diff = torch.sigmoid(diff / self.temperature).squeeze()

        y = (y0 - y1) > 0.
        if self.invert:
            y = torch.logical_not(y)
        labels = (1. * y)

        loss = self.criterion(p_diff, labels)

        acc = (1. * ((p_diff > 0.5) == (y > 0.))).mean()

        return loss, acc


class PearsonCorrSample(nn.Module):
    def __init__(self, l4p=False):
        super(PearsonCorrSample, self).__init__()

        self.criterion = self.pearson
        self.l4p = l4p

    def get_standard_score(self, x, eps=1e-8):
        res = ((x - x.mean()) ** 2).sum() / (len(x) - 1)
        res = (res + eps) ** 0.5
        return res

    def pearson(self, x, y):
        n_samples = float(len(x))

        s_x = self.get_standard_score(x)
        s_y = self.get_standard_score(y)

        num = (x * y).sum() - n_samples * x.mean() * y.mean()
        den = (n_samples - 1) * s_x * s_y

        return num / den
    
    def expit(self, x):
        return 1 / (1 + torch.exp(-x))

    def least_square_l4p(self, x, y):
        x_np = x.squeeze().detach().cpu().numpy()
        y_np = y.squeeze().detach().cpu().numpy()

        _, params = logistic4parameters(y_np, x_np, return_params=True)

        A, B, C, D = params
        num = A - D

        den = self.expit((x - B) / np.abs(C))

        return num * den + D

    def forward(self, inputs, labels):
        if self.l4p:
            inputs = self.least_square_l4p(inputs, labels)
        res = self.criterion(inputs.squeeze(), labels)
        return res


class SpearmanCorrSample(nn.Module):
    def __init__(self, temperature=0.01):
        super(SpearmanCorrSample, self).__init__()

        self.temperature = temperature
        self.criterion = self.spearman

    def spearman(self, x, y):
        n_samples = float(len(x))

        d_x = x[:, None] - x[None, :]
        r_x = torch.sigmoid(d_x / self.temperature).sum(0) + 0.5

        d_y = y[:, None] - y[None, :]
        r_y = torch.sigmoid(d_y / self.temperature).sum(0) + 0.5

        num = 6 * ((r_x - r_y) ** 2).sum()
        den = n_samples * (n_samples**2 - 1)
        
        return 1 - num/den

    def forward(self, inputs, labels):
        res = self.criterion(inputs.squeeze(), labels)
        return res


class KendallCorrSample(nn.Module):
    def __init__(self, temperature=0.01):
        super(KendallCorrSample, self).__init__()

        self.temperature = temperature
        self.criterion = self.kendall

    def kendall(self, x, y):
        n_samples = float(len(x))

        d_x = torch.triu(x[:, None] - x[None, :], diagonal=1)
        r_x = torch.tanh(d_x / self.temperature)

        d_y = torch.triu(y[:, None] - y[None, :], diagonal=1)
        r_y = torch.tanh(d_y / self.temperature)

        factor = 2 / (n_samples * (n_samples - 1))

        return (r_x * r_y).sum() * factor

    def forward(self, inputs, labels):
        res = self.criterion(inputs.squeeze(), labels)
        return res




class PairwiseMSEwRefBatchAll(nn.Module):
    def __init__(self, invert=True):
        super(PairwiseMSEwRefBatchAll, self).__init__()

        self.criterion = nn.MSELoss(reduction='none')
        self.invert = invert
        self.debug = False

    def forward(self, inputs, labels):
        # input: [N, 1]
        # labels: [N]
        if self.debug:
            print("-------------------------------------------------------------------")
            print("[PW-MSE-wRef-all] preds:",inputs.shape, \
                    "\n                         ",inputs[:3].t())      # [1,N]
            print("[PW-MSE-wRef-all]   mos:",labels.shape, \
                    "\n                         ",labels[:3])     # [N]
        n_samples = len(inputs)

        diff = inputs - inputs.t()
        if self.debug:
            print("--------------")
            print("[PW-MSE-wRef-all] diff (preds-preds^t):",diff.shape, \
                    "\n",diff[:3,:3]) # [N,N]

        y = labels[None, :] - labels[:, None]
        if self.debug:
            print("--------------")
            print("[PW-MSE-wRef-all] y (labels-labels^t):",y.shape, \
                    "\n",y[:3,:3]) # [N,N]

        if self.invert:
            y *= -1
        if self.debug:
            print("--------------")
            print("[PW-MSE-wRef-all] y (invert?):",y.shape, \
                    "\n",y[:3,:3]) # [N,N]

        loss = self.criterion(diff,y.type(torch.FloatTensor).cuda(non_blocking=True))
        if self.debug:
            print("[PW-MSE-wRef-all] diff:",diff.dtype)
            print("[PW-MSE-wRef-all]    y:",y.dtype)
            print("[PW-MSE-wRef-all]  MSE:",loss.shape,"\n",loss[:3,:3])       # [N,N]

        loss = torch.triu(loss, diagonal=1)

        cond = loss > 0.
        idx_x, idx_y = torch.where(cond)
        loss = loss[idx_x, idx_y]
        if self.debug:
            print("[PW-MSE-wRef-all] loss(cond):",loss.shape)

        y = torch.sign(y)
        num = torch.triu(torch.sign(diff) == y, diagonal=1).sum()
        den = (n_samples ** 2 - n_samples) / 2
        acc = num / den
        return loss.mean(), acc


