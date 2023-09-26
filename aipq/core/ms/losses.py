# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import sys

sys.path.append('..')
from mindspore.scipy import optimize





class PairwiseBCEwRefBatchAll(nn.Cell):
    ''' the order of operations changed a bit from Pytorch implementation'''
    def __init__(self, temperature=0.05, invert=True):
        super(PairwiseBCEwRefBatchAll, self).__init__()

        self.temperature = temperature
        self.criterion = nn.BCELoss(reduction='none')
        self.invert = invert

    def construct(self, inputs, labels):
        """Call the operator."""
        n_samples = len(inputs)
        n_pairs = (n_samples ** 2 - n_samples) / 2

        # predictions
        diff = inputs - inputs.transpose()
        p_diff = mindspore.ops.sigmoid(diff / self.temperature)

        # labels
        y = (labels[None, :] - labels[:, None]) < 0.
        if self.invert:
            y *= -1
        labels = (1. * y)

        # loss
        loss = self.criterion(p_diff, labels)

        # upper-triangular loss mtx, if loss > 0
        L = mindspore.numpy.triu(loss, k=1)
        zeroslike = mindspore.ops.ZerosLike()
        z = zeroslike(L)
        cond = L > 0.

        cond_loss = mindspore.numpy.where(cond,L,z)
        loss = cond_loss

        num = mindspore.numpy.triu((p_diff > 0.5) == (y > 0.), k=1).sum()
        acc = num / n_pairs

        return loss.mean(), acc



#---------------------------------------------

class PearsonCorrSample(nn.Cell):
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
        return 1 / (1 + ops.exp(-x))

    def log4params(self, t, pred, return_params=False):
        '''from:
        https://people.duke.edu/~ccc14/pcfb/analysis.html
        '''
        def logistic4(x, A, B, C, D):
            """4PL lgoistic equation."""
            num = A - D
            den = mindspore.nn.Sigmoid((x - B) / mindspore.numpy.abs(C))
            return num * den + D
    
        def residuals(p, y, x):
            """Deviations of data from fitted 4PL curve"""
            A, B, C, D = p
            err = y - logistic4(x, A, B, C, D)
            return err
    
    
        # Initial guess for parameters
        p0 = [mindspore.numpy.min(t), 1, 1, mindspore.numpy.max(t)]
    
        return p0


    def least_square_l4p(self, x, y):
        x_s = ops.squeeze(x)
        y_s = ops.squeeze(y)
        return x

    def construct(self, inputs, labels):
        """Call the operator. This is equivalent to the forward() fn in Pytorch."""
        if self.l4p:
            sys.exit("'--l4p' option is not suported in Mindspore.")
        res = self.criterion(ops.squeeze(inputs), labels)
        return res




class SpearmanCorrSample(nn.Cell):
    def __init__(self, temperature=0.01):
        super(SpearmanCorrSample, self).__init__()
        self.temperature = temperature
        self.criterion = self.spearman

    def construct(self, inputs, labels):
        """Call the operator. This is equivalent to the forward() fn in Pytorch."""
        res = self.criterion(ops.squeeze(inputs), labels)
        return res

    def spearman(self, x, y):
        n_samples = float(len(x))

        d_x = x[:, None] - x[None, :]
        r_x = ops.sigmoid(d_x / self.temperature).sum(0) + 0.5

        d_y = y[:, None] - y[None, :]
        r_y = ops.sigmoid(d_y / self.temperature).sum(0) + 0.5

        num = 6 * ((r_x - r_y) ** 2).sum()
        den = n_samples * (n_samples**2 - 1)
        return 1 - num/den



class KendallCorrSample(nn.Cell):
    def __init__(self, temperature=0.01):
        super(KendallCorrSample, self).__init__()

        self.temperature = temperature
        self.criterion = self.kendall

    def construct(self, inputs, labels):
        """Call the operator. This is equivalent to the forward() fn in Pytorch."""
        res = self.criterion(ops.squeeze(inputs), labels)
        return res

    def kendall(self, x, y):
        n_samples = float(len(x))

        d_x = mindspore.numpy.triu(x[:, None] - x[None, :], k=1)
        r_x = ops.tanh(d_x / self.temperature)

        d_y = mindspore.numpy.triu(y[:, None] - y[None, :], k=1)
        r_y = ops.tanh(d_y / self.temperature)

        factor = 2 / (n_samples * (n_samples - 1))
        return (r_x * r_y).sum() * factor






class PairwiseMSEwRefBatchAll(nn.Cell):

    def __init__(self, invert=True):
        """Initiallize."""
        super(PairwiseMSEwRefBatchAll, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.invert = invert
        self.debug = False

    def construct(self, inputs, labels):
        """Call the operator."""
        if self.debug:
            print("-------------------------------------------------------------------")
            print("[PW-MSE-wRef-all] preds:",inputs.shape, \
                    "\n                         ",inputs[:3].transpose())      # [1,N]
            print("[PW-MSE-wRef-all]   mos:",labels.shape, \
                    "\n                         ",labels[:3])     # [N]
        n_samples = len(inputs)
        # input: [N, 1]
        # labels: [N]

        diff = inputs - inputs.transpose()
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

        loss = self.criterion(diff,y)

        if self.debug:
            print("[PW-MSE-wRef-all] diff:",diff.dtype)
            print("[PW-MSE-wRef-all]    y:",y.dtype)

        loss = mindspore.numpy.triu(loss, k=1)
        zeroslike = mindspore.ops.ZerosLike()
        z = zeroslike(loss)

        cond = loss > 0.
        cond_loss = mindspore.numpy.where(cond,loss,z)
        loss=cond_loss
        if self.debug:
            print("[PW-MSE-wRef-all] loss(cond):",loss.shape)

        y = mindspore.numpy.sign(y)
        num = mindspore.numpy.triu(mindspore.numpy.sign(diff) == y, k=1).sum()
        den = (n_samples ** 2 - n_samples) / 2
        acc = num / float(den)
        return loss.mean(), acc


