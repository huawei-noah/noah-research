# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

import os
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.optimize import leastsq
from scipy.special import expit



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, log_file=None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
        if log_file is not None:
            write_info(log_file, '\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def write_info(fpath, txt):
    with open(fpath, 'a') as f:
        txt = '\n' + txt
        f.write(txt)


def create_folder(fdir):
    if not os.path.exists(fdir):
        os.makedirs(fdir)


def get_pearson(true, pred, l4p=False):
    if l4p:
        pred_l4p = logistic4parameters(true, pred)
        return pearsonr(true, pred_l4p)[0]
    else:
        return pearsonr(true, pred)[0]


def get_spearman(true, pred):
    return spearmanr(true, pred)[0]


def get_kendalltau(true, pred):
    return kendalltau(true, pred)[0]


def logistic4parameters(true, pred, return_params=False):
    '''from:
    https://people.duke.edu/~ccc14/pcfb/analysis.html
    '''
    def logistic4(x, A, B, C, D):
        """4PL lgoistic equation."""
        num = A - D
        den = expit((x - B) / np.abs(C))
        return num * den + D

    def residuals(p, y, x):
        """Deviations of data from fitted 4PL curve"""
        A, B, C, D = p
        err = y - logistic4(x, A, B, C, D)
        return err

    def peval(x, p):
        """Evaluated value at x with current parameters."""
        A, B, C, D = p
        return logistic4(x, A, B, C, D)

    # Initial guess for parameters
    p0 = [np.min(true), 1, 1, np.max(true)]

    # Fit equation using least squares optimization
    plsq = leastsq(residuals, p0, args=(true, pred))

    if return_params:
        return peval(pred, plsq[0]), plsq[0]
    else:
        return peval(pred, plsq[0])
