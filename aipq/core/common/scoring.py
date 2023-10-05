# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

import numpy as np
import sys

sys.path.append('..')
from core.common.utils import get_spearman, get_pearson, get_kendalltau


def score(x, y, prefix='\n(all) '):
    srcc = get_spearman(y, x)
    plcc = get_pearson(y, x)
    p4lcc = get_pearson(y, x, l4p=True)
    krcc = get_kendalltau(y, x)
    txt = prefix
    txt += '\n    Pearson (l4p) LCC: %.4f\tPearson LCC: %.4f' % (p4lcc, plcc)
    txt += '\n    Spearman RCC: %.4f' % (srcc)
    txt += '\n    Kendall RCC: %.4f\n' % (krcc)
    print("\n\nP4LCC   SROCC   KROCC (python rounded to 3 decimal places)")
    print('%.3f\t%.3f\t%.3f' % (np.abs(p4lcc), np.abs(srcc), np.abs(krcc))) 
    return txt


def score_tqd(x, y, df):
    spcc = []
    pocc = []
    pocc4 = []
    kt = []
    for ref in df['ref_img'].unique():
        cond = df['ref_img'] == ref
        spcc.append(get_spearman(y[cond], x[cond]))
        pocc.append(get_pearson(y[cond], x[cond]))
        pocc4.append(get_pearson(y[cond], x[cond], l4p=True))
        kt.append(get_kendalltau(y[cond], x[cond]))
    txt = 'Spearman (per img): %.4f' % np.mean(spcc)
    txt += ('\nPreason (l4p, per img): %.4f\tPearson(per img): %.4f' %
            (np.mean(pocc4), np.mean(pocc)))
    txt += '\nKendall tau (per img): %.4f\n' % np.mean(kt)
    print('%.4f\t%.4f\t%.4f' %
          (np.abs(np.mean(spcc)), np.mean(pocc4), np.abs(np.mean(kt))))
    return txt


def score_shrq(x, y):
    spcc = []
    pocc = []
    pocc4 = []
    kt = []

    spcc.append(get_spearman(y[:360], x[:360]))
    spcc.append(get_spearman(y[360:], x[360:]))

    pocc.append(get_pearson(y[:360], x[:360]))
    pocc.append(get_pearson(y[360:], x[360:]))

    pocc4.append(get_pearson(y[:360], x[:360], l4p=True))
    pocc4.append(get_pearson(y[360:], x[360:], l4p=True))

    kt.append(get_kendalltau(y[:360], x[:360]))
    kt.append(get_kendalltau(y[360:], x[360:]))

    txt = 'Spearman (per img): %.4f' % np.mean(spcc)
    txt += ('\nPreason (l4p, per img): %.4f\tPearson(per img): %.4f' %
            (np.mean(pocc4), np.mean(pocc)))
    txt += '\nKendall tau (per img): %.4f\n' % np.mean(kt)
    print('%.4f\t%.4f\t%.4f' %
          (np.abs(np.mean(spcc)), np.mean(pocc4), np.abs(np.mean(kt))))
    return txt


