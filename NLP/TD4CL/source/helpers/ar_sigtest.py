#!/usr/bin/python

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
import json
import numpy as np
import os
from tqdm import tqdm
import random
from glob import glob


def calc_acc(system):
    true_positives = 0

    for i in system:
        if i['pred'] == i['label']:
            true_positives += 1

    perf = true_positives / len(system)
    return perf


def calc_delta(sys_A, sys_B):
    return np.abs(calc_acc(sys_A) - calc_acc(sys_B))


# Permutation-randomization
# Repeat R times: randomly flip each m_i(A),m_i(B) between A and B with probability 0.5, calculate delta(A,B).
# let r be the number of times that delta(A,B)<orig_delta(A,B)
# significance level: (r+1)/(R+1)
# Assume that larger value (metric) is better
def rand_permutation(data_A, data_B, n, R):
    delta_orig = calc_delta(data_A, data_B)
    print(f'Original delta: {delta_orig}')
    r = 0
    pbar = tqdm(range(0, R), disable=False)
    for x in pbar:
        temp_A = data_A
        temp_B = data_B
        samples = random.choices([0,1], k=n)

        swap_ind = np.nonzero(samples)[0]
        # print(swap_ind)
        for ind in swap_ind:
            temp_B[ind], temp_A[ind] = data_A[ind], data_B[ind]

        delta = calc_delta(temp_A, temp_B)
        # print(delta)
        pbar.set_description(f'New delta {delta}')
        if delta >= delta_orig:
            r = r+1
    pval = float(r+1.0)/(R+1.0)
    return pval


parser = argparse.ArgumentParser()
parser.add_argument('--dataA', type=str, nargs='*')
parser.add_argument('--dataB', type=str, nargs='*')
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--file_type', type=str, default='test')
args = parser.parse_args()

# print(args.dataA)
# print(args.dataB)

data_A = []
for dat in args.dataA:
    for datafile in glob(os.path.join(dat, f'{args.file_type}*.json')):
        with open(datafile, 'r') as infile:
            data_A += [json.loads(l) for l in infile]

data_B = []
for dat in args.dataB:
    for datafile in glob(os.path.join(dat, f'{args.file_type}*.json')):
        with open(datafile, 'r') as infile:
            data_B += [json.loads(l) for l in infile]

print(len(data_A))
print(f'Perf A: {calc_acc(data_A)}')
print(f'Perf B: {calc_acc(data_B)}')

R = 10000
pval = rand_permutation(data_A, data_B, len(data_A), R)

if float(pval) <= float(args.alpha):
    print(f"\nTest result is SIGNIFICANT =D with p-value: {pval}\n\n")
else:
    print(f"\nTest result is not significant =( with p-value: {pval}\n\n")