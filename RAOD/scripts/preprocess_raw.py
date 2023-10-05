# Copyright 2023 Huawei Technologies Co., Ltd
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

import os
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
import cv2
import gzip
import shutil
import argparse
import numpy as np
import multiprocessing

from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed

import torch
import torch.nn as nn
import torch.nn.functional as F


BIT8  = 2 ** 8
BIT16 = 2 ** 16
BIT24 = 2 ** 24

class Debayer3x3(nn.Module):
    # This code is adjusted from the following url
    # https://github.com/cheind/pytorch-debayer/blob/master/debayer/modules.py

    def __init__(self):
        super(Debayer3x3, self).__init__()

        self.kernels = nn.Parameter(
            torch.tensor([
                [0,0,0],
                [0,1,0],
                [0,0,0],
                
                [0, 0.25, 0],
                [0.25, 0, 0.25],
                [0, 0.25, 0],
                
                [0.25, 0, 0.25],
                [0, 0, 0],
                [0.25, 0, 0.25],
                
                [0, 0, 0],
                [0.5, 0, 0.5],
                [0, 0, 0],
                
                [0, 0.5, 0],
                [0, 0, 0],
                [0, 0.5, 0],
            ]).view(5,1,3,3), requires_grad=False
        )
        
        self.index = nn.Parameter(
            torch.tensor([
                # dest channel r
                [0, 3], # pixel is R,G1
                [4, 2], # pixel is G2,B
                # dest channel g
                [1, 0], # pixel is R,G1
                [0, 1], # pixel is G2,B
                # dest channel b
                [2, 4], # pixel is R,G1
                [3, 0], # pixel is G2,B
            ]).view(1,3,2,2), requires_grad=False
        )
        
    def forward(self, x):
        
        B,C,H,W = x.shape

        x = F.pad(x, (1,1,1,1), mode='replicate')
        c = F.conv2d(x, self.kernels, stride=1)
        rgb = torch.gather(c, 1, self.index.repeat(B,1,H//2,W//2))
        return rgb


def read_raw_24b(file_path, img_shape=(1, 1, 1856, 2880), read_type=np.uint8):
    raw_data = np.fromfile(file_path, dtype=read_type)
    raw_data = raw_data[0::3] + raw_data[1::3] * BIT8 + raw_data[2::3] * BIT16
    raw_data = raw_data.reshape(img_shape).astype(np.float32)
    
    return raw_data


def func(filename, debayer, out_path):
    im = read_raw_24b(filename)
    im = torch.from_numpy(im).cuda().float()

    with torch.no_grad():
        im = debayer(im).detach().cpu().numpy()
    
    im = im.squeeze(0).transpose(1, 2, 0)
    im = cv2.resize(im, (1280, 1280), interpolation=cv2.INTER_LINEAR)

    mean_r = im[:, :, 0].mean()
    mean_g = im[:, :, 1].mean()
    mean_b = im[:, :, 2].mean()
    im[:, :, 0] *= mean_g / mean_r
    im[:, :, 2] *= mean_g / mean_b
    im = np.clip(im, 0, BIT24 - 1) / (BIT24 - 1) * (BIT8 - 1)
    
    save_path = os.path.join(out_path, os.path.basename(filename))
    with gzip.GzipFile(save_path.replace('.raw', '.npy.gz'), 'w') as f:
        np.save(file=f, arr=im)


def main(args):
    in_path  = os.path.realpath(args['path']).rstrip('/') + '/'
    out_path = in_path.replace('/raws/', '/raws_debayer_awb_fp32_1280x1280/')
    assert os.path.isdir(in_path), f'Invalid path < {args["path"]} >!'
    assert in_path != out_path, f'in_path should NOT be the same as out_path!'

    shutil.rmtree(out_path, ignore_errors=True); os.makedirs(out_path)
    print(f'input  path: {in_path}')
    print(f'output path: {out_path}')

    lines = glob(os.path.join(in_path, '*.raw'))
    print(f'{len(lines)} raw images found')
    debayer = Debayer3x3().cuda()

    if args['threads'] in [0, 1]:
        print('Single thread')
        for fn in tqdm(lines):
            func(fn, debayer, out_path)
    
    else:
        if args['threads'] == -1:
            threads = multiprocessing.cpu_count() // 4
        else: threads = args['threads']
        print(f'{threads} threads')

        para = Parallel(n_jobs=threads, backend='threading')
        para(delayed(func)(filename, debayer, out_path) for filename in tqdm(lines))

    files_out = glob(os.path.join(out_path, '*.npy.gz'))
    print(f'output number: {len(files_out)}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path',    type=str, default='/home/data')
    parser.add_argument('-t', '--threads', type=int, default=-1)
    args = parser.parse_args()
    args = args.__dict__
    main(args)
