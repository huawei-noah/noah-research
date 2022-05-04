#!/usr/bin/env python3
# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

'''split train set into 2 parts:
  1 for labeled training -> mono-src
  1 for unlabeled training
'''

import os
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", type=str, help="")
parser.add_argument("--out_dirs", type=str, help="e.g. `labeled`, `unlabeled`")
parser.add_argument("--task_type", type=str, help="e.g. `scls`, `tag`")
parser.add_argument("--seed", type=int, default=42, help="")
args = parser.parse_args()

IDX_SUF = ".idx"

def main():
  out_dirs = args.out_dirs.split(',')
  n_outs = len(out_dirs)

  filename = os.path.basename(args.in_file)
  dirname = os.path.basename(os.path.dirname(args.in_file))
  out_paths = []
  for out_p_dir in out_dirs:
    out_dir = os.path.join(out_p_dir, dirname)
    os.makedirs(out_dir, exist_ok=True)
    out_paths.append(os.path.join(out_dir, filename))

  in_data = []
  with open(args.in_file, 'r') as fin:
    in_data = fin.readlines()

  if args.task_type == "scls":
    headings, data = in_data[0], in_data[1:]
  elif args.task_type == "tag": # 211205
    in_data_idx = []
    with open(args.in_file + IDX_SUF, 'r') as fin:
      in_data_idx = fin.readlines()
    assert len(in_data) == len(in_data_idx), f" [main] len(in_data) != len(in_data_idx): {len(in_data)} != {len(in_data_idx)}"
    data = []
    sent = []
    for token, idx in zip(in_data, in_data_idx):
      ''' fmt of `token`, `idx` refers to `./convert_fmt.py` '''
      if token == '\n':
        data.append(sent)
        sent = []
      else:
        sent.append((token, idx)) # NOTE: '\n' kept
    if sent: data.append(sent)
  else:
    raise NotImplementedError(f" [main] unknown task_type = {args.task_type}")

  n_data = len(data)
  s_split = n_data // n_outs
  random.seed(args.seed)
  random.shuffle(data)
  for i, out_path in enumerate(out_paths):
    if i == len(out_paths) - 1:
      split = data[i*s_split:]
    else:
      split = data[i*s_split:(i+1)*s_split]
    if args.task_type == "scls":
      with open(out_path, 'w') as fout:
        fout.write(headings)
        for line in split:
          fout.write(line)
    elif args.task_type == "tag": # 211205
      with open(out_path, 'w') as fout, open(out_path + IDX_SUF, 'w') as fouti:
        for sent in split:
          for token, idx in sent:
            fout.write(token)
            fouti.write(idx)
          fout.write('\n')
          fouti.write('\n')


if __name__ == "__main__":
  main()
