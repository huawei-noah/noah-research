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

'''extract translations 
  from s_cls sent-level translations
  according to seq_tag train src.idx
'''


import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--in_unlabeled_idx", type=str, help="`unlabeled data`'s index mapping to the original data")
parser.add_argument("--in_original_trans", type=str, help="tsv, `original data`'s translated sentences w/ s_cls labels") # NOTE: e.g. `original/trans_m_atis-s_cls_tokenized/trans-train/zh`
parser.add_argument("--out_trans_data", type=str, help="tsv, `unlabeled data`'s translated sentences, taken as the `--in_trans_data` of `tag_map_label_to_trans.py`") # NOTE: e.g. `original/m_atis-seq_tag/test/zh`
args = parser.parse_args()


def main():
  idx = []
  with open(args.in_unlabeled_idx, 'r') as fidx:
    for line in fidx:
      line = line.strip()
      if line and int(line) not in idx:
        idx.append(int(line))
  print(f" len(idx) = {len(idx)}")

  otrans = []
  heading = ""
  with open(args.in_original_trans, 'r') as fotrans:
    for i, line in enumerate(fotrans):
      if i == 0:
        heading = line
        continue
      otrans.append(line)

  dirname = os.path.dirname(args.out_trans_data)
  os.makedirs(dirname, exist_ok=True)
  with open(args.out_trans_data, 'w') as futrans:
    futrans.write(heading)
    for i in idx:
      futrans.write(otrans[i])


if __name__ == "__main__":
  main()
