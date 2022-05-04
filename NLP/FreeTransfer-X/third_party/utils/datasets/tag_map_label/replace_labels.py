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


import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_ori_data", type=str, help="tsv, phrase-level formatted real data with true labels") # NOTE: e.g. `unlabeled/m_atis-seq_tag/train/en`
parser.add_argument("--in_new_label", type=str, help="label list") # NOTE: e.g. model's prediction `test-en.tsv`
parser.add_argument("--out_data", type=str, help="tsv, with new labels")
args = parser.parse_args()


def main():
  dirname = os.path.dirname(args.out_data)
  os.makedirs(dirname, exist_ok=True)

  ori = []
  labels = []
  with open(args.in_ori_data, 'r') as fori, open(args.in_new_label, 'r') as flab, open(args.out_data, 'w') as fout:
    for line in fori:
      line = line.strip()
      if line:
        fields = line.split('\t')
        tok, lab = fields[0], fields[1]
        ori.append(tok)
      else:
        ori.append('\n')
    if ori[-1] == '\n': ori = ori[:-1]

    for line in flab:
      line = line.strip()
      if line:
        labels.append(line)
      else:
        labels.append('\n')

    assert len(ori) == len(labels), f" len(ori) = {len(ori)}, len(labels) = {len(labels)}"

    for tok, lab in zip(ori, labels):
      if tok == '\n':
        fout.write(tok)
      else:
        fout.write(f"{tok}\t{lab}\n")



if __name__ == "__main__":
  main()
