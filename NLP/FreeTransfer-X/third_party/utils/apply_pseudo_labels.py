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
'''
data/trans-train file: with headings, all lines are aligned to `train`
pseudo-label file:  w/o headings, other lines are aligned to `train` & `trans-train`
'''

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_data", type=str, help="")
parser.add_argument("--in_labels", type=str, help="")
parser.add_argument("--out_data", type=str, help="")
parser.add_argument("--dim", type=str, default="\t", help="")
parser.add_argument("--label_idx", type=int, default=1, help="")
args = parser.parse_args()

def main():
  in_data = []
  in_labels = []
  with open(args.in_data, 'r') as find, open(args.in_labels, 'r') as finl:
    in_data = find.readlines()
    in_labels = finl.readlines()
    
  with open(args.out_data, 'w') as foutd:
    foutd.write(in_data[0]) # headings
    for line, pseudo_label in zip(in_data[1:], in_labels):
      fields = line.strip().split(args.dim)
      fields[args.label_idx] = pseudo_label.strip()
      foutd.write(
          args.dim.join(fields) + '\n'
          )

if __name__ == "__main__":
  main()
