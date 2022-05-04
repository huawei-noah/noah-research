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

from jieba import Tokenizer as zh_Tokenizer
from pythainlp.tokenize import Tokenizer as th_Tokenizer
from MeCab import Tagger as ja_Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--in_file", default="", help="formatted data file")
parser.add_argument("--out_dir", default="", help="")
parser.add_argument("--lang", default="", help="default is the `in_file`'s filename")
args = parser.parse_args()

DIM = "\t"

def main():
  filename = os.path.basename(args.in_file)
  idx_file = args.in_file + ".idx"
  out_file = os.path.join(args.out_dir, filename)
  out_idx_file = out_file + ".idx"

  os.makedirs(args.out_dir, exist_ok=True)
  if os.path.isfile(out_file): raise RuntimeError(f" {out_file} existed")

  lang = args.lang if args.lang else filename

  if lang == "zh":
    pre_tokenizer = zh_Tokenizer()
    pre_tokenize_func = pre_tokenizer.tokenize
  elif lang == "th":
    pre_tokenizer = th_Tokenizer()
    pre_tokenize_func = pre_tokenizer.word_tokenize
  elif lang == "ja":
    pre_tokenizer = ja_Tokenizer("-Owakati")
    pre_tokenize_func = pre_tokenizer.parse
  else:
    pre_tokenize_func = None

  with open(args.in_file, 'r') as fin, open(idx_file, 'r') as fini, open(out_file, 'w') as fout, open(out_idx_file, 'w') as fouti:
    for line, idx in zip(fin, fini):
      if line.strip():
        phrase, label = line.strip().split(DIM)
        if lang == "zh":
          words = ' '.join([w for w, _, _ in pre_tokenize_func(phrase)])
        elif lang == "ja":
          words = pre_tokenize_func(phrase).strip()
        else:
          words = ' '.join(pre_tokenize_func(phrase))
        for word in words.split(' '):
          fout.write(f"{word}{DIM}{label}\n")
          fouti.write(f"{idx.strip()}\n")
      else:
        fout.write('\n')
        fouti.write('\n')


if __name__ == "__main__":
  main()
