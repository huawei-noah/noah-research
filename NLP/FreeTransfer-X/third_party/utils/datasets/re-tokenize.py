#!/usr/bin/env python
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
parser.add_argument("--in_file", type=str)
parser.add_argument("--out_file", type=str)
parser.add_argument("--lang", type=str)
parser.add_argument("--field", default="0", type=str)
parser.add_argument("--n_skip_rows", default=1, type=int)
args = parser.parse_args()

TOKENIZERS = {
    "zh": zh_Tokenizer(),
    "th": th_Tokenizer(),
    "ja": ja_Tokenizer("-Owakati"),
    }

def main():
  os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
  field = int(args.field)

  tokenizer = TOKENIZERS[args.lang]
  
  with open(args.in_file, 'r') as fin, open(args.out_file, 'w') as fout:
    for line in fin:
      fields = line.strip().split('\t')
      sent = fields[field].strip()
      if args.lang == "zh":
        words = [w for w,_,_ in tokenizer.tokenize(sent.replace(' ', ''))]
      elif args.lang == "th":
        words = tokenizer.word_tokenize(sent)
      elif args.lang == "ja":
        words = tokenizer.parse(sent.replace(' ', '')).strip().split(' ')
      else:
        raise NotImplementedError(" lang = {args.lang}") 
      fields[field] = ' '.join(words)
      fout.write('\t'.join(fields) + '\n')

if __name__ == "__main__":
  main()
