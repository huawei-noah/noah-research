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
  NOTE: remember to generate .idx for each `--out_file`. 
  index from 0 by ascending order
'''

import os
import argparse
import tempfile
import logging
import subprocess
import threading 

from jieba import Tokenizer as zh_Tokenizer
from MeCab import Tagger as ja_Tokenizer
from pythainlp.tokenize import Tokenizer as th_Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument("--in_fmt_label_data", type=str, help="tsv, phrase-level formatted real data with true labels") # NOTE: e.g. `original/m_atis-seq_tag/test/zh`
parser.add_argument("--in_trans_data", type=str, help="tsv, translated sentences w/ s_cls labels") # NOTE: e.g. `original/trans_m_atis-s_cls_tokenized/trans-test/zh`
parser.add_argument("--out_file", type=str, help="tsv, formatted like `in_fmt_label_data`")
parser.add_argument("--fast_align_root", type=str, default="./others/fast_align-master/build", help="tsv, formatted like `in_fmt_label_data`")
parser.add_argument("--lang", type=str, help="language")
args = parser.parse_args()

TOKENIZERS = {
    "zh": zh_Tokenizer(),
    "th": th_Tokenizer(),
    "ja": ja_Tokenizer("-Owakati"),
    }
DIM_ALIGN = " ||| "

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def read_tag_fmt_lines(fd, lang):
  lines = []
  toks, tok_labels = [], []
  for i, line in enumerate(fd):
    if line.strip():
      phrase, label = line.strip().split('\t')
      '''
      if lang == "zh":
        ptoks = [w for w,_,_ in tokenizer.tokenize(phrase.replace(' ', ''))]
      elif lang == "th":
        ptoks = tokenizer.word_tokenize(phrase.replace(' ', ''))
        if i < 5: logger.info(f" th tokenization: {phrase} -> {ptoks}")
      elif lang == "ja":
        ptoks = tokenizer.parse(phrase.replace(' ', '')).strip().split(' ')
      else:
        ptoks = phrase.strip().split(' ')
      '''
      ptoks = phrase.strip().split(' ')
      toks.extend(ptoks)
      tok_labels.extend([label] * len(ptoks))
    else:
      lines.append((toks, tok_labels))
      toks, tok_labels = [], []
  if toks:
    lines.append((toks, tok_labels))
    
  return lines


def preprocess(lang, label_data, trans_data): #, align_in_pairs):
  '''format for `fast-align`, tokenized pairs format (de-en):
    doch jetzt ist der Held gefallen . ||| but now the hero has fallen .
    neue Modelle werden erprobt . ||| new models are being tested .
    doch fehlen uns neue Ressourcen . ||| but we lack new resources .
  '''
  if lang in TOKENIZERS:
    tokenizer = TOKENIZERS[lang]
    logger.info(f" {lang} with special tokenizer = {tokenizer}")

  pairs = []
  labels = [] # label list
  with open(label_data, 'r') as filabeld, open(trans_data, 'r') as fitransd: # , open(align_in_pairs) as fopairs:
    llines = read_tag_fmt_lines(filabeld, lang)
    tlines = fitransd.readlines()
    tlines = tlines[1:]
    if not tlines[-1].strip(): tlines = tlines[:-1]
    assert len(llines) == len(tlines), f"len(labeled) != len(trans), {len(llines)} != {len(tlines)}"

    logger.info(f" [preprocess] ###### format pairs for alignment ######")
    for i, (lline, tline) in enumerate(zip(llines, tlines)):
      toks, tok_labels = lline
      tline = tline.split('\t')[0].strip()
      if lang == "zh":
        tline = [w for w,_,_ in tokenizer.tokenize(tline.replace(' ', ''))]
      elif lang == "th":
        tline = tokenizer.word_tokenize(tline.replace(' ', ''))
      elif lang == "ja":
        tline = tokenizer.parse(tline.replace(' ', '')).strip().split(' ')
      if lang in TOKENIZERS:
        tline = ' '.join(tline)
      pair = f"{' '.join(toks)}{DIM_ALIGN}{tline}\n" # NOTE: original sentence as the source (left) sentence to ensure each label align on only one token, non-aligned token would be labeled as `O`
      # fopairs.write(pair)
      pairs.append(pair)
      labels.append(tok_labels)
      if i < 5: logger.info(f" [preprocess] {pair}")
  return pairs, labels


def learn_alignments(pairs, pairs_file):
  fast_align_root = os.path.abspath(args.fast_align_root)
  fast_align = os.path.join(fast_align_root, 'fast_align')
  # atools = os.path.join(fast_align_root, 'atools')
  with open(pairs_file, 'w') as fout:
    for pair in pairs:
      fout.write(pair)

  fwd_cmd = [fast_align, '-i', pairs_file, '-d', '-o', '-v']
  proc = subprocess.run(fwd_cmd, stdout=subprocess.PIPE, encoding="utf-8")
  # return proc.stdout.read() # NOTE: `Popen()`'s `stdin.write()` / `stdout.read()` may lead to deadlock
  results = proc.stdout.split('\n')
  if not results[-1].strip(): results = results[:-1]
  assert len(pairs) == len(results), f"len(pairs) != len(results), {len(pairs)} != {len(results)}"
  logger.info(f" [learn_alignments] align {len(pairs)} pairs")
  for i in range(5):
    logger.info(f" [learn_alignments] pairs[{i}] = {pairs[i]}")
    logger.info(f" [learn_alignments] results[{i}] = {results[i]}")
  return results


def apply_alignments(args, pairs, alignments, labels, out_file):
  out_dir = os.path.dirname(out_file)
  os.makedirs(out_dir, exist_ok=True)

  logger.info(f" [apply_alignments] pairs = {len(pairs)}, alignments = {len(alignments)}, labels = {len(labels)}")

  with open(out_file, 'w') as fout:
    for i, (pair, aligns, label) in enumerate(zip(pairs, alignments, labels)):
      trans = pair.split(DIM_ALIGN)[1].strip().split(' ')
      trans2label = ['O'] * len(trans)
      for align in aligns.strip().split(' '):
        lab_idx, trans_idx = align.split('-')
        lab_idx, trans_idx = int(lab_idx), int(trans_idx)
        try:
          trans2label[trans_idx] = label[lab_idx]
        except IndexError:
          logger.error(f" [apply_alignments] pair = {pair}, label = {label}, alignments = {aligns}")
          raise
      for tok, lab in zip(trans, trans2label):
        fout.write(f"{tok}\t{lab}\n")
      fout.write("\n")
      if i < 5:
        logger.info(f" [apply_alignments] pair = {pair}, label = {label}, alignments = {aligns}")


def main():
  tmpdir_obj = tempfile.TemporaryDirectory()
  tmpdir = tmpdir_obj.name

  args.pairs_file = os.path.join(tmpdir, "align_input_pairs")
  # align_in_pairs = os.path.join(tmpdir, "align_in_pairs")
  # tmp_align_info = os.path.join(tmpdir, "align_info")

  pairs, labels = preprocess(args.lang, args.in_fmt_label_data, args.in_trans_data)
  alignments = learn_alignments(pairs, args.pairs_file)
  apply_alignments(args, pairs, alignments, labels, args.out_file)


if __name__ == "__main__":
  main()
