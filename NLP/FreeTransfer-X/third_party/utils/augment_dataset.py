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
import logging
from multiprocessing import Pool

# from nltk.translate.bleu_score import corpus_bleu
from jieba import Tokenizer as zh_Tokenizer
from MeCab import Tagger as ja_Tokenizer
from pythainlp.tokenize import Tokenizer as th_Tokenizer

from run_trans_m2m100 import M2M100

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--spec_langs", type=str, default="zh", help="only translate these languages")
parser.add_argument("--in_dir", type=str, default="", help="")
parser.add_argument("--out_dir", type=str, default="", help="")
parser.add_argument("--model_dir", type=str, help="")
parser.add_argument("--fields", type=str, default="0", help="field ids, comma-split & 0-start")
parser.add_argument("--fields_dim", type=str, default="\t", help="")
parser.add_argument("--aug_times", type=int, default=1, help="")
parser.add_argument("--sample_topk", type=int, default=5, help="")
parser.add_argument("--temperature", type=float, default=1., help="")
parser.add_argument("--n_procs", type=int, default=1, help="")
args = parser.parse_args()


def run_augment(lang, split_dir, out_split_dir):
  Tokenizers = {
      "zh": zh_Tokenizer(),
      "ja": ja_Tokenizer("-Owakati"),
      "th": th_Tokenizer(),
      }

  n_lines = 0
  pid = os.getpid()
  model = M2M100(args.model_dir)
  model.set_fields(args.fields.split(','), args.fields_dim)
  model.set_src_lang(lang)
  in_path = os.path.join(split_dir, lang)
  out_path = os.path.join(out_split_dir, lang)
  logger.info(f" [{pid}][run_augment]     [{in_path}] -> [{out_path}]")
  with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
    lines = fin.readlines()
    logger.info(f" [{pid}][run_augment]     loaded {len(lines)} lines.")
    fout.write(args.fields_dim.join([lines[0].strip(), "orig_idx"]) + '\n') # keep original headings & data 
    for k in range(args.aug_times):
      for i, line in enumerate(lines[1:]): 
        if lang in Tokenizers:
          line = line.replace(' ', '')
        trans_lines = model.translate_fields(line.strip(), lang, do_sample=True, num_return_sequences=args.sample_topk, temperature=args.temperature)
        for j, trans_line in enumerate(set(trans_lines)):
          if lang == "zh":
            trans_sent, label = trans_line.split(args.fields_dim)
            trans_words = ' '.join([w for w, _, _ in Tokenizers[lang].tokenize(trans_sent)]).strip()
            fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
          elif lang == "ja":
            trans_sent, label = trans_line.split(args.fields_dim)
            trans_words = Tokenizers[lang].parse(trans_sent).strip()
            fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
          elif lang == "th":
            trans_sent, label = trans_line.split(args.fields_dim)
            trans_words = ' '.join(Tokenizers[lang].word_tokenize(trans_sent)).strip()
            fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
          else:
            fout.write(args.fields_dim.join([trans_line.strip(), str(i)]) + '\n')
          n_lines += 1
          # if ((k*len(lines)+i)*args.sample_topk+j) % 500 == 0: logger.info(f" [{pid}][run_augment]     {(k*len(lines)+i)*args.sample_topk+j} lines")
          if n_lines % 500 == 0: logger.info(f" [{pid}][run_augment]     {n_lines} lines")
  return (lang, n_lines)


def main():
  ''' file struct should be:
    p_dir
      - train
        - lang_center
        - lang1
        ...
        - langk
      - dev
      - test
  '''

  # model = M2M100(args.model_dir)
  # model.set_fields(args.fields.split(','), args.fields_dim)

  os.makedirs(args.out_dir, exist_ok=True)

  for p_dir, splits, _ in os.walk(args.in_dir):
    logger.info(f" [main] {p_dir}")
    for split in splits:
      if split != "train": continue
      logger.info(f" [main]   {split}")
      split_dir = os.path.join(p_dir, split)
      langs = os.listdir(split_dir)
      out_split_dir = os.path.join(args.out_dir, "aug-" + split)
      os.makedirs(out_split_dir, exist_ok=True)
      # with Pool(args.n_procs) as pool:
      pool = Pool(args.n_procs)
      results = []
      for lang in langs:
        if lang not in args.spec_langs.split(','): continue
        results.append(pool.apply_async(run_augment, (lang, split_dir, out_split_dir)))
      for res in results:
        lang, cnt = res.get()
        logger.info(f" [main]     {lang} {cnt} lines")
      pool.close()
      pool.join()
    break


if __name__ == "__main__":
  main()
