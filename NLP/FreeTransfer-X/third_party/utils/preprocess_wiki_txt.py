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


'''should be applied after `WikiExtract` like: 
  python3 -m wikiextractor.WikiExtractor urwiki-20200520-pages-articles.xml.bz2 --processes 8 -q -o - | sed "/^\s*\$/d" | grep -v "^<doc id=" | grep -v "</doc>\$"
'''

import os
import argparse
import subprocess
import logging 
import re
import shutil


import pythainlp
from nltk.tokenize import sent_tokenize
from tokenizers.trainers import BpeTrainer

from build_vocabs import *

FORMAT = "%(asctime)-15s: %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

LANG_ISO639 = {
    "czech": "cs",
    "danish": "da",
    "dutch": "nl",
    "english": "en",
    "estonian": "et",
    "finnish": "fi",
    "french": "fr",
    "german": "de",
    "greek": "el",
    "italian": "it",
    "norwegian": "no",
    "polish": "pl",
    "portuguese": "pt",
    "slovene": "sl",
    "spanish": "es",
    "swedish": "sv",
    "turkish": "tr",
    }

ISO639_LANG = {v: k for k, v in LANG_ISO639.items()}

def sent_tokenize_zh(sents):
  sents = re.sub('([。！？\?])([^”])',r"\1\n\2", sents)
  sents = re.sub('(\.{6})([^”])',r"\1\n\2", sents)
  sents = re.sub('(\…{2})([^”])',r"\1\n\2", sents)
  sents = re.sub('(”)','”\n', sents)
  sents = sents.rstrip()
  return sents.split("\n")

SENT_TOKENIZE = {
    "th": pythainlp.tokenize.sent_tokenize,
    "zh": sent_tokenize_zh,
    }

parser = argparse.ArgumentParser()
parser.add_argument("--langs", type=str, help="languages")
parser.add_argument("--vocab_sizes", type=str, help="")
parser.add_argument("--data_pat", type=str, help="")
parser.add_argument("--data_dir", type=str, help="local cache")
parser.add_argument("--out_dir", type=str, help="local output")
parser.add_argument("--preprocess", action="store_true", help="")
parser.add_argument("--build_vocab", action="store_true", help="")
parser.add_argument("--bpe_package", type=str, default="subword-nmt", help="local output")
args = parser.parse_args()


def subprocess_pipe(cmd_args, fin=None, fout=None):
  def parse_args(args_str, dim):
    splits = []
    naive_splits = args_str.split(dim)
    s_stack = []
    stack_on = False
    for split in naive_splits:
      if split[0] == '"':
        stack_on = True
      if stack_on:
        s_stack.append(split)
      else:
        splits.append(split.strip('"'))
      if stack_on and split[-1] == '"':
        stack_on = False
        splits.append(dim.join(s_stack).strip('"'))
    return splits

  results = []
  cmds = cmd_args.split('|')
  STDOUT = subprocess.PIPE
  cmd = parse_args(cmds[0].strip(), ' ')
  logger.info(f" [subprocess_pipe] stdout={STDOUT}, {cmd}")
  results.append(subprocess.Popen(cmd, stdin=fin, stdout=STDOUT))
  for i, cmd in enumerate(cmds[1:]):
    if i == len(cmds)-2: # the last cmd
      STDOUT = fout
    else:
      STDOUT = subprocess.PIPE
    cmd = parse_args(cmd.strip(), ' ')
    logger.info(f" [subprocess_pipe] stdout={STDOUT}, {cmd}")
    results.append(subprocess.Popen(cmd, stdin=results[-1].stdout, stdout=STDOUT))
  # for res in results[:-1]:
  #   res.stdout.close()
  return results[-1].communicate()[0]


def fetch_and_preprocess(lang):
  '''
    in:   .xml.bz2
    out:  sentences
  '''
  path = args.data_pat.format(lang)
  filename = os.path.basename(path)
  xlm_file = os.path.join(args.data_dir, filename)
  clean_file = os.path.join(args.data_dir, filename + ".clean")
  out_file = os.path.join(args.data_dir, filename + ".sents")

  if args.preprocess:
    logger.info(f" [fetch_and_preprocess] copying {path} to {xlm_file}")
    shutil.copyfile(path, xlm_file)

    # wikiextract & clean
    cmd_args = f'python3 -m wikiextractor.WikiExtractor {xlm_file} --processes 32 -q -o - | sed "/^\s*$/d" | grep -v "^<doc id=" | grep -v "</doc>$"'
    with open(clean_file, 'w') as fout:
      res = subprocess_pipe(cmd_args, fout=fout)

    # sentence tokenization
    logger.info(f" [fetch_and_preprocess] tokenizing {clean_file}")
    tokenize = SENT_TOKENIZE.get(lang, sent_tokenize)
    with open(clean_file, 'r') as inf, open(out_file, 'w') as outf:
      for i, line in enumerate(inf):
        # if i >= 100: break
        try:
          sents = tokenize(line.strip(), language=ISO639_LANG.get(lang, "english"))
        except TypeError: # Non-NLTK
          sents = tokenize(line.strip())
        for sent in sents:
          outf.write(sent + '\n')

  else:
    logger.info(f" [fetch_and_preprocess] copying {path} to {out_file}")
    shutil.copyfile(path, out_file)

  return out_file


def build_and_upload(lang, in_path, vocab_size):
  trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=vocab_size) # FIXME: bugs? `vocab_size` may not take effect
  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)
  build_vocab(lang, "wiki", in_path, args.out_dir, vocab_size, trainer, spec_file=True, bpe_package=args.bpe_package)


def main():
  logger.info(f" [main] args = {args}")
  for lang in args.langs.split(','):
    sent_file = fetch_and_preprocess(lang)
    if args.build_vocab:
      for vocab_size in args.vocab_sizes.split(','):
        build_and_upload(lang, sent_file, int(vocab_size))

if __name__ == "__main__":
  main()
