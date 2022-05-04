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


'''no tokenization.
  convert fmt into .tsv, w/ a heading line
  single_cls: sent\tlabel\n
  pair_cls: sent1\tsent2\tlabel\n
  seq_tag:
    text_file: e.g. `en.text`
      sent-0_token-0\tlabel-0\n
      ...                      
      sent-0_token-n\tlabel-n\n
      \n                       
      ...
      sent-k_token-0\tlabel-0\n
      ...                      
      sent-k_token-n\tlabel-n\n
    idx_file: `en.text.idx`
      0\n
      ...
      0\n
      \n 
      ...
      k\n
      ...
      k\n
'''

import os
import re
import argparse
import logging
import json

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

parser = argparse.ArgumentParser()
parser.add_argument("--task_type", type=str, default="s_cls", help="s_cls, p_cls, seq_tag")
parser.add_argument("--task_name", type=str, default="mtop", help="mtop, m_atis")
parser.add_argument("--in_dir", type=str, help="task's root dir that contains lang-spec subdirs")
parser.add_argument("--out_dir", type=str, help="")
args = parser.parse_args()

TASK_EXT = {
    "mtop": {
      "s_cls": [".txt"],
      "seq_tag": [".txt"],
      },
    "m_atis": {
      "s_cls": [".txt"],
      "seq_tag": [".txt"],
      },
    }

TASK_DIM_RAW = {
    "mtop": {
      "s_cls": "\t",
      "seq_tag": "\t",
      },
    "m_atis": {
      "s_cls": "\t",
      "seq_tag": "\t",
      },
    }

LABEL_DIM = {
    "mtop": {
      "s_cls": "",
      "seq_tag": ",",
      },
    "m_atis": {
      "s_cls": "#| ", # FIXME: shouldn't contain multi-labels, but in fact the raw data contains
      "seq_tag": "-",
      },
    }

SET_TYPE_NORM = {
    "test": "test",
    "dev": "dev",
    "eval": "dev",
    "train": "train",
    }

TASKS_WITH_HEAD = ["m_atis"]

def parse_raw(in_file, task_name, task_type):
  examples = []
  dim = TASK_DIM_RAW[task_name][task_type]
  with open(in_file, 'r') as fin:
    for i, line in enumerate(fin):
      if task_name in TASKS_WITH_HEAD and i < 1: continue
      ex = {}
      fields = line.strip().split(dim)
      if task_name == "mtop":
        ex["id"], ex["intents"], ex["slots"], ex["sent"], ex["domain"], ex["lang"], ex["decoupled_form"], ex["tokens"] = fields
      elif task_name == "m_atis":
        ex["id"], ex["sent"], ex["slots"], ex["intents"] = fields
      examples.append(ex)
  return examples


def format_tsv(examples, task_name, task_type, out_file):
  logger.info(f" [format_tsv] formating {out_file}, {len(examples)} exs")
  file_labels = set()
  with open(out_file, 'w') as fout:
    if task_type == "s_cls":
      fout.write("sent\tlabel\n")
    elif task_type == "p_cls":
      pass
    elif task_type == "seq_tag":
      pass

    if task_type == "seq_tag":
      fidx = open(out_file + ".idx", 'w')

    for idx, ex in enumerate(examples):
      if task_type == "s_cls":
        fout.write("{}\t{}\n".format(ex["sent"], ex["intents"]))
        if LABEL_DIM[task_name][task_type]:
          # file_labels.update(re.split(LABEL_DIM[task_name][task_type], ex["intents"]))
          file_labels.add(re.split(LABEL_DIM[task_name][task_type], ex["intents"])[0]) # NOTE: should be 1-label classification, only take the first label
        else:
          file_labels.add(ex["intents"])
      elif task_type == "p_cls":
        pass
      elif task_type == "seq_tag":
        if task_name == "mtop":
          # NOTE: `sent` is not tokenized, instead `tokens` is
          # we use `tokens` since it's aligned with `slots` labels
          try:
            tokens_raw = json.loads(ex["tokens"])
          except json.decoder.JSONDecodeError:
            logger.warning(ex["tokens"])
            tokens_raw = json.loads(ex["tokens"].replace('"""', '"\\""'))
            # raise
          tokens, token_spans = tokens_raw["tokens"], tokens_raw["tokenSpans"]
          labels_raw = ex["slots"].split(',')
          labels = ['O'] * len(tokens)
          offset2label = {}
          for label_raw in labels_raw:
            if not label_raw.strip(): continue # skip null label_raw
            try:
              start, end, _, label = label_raw.split(':') # fmt: start_byte:end_byte:SL:slot_name
            except ValueError:
              logger.error(f" [format_tsv] ex = {ex}")
              logger.error(f" [format_tsv] label_raw = {label_raw}")
              raise
            for offset in range(int(start), int(end)):
              offset2label[offset] = label
          for tok_i, span in enumerate(token_spans):
            if span["start"] in offset2label:
              labels[tok_i] = offset2label[span["start"]]
        elif task_name == "m_atis":
          tokens = ex["sent"].split(' ') # NOTE: zh `sent` is tokenized
          labels = ex["slots"].split(' ')
          # for word, label in zip(ex["sent"].split(' '), ex["slots"].split(' ')):

        for l_i in range(len(labels)): # replace null labels as 'O'
          if not labels[l_i].strip():
            labels[l_i] = 'O'
        for word, label in zip(tokens, labels):
          fout.write(f"{word}\t{label}\n") # NOTE: did NOT distinguish B-/I-
          fidx.write(f"{idx}\n")
        fout.write(f"\n")
        fidx.write(f"\n")
        file_labels = file_labels.union(labels)

    if task_type == "seq_tag":
      fidx.close()

  logger.info(f" [format_tsv] done. {len(file_labels)} file_labels: {file_labels}")
  return file_labels


def main():
  all_labels = set()
  for p_dir, sub_dirs, _ in os.walk(args.in_dir):
    for lang in sub_dirs:
      lang_labels = set()
      for pp_dir, _, files in os.walk(os.path.join(p_dir, lang)):
        if args.task_name == "mtop":
          in_files = [os.path.join(pp_dir, filename) for filename in files if os.path.splitext(filename)[-1] in TASK_EXT[args.task_name][args.task_type]]
        elif args.task_name == "m_atis":
          in_files = [os.path.join(pp_dir, filename) for filename in files if os.path.splitext(filename)[-1] in TASK_EXT[args.task_name][args.task_type]]
        else:
          raise NotImplementedError(f" unknown args.task_name {args.task_name}")
        for in_file in in_files:
          examples = parse_raw(in_file, args.task_name, args.task_type)
          set_type = ""
          for ftype in SET_TYPE_NORM:
            if ftype in os.path.basename(in_file):
              set_type = SET_TYPE_NORM[ftype]
              break
          work_dir = os.path.join(args.out_dir, f"{args.task_name}-{args.task_type}", set_type)
          os.makedirs(work_dir, exist_ok=True)
          out_file = os.path.join(work_dir, lang)
          lang_labels.update(format_tsv(examples, args.task_name, args.task_type, out_file))
        break
      logger.info(f" [format_tsv] {lang}: {len(lang_labels)} lang_labels: {lang_labels}")
      all_labels.update(lang_labels)
    break
  logger.info(f" [format_tsv] {len(all_labels)} all_labels: {all_labels}")

if __name__ == "__main__":
  main()
