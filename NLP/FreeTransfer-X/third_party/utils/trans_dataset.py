#!/usr/bin/env python3

import os
import argparse
import logging
from shutil import copyfile

from nltk.translate.bleu_score import corpus_bleu

from run_trans_m2m100 import M2M100

from jieba import Tokenizer as zh_Tokenizer
from MeCab import Tagger as ja_Tokenizer
from pythainlp.tokenize import Tokenizer as th_Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument("--lang_center", type=str, default="en", help="trans-test to this, trans-train from this")
parser.add_argument("--in_dir", type=str, default="", help="")
parser.add_argument("--out_dir", type=str, default="", help="")
parser.add_argument("--model_dir", type=str, help="")
parser.add_argument("--fields", type=str, default="0", help="field ids, comma-split & 0-start")
parser.add_argument("--fields_dim", type=str, default="\t", help="")
parser.add_argument("--test_bleu", action="store_true", help="")
parser.add_argument("--test_model", action="store_true", help="")
parser.add_argument("--task_type", type=str, default="s_cls", help="e.g. `s_cls`, `seq_tag`")
parser.add_argument("--skip_splits", type=str, default="", help="splits to be skipped, comma-splitted")
parser.add_argument("--skip_langs", type=str, default="", help="languages to be skipped, comma-splitted")
args = parser.parse_args()

Tokenizers = {
    "zh": zh_Tokenizer(),
    "ja": ja_Tokenizer("-Owakati"),
    "th": th_Tokenizer(),
    }

IDX_EXT = ".idx"


def compute_corpus_bleu(path1, path2, fid, char=False):
  hyps, refs = [], []
  with open(path1, 'r') as fin1, open(path2, 'r') as fin2:
    for line in fin1:
      if char:
        tokens = [ch for ch in line.split(args.fields_dim)[fid].strip()]
      else:
        tokens = line.split(args.fields_dim)[fid].strip().split(' ')
      hyps.append(tokens)
      logger.info(f" [compute_corpus_bleu] hyp: [{tokens}]")
    for line in fin2:
      if char:
        tokens = [ch for ch in line.split(args.fields_dim)[fid].strip()]
      else:
        tokens = line.split(args.fields_dim)[fid].strip().split(' ')
      refs.append([tokens])
      logger.info(f" [compute_corpus_bleu] ref: [{tokens}]")
  if len(hyps) == len(refs):
    return corpus_bleu(refs, hyps)
  else:
    logger.error(f" [compute_corpus_bleu] len(hyps) = {len(hyps)}, len(refs) = {len(refs)}")
    return -1


def retokenize(trans_line, lang):
  trans_words = ""
  label = ""
  trans_line = trans_line.replace(' ', '')
  if lang == "zh":
    trans_sent, label = trans_line.split(args.fields_dim)
    trans_words = ' '.join([w for w, _, _ in Tokenizers[lang].tokenize(trans_sent)]).strip()
    # fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
  elif lang == "ja":
    trans_sent, label = trans_line.split(args.fields_dim)
    trans_words = Tokenizers[lang].parse(trans_sent).strip()
    # fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
  elif lang == "th":
    trans_sent, label = trans_line.split(args.fields_dim)
    trans_words = ' '.join(Tokenizers[lang].word_tokenize(trans_sent)).strip()
    # fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
  else:
    logger.error(f" [retokenize] no tokenizer for {lang}")
  return trans_words, label


def trans_from_center(model, split_dir, langs, out_split_dir):
  model.set_src_lang(args.lang_center)
  in_path = os.path.join(split_dir, args.lang_center)
  for lang in langs:
    if os.path.splitext(lang)[-1] == IDX_EXT: continue
    lowercase = False
    out_path = os.path.join(out_split_dir, lang)
    logger.info(f" [trans_from_center]     [{in_path}] -> [{out_path}]")
    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
      for i, line in enumerate(fin): 
        if args.task_type == "s_cls":
          if i < 1:
            fout.write(args.fields_dim.join([line.strip(), "orig_idx"]) + '\n') # keep original headings
          else:
            trans_line = model.translate_fields(line.strip(), lang)[0]
            if lang in Tokenizers:
              trans_words, label = retokenize(trans_line, lang)
              if trans_words and label:
                fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
            else:
              fout.write(args.fields_dim.join([trans_line.strip(), str(i)]) + '\n')
        elif args.task_type == "seq_tag": # 211212
          if line.strip():
            trans_line = model.translate_fields(line.strip(), lang)[0]
            if lang in Tokenizers:
              trans_words, label = retokenize(trans_line, lang)
              token_status = True
              if (not trans_words) or (not label): # failed to retokenize
                # fout.write(args.fields_dim.join([line.strip(), str(i), str(False)]) + '\n')
                token_status = False
                fields = line.strip().split(args.fields_dim)
                trans_words, label = fields[0], fields[1]
            else:
              # fout.write(args.fields_dim.join([trans_line.strip(), str(i)]) + '\n')
              token_status = False
              fields = trans_line.strip().split(args.fields_dim)
              try: # 211213
                trans_words, label = fields[0], fields[1]
              except IndexError:
                logger.warning(f" [trans_from_center] len(fields) = {fields}, fields = {fields}, trans_line = [{trans_line}]")
                ori_fields = line.strip().split(args.fields_dim) # NOTE: keep the original text
                trans_words, label = ori_fields[0], ori_fields[1]
            if lowercase: trans_words = trans_words.lower()
            fout.write(args.fields_dim.join([trans_words, label, str(i), str(token_status)]) + '\n')
            lowercase = True
          else:
            fout.write('\n')
            lowercase = False
        else:
          raise NotImplementedError(f" [trans_from_center] unknown task_type = {args.task_type}")
        if i % 500 == 0: logger.info(f" [trans_from_center]     {i} lines")
      logger.info(f" [trans_from_center]     {i} lines")
    # for fid in args.fields:
    #   ori_path = os.path.join(split_dir, lang)
    #   logger.info(f" [trans_from_center]     bleu = {compute_corpus_bleu(out_path, ori_path, int(fid))}, hyps = [{out_path}], refs = [{ori_path}]")
    if args.task_type == "seq_tag": # 211212
      copyfile(in_path + IDX_EXT, out_path + IDX_EXT)


def trans_to_center(model, split_dir, langs, out_split_dir):
  for lang in langs:
    if os.path.splitext(lang)[-1] == IDX_EXT: continue
    lowercase = False
    model.set_src_lang(lang)
    in_path = os.path.join(split_dir, lang)
    out_path = os.path.join(out_split_dir, lang)
    logger.info(f" [trans_to_center]     [{in_path}] -> [{out_path}]")
    with open(in_path, 'r') as fin, open(out_path, 'w') as fout:
      for i, line in enumerate(fin): 
        if args.task_type == "s_cls":
          if i < 1:
            fout.write(args.fields_dim.join([line.strip(), "orig_idx"]) + '\n') # keep original headings
          else:
            if lang in Tokenizers:
              line = line.replace(' ', '')
            trans_line = model.translate_fields(line.strip(), args.lang_center)[0].strip()
            fout.write(args.fields_dim.join([trans_line, str(i)]) + '\n')
        elif args.task_type == "seq_tag": # 211212
          if line.strip():
            if lang in Tokenizers:
              line = line.replace(' ', '')
            trans_line = model.translate_fields(line.strip(), args.lang_center)[0].strip()
            fields = trans_line.strip().split(args.fields_dim)
            # trans_words, label = fields[0], fields[1]
            try: # 211214
              trans_words, label = fields[0], fields[1]
            except IndexError:
              logger.warning(f" [trans_to_center] len(fields) = {fields}, fields = {fields}, trans_line = [{trans_line}]")
              ori_fields = line.strip().split(args.fields_dim) # NOTE: keep the original text
              trans_words, label = ori_fields[0], ori_fields[1]
            if lowercase: trans_words = trans_words.lower()
            # fout.write(args.fields_dim.join([trans_line, str(i)]) + '\n')
            fout.write(args.fields_dim.join([trans_words, label, str(i)]) + '\n')
            lowercase = True
          else:
            fout.write('\n')
            lowercase = False
        else:
          raise NotImplementedError(f" [trans_to_center] unknown task_type = {args.task_type}")
        if i % 500 == 0: logger.info(f" [trans_to_center]     {i} lines")
      logger.info(f" [trans_to_center]     {i} lines")
    if args.task_type == "seq_tag": # 211212
      copyfile(in_path + IDX_EXT, out_path + IDX_EXT)
    # for fid in args.fields:
    #   ori_path = os.path.join(split_dir, args.lang_center)
    #   logger.info(f" [trans_to_center]     bleu = {compute_corpus_bleu(out_path, ori_path, int(fid))}, hyps = [{out_path}], refs = [{ori_path}]")


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
  if args.test_bleu:
    for fid in args.fields:
      logger.info(f" [main]     bleu = {compute_corpus_bleu(args.out_dir, args.in_dir, int(fid))}")
    exit()

  model = M2M100(args.model_dir)
  model.set_fields(args.fields.split(','), args.fields_dim)

  if args.test_model:
    src_path, tgt_path = args.in_dir, args.out_dir
    src, tgt = os.path.splitext(src_path)[-1].strip('.'), os.path.splitext(tgt_path)[-1].strip('.')
    src_path_out, tgt_path_out = src_path + ".out", tgt_path + ".out"
    '''
    with open(src_path, 'r') as fin, open(src_path_out, 'w') as fout:
      model.set_src_lang(src)
      for line in fin:
        fout.write(model.translate_fields(line.strip(), tgt) + '\n')
    with open(tgt_path, 'r') as fin, open(tgt_path_out, 'w') as fout:
      model.set_src_lang(tgt)
      for line in fin:
        fout.write(model.translate_fields(line.strip(), src) + '\n')
    '''
    for fid in args.fields:
      logger.info(f" [main]     bleu = {compute_corpus_bleu(src_path_out, tgt_path, int(fid), char=True)}, hyps = [{src_path_out}], refs = [{tgt_path}]")
      logger.info(f" [main]     bleu = {compute_corpus_bleu(tgt_path_out, src_path, int(fid), char=True)}, hyps = [{tgt_path_out}], refs = [{src_path}]")
    exit()

  os.makedirs(args.out_dir, exist_ok=True)

  for p_dir, splits, _ in os.walk(args.in_dir):
    logger.info(f" [main] {p_dir}")
    for split in splits:
      if split in args.skip_splits.split(','):
        logger.info(f" [main] skip {split}, from {args.skip_splits}")
        continue
      logger.info(f" [main]   {split}")
      split_dir = os.path.join(p_dir, split)
      langs = os.listdir(split_dir)
      for lang in args.skip_langs.split(','):
        if lang in langs: langs.remove(lang)
      out_split_dir = os.path.join(args.out_dir, "trans-" + split)
      os.makedirs(out_split_dir, exist_ok=True)
      if split == "train": # trans-train
        trans_from_center(model, split_dir, langs, out_split_dir)
      else: # trans-test / trans-dev
        trans_to_center(model, split_dir, langs, out_split_dir)
    break


if __name__ == "__main__":
  main()
