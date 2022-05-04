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

import tokenizers as tks
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Split
from transformers import BertTokenizer

from subword_nmt.learn_bpe import learn_bpe, get_vocabulary
from subword_nmt import apply_bpe

from jieba import Tokenizer as zh_Tokenizer
from pythainlp.tokenize import Tokenizer as th_Tokenizer
from MeCab import Tagger as ja_Tokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


PRE_TOKENIZERS = {
    "default": Whitespace(), # most languages using spaces to split words
    "vi": Whitespace(), # Vietnamese, spaces for syllables
    "zh": Whitespace(), # Chinese
    # "zh": zh_tokenizer.tokenize, # Chinese
    "ja": Split("", "removed"), # Japanese
    "th": Whitespace(), # Thai
    # "th": th_tokenizer.word_tokenize, # Thai
    "lo": Split("", "removed"), # Lao
    "my": Split("", "removed"), # Burmese
    "dz": Split("", "removed"), # Dzongkha, no spaces, other marks syllables
    "bo": Split("", "removed"), # Tibetan, no spaces, other marks syllables
    }


def sent_iterator_from_xnli_files(paths, tokenize=None, lang=None):
  for path in paths:
    if os.path.isfile(path):
      logger.info(f" reading from: {path}")
      with open(path, 'r') as inf:
        for i, line in enumerate(inf):
          if i > 0:
            for sent in line.strip().split('\t')[:2]:
              if tokenize:
                if lang == "zh":
                  yield ' '.join([w for w, _, _ in tokenize(sent)])
                else:
                  yield ' '.join(tokenize(sent))
              else:
                yield sent
    else:
      logger.warning(f" file not found: {path}")


def sent_iterator_from_wiki_files(paths, tokenize=None, lang=None):
  for path in paths:
    if os.path.isfile(path):
      logger.info(f" reading from: {path}")
      with open(path, 'r') as inf:
        for i, line in enumerate(inf):
          sent = line.strip()
          if tokenize:
            if lang == "zh":
              yield ' '.join([w for w, _, _ in tokenize(sent)])
            else:
              yield ' '.join(tokenize(sent))
          else:
            yield sent
    else:
      logger.warning(f" file not found: {path}")


def sent_iterator_from_scls_files(paths, tokenize=None, lang=None):
  for path in paths:
    if os.path.isfile(path):
      logger.info(f" reading from: {path}")
      with open(path, 'r') as inf:
        for i, line in enumerate(inf):
          if i > 0:
            sent = line.split('\t')[0].strip()
            if tokenize:
              if lang == "zh":
                yield ' '.join([w for w, _, _ in tokenize(sent)])
              else:
                yield ' '.join(tokenize(sent))
            else:
              yield sent
    else:
      logger.warning(f" file not found: {path}")


SENT_ITERATOR_FROM_FILES = {
    "xnli": sent_iterator_from_xnli_files,
    "wiki": sent_iterator_from_wiki_files,
    "scls": sent_iterator_from_scls_files,
    "matis": sent_iterator_from_scls_files,
    }


def build_vocab(lang, file_type, path_pattern, out_dir, vocab_size, trainer, spec_file=False, bpe_package="subword-nmt"):
  if file_type == "matis":
    pre_tokenize_func = None
  else:
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
  logger.info(f" processing {lang}")

  if spec_file:
    files = [path_pattern]
  else:
    if file_type == "xnli":
      files = [path_pattern.format(ftype, lang) for ftype in ["dev", "test", "train", "trans-train", "trans-test"]]
    elif file_type == "wiki":
      files = [path_pattern.format(lang)]
    elif file_type == "scls" or file_type == "matis": # 211101: single sent cls
      files = [path_pattern.format(ftype, lang) for ftype in ["dev", "train", "trans-train"]]
    else:
      raise NotImplementedError(f"Unknown file_type: {file_type}")

  if bpe_package == "huggingface":
    bpe_huggingface(lang, files, file_type, out_dir, pre_tokenize_func, vocab_size, trainer)
  elif bpe_package == "subword-nmt":
    bpe_subword_nmt(lang, files, file_type, out_dir, pre_tokenize_func, vocab_size)
  else:
    raise NotImplementedError(f" [build_vocab] bpe_package = {bpe_package}")


def bpe_subword_nmt(lang, files, file_type, out_dir, pre_tokenize_func, vocab_size):
  assert file_type == "wiki", f" [bpe_subword_nmt] only supports `wiki` but {file_type}"
  
  special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
  bpe_sep = "@@"
  bert_sep = "##"

  for raw_file in files:
    if pre_tokenize_func:
      in_file = raw_file + ".words"
      with open(raw_file, 'r') as fin, open(in_file, 'w') as fout:
        for line in fin:
          if lang == "zh":
            words = ' '.join([w for w, _, _ in pre_tokenize_func(line.strip())])
          elif lang == "ja":
            words = pre_tokenize_func(line.strip()).strip()
          else:
            words = ' '.join(pre_tokenize_func(line.strip()))
          fout.write(words)
          fout.write('\n')
    else:
      in_file = raw_file

    work_dir = os.path.dirname(in_file)
    codes_file = in_file + ".codes"
    vocab_file = os.path.join(out_dir, f"vocab.{lang}.{vocab_size}.txt")
    tmp_file = in_file + ".tmp"

    with open(in_file, 'r') as fin, open(codes_file, 'w') as fout:
      learn_bpe(fin, fout, vocab_size, 2, False, is_dict=False, total_symbols=False)

    with open(codes_file, 'r') as fin:
      bpe = apply_bpe.BPE(fin, separator=bpe_sep)

    with open(in_file, 'r') as fin, open(tmp_file, 'w') as ftmp:
      for line in fin:
        bpe_line = bpe.segment(line).strip()
        bert_line = bpe_line.replace(bpe_sep + ' ', ' ' + bert_sep) # replace `@@ ` with ` ##`
        ftmp.write(bert_line)
        ftmp.write('\n')

    with open(tmp_file, 'r') as fin:
      vocab = get_vocabulary(fin)
    os.remove(tmp_file)

    with open(vocab_file, 'w') as fout:
      for i, (key, freq) in enumerate([(token, 1) for token in special_tokens] + sorted(vocab.items(), key=lambda x: x[1], reverse=True)):
        if i >= vocab_size: break
        fout.write("{0} {1}\n".format(key, freq))


def bpe_huggingface(lang, files, file_type, out_dir, pre_tokenize_func, vocab_size, trainer):
  tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
  tokenizer.pre_tokenizer = PRE_TOKENIZERS.get(lang, Whitespace())

  logger.info(f"  -- files: {files}")
  sent_iter = SENT_ITERATOR_FROM_FILES[file_type](files, pre_tokenize_func, lang)
  tokenizer.train_from_iterator(sent_iter, trainer)
  logger.info(f"  -- in-memory testing:")
  test_sent_iter = SENT_ITERATOR_FROM_FILES[file_type](files[:1], pre_tokenize_func, lang)
  for i, sent in enumerate(test_sent_iter):
    if i >= 5: break
    tokens = tokenizer.encode(sent).tokens
    logger.info(f"    -- sent:   {sent}")
    logger.info(f"    -- tokens: {tokens}")

  json_path = os.path.join(out_dir, "tokenizer.{}.{}.json".format(lang, vocab_size))
  tokenizer.save(json_path)
  logger.info(f"  -- save json to: {json_path}")

  logger.info(f"  -- .json testing:")
  tokenizer_json = Tokenizer.from_file(json_path)
  test_sent_iter = SENT_ITERATOR_FROM_FILES[file_type](files[:1], pre_tokenize_func, lang)
  for i, sent in enumerate(test_sent_iter):
    if i >= 5: break
    tokens = tokenizer_json.encode(sent).tokens
    logger.info(f"    -- sent:   {sent}")
    logger.info(f"    -- tokens: {tokens}")

  vocab_dict = tokenizer.get_vocab()
  vocab_list = [(token, sn) for token, sn in vocab_dict.items()]
  vocab_list.sort(key=lambda x:x[1]) # , reverse=True)
  actual_vocab_size = tokenizer.get_vocab_size(with_added_tokens=True)
  if actual_vocab_size > vocab_size:
    logger.info(f"  -- actual vocab size = {actual_vocab_size}")
  txt_path = os.path.join(out_dir, "vocab.{}.{}.txt".format(lang, vocab_size))
  with open(txt_path, 'w') as outf:
    for i, (token, sn) in enumerate(vocab_list):
      # if i >= vocab_size: break
      outf.write(f"{token}\n")
  logger.info(f"  -- save vocab to: {txt_path}")

  '''
  logger.info(f"  -- .txt testing:")
  tokenizer_txt = BertTokenizer(txt_path, do_lower_case=False)
  test_sent_iter = SENT_ITERATOR_FROM_FILES[file_type](files[:1], pre_tokenize_func, lang)
  for i, sent in enumerate(test_sent_iter):
    if i >= 5: break
    tokens = tokenizer_txt.encode(sent)
    logger.info(f"    -- sent:   {sent}")
    logger.info(f"    -- tokens: {tokens}")
  '''


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--vocab_size", type=int, default=10000, help="")
  parser.add_argument("--langs", type=str, default="ar,bg,de,el,en,es,fr,hi,ru,sw,th,tr,ur,vi,zh", help="")
  parser.add_argument("--path_pattern", type=str, default="/home/ma-user/work/cache/train_xnli_cloud/42/download/xnli/{}/{}", help="")
  parser.add_argument("--out_dir", type=str, default="./", help="")
  parser.add_argument("--file_type", type=str, default="xnli", help="xnli, wiki, scls, matis")
  args = parser.parse_args()

  trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=args.vocab_size)
  if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

  for lang in args.langs.split(','):
    build_vocab(lang, args.file_type, args.path_pattern, args.out_dir, args.vocab_size, trainer)


if __name__ == "__main__":
  main()
