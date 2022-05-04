# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors, 
# The HuggingFace Inc. team, and The XTREME Benchmark Authors.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from __future__ import absolute_import, division, print_function

import logging
import os
from io import open
from transformers import XLMTokenizer

logger = logging.getLogger(__name__)


class InputExample(object):
  """A single training/test example for token classification."""

  def __init__(self, guid, words, labels, langs=None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      words: list. The words of the sequence.
      labels: (Optional) list. The labels for each word of the sequence. This should be
      specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.words = words
    self.labels = labels
    self.langs = langs


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, input_ids, input_mask, segment_ids, label_ids, langs=None, dist_pos=None):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.langs = langs
    self.dist_pos = dist_pos


def read_examples_from_file(file_path, lang, lang2id=None):
  if not os.path.exists(file_path):
    logger.info("[Warming] file {} not exists".format(file_path))
    return []
  guid_index = 1
  examples = []
  subword_len_counter = 0
  if lang2id:
    lang_id = lang2id.get(lang, lang2id['en'])
  else:
    lang_id = 0
  logger.info("lang_id={}, lang={}, lang2id={}".format(lang_id, lang, lang2id))
  with open(file_path, encoding="utf-8") as f:
    words = []
    labels = []
    langs = []
    for line in f:
      if line.startswith("-DOCSTART-") or line == "" or line == "\n":
        if words: # TODO: `words`?
          examples.append(InputExample(guid="{}-{}".format(lang, guid_index),
                         words=words,
                         labels=labels,
                         langs=langs))
          guid_index += 1
          words = []
          labels = []
          langs = []
          subword_len_counter = 0
        else:
          print(f'guid_index', guid_index, words, langs, labels, subword_len_counter)
      else:
        splits = line.split("\t")
        word = splits[0]
      
        words.append(splits[0])
        langs.append(lang_id)
        if len(splits) > 1:
          labels.append(splits[1].replace("\n", "")) # 211212: -1 -> 1. 2: trans-orig index, 3: whether retokenize successfully
        else:
          # Examples could have no label for mode = "test"
          labels.append("O")
    if words:
      examples.append(InputExample(guid="%s-%d".format(lang, guid_index),
                     words=words,
                     labels=labels,
                     langs=langs))
  return examples


def convert_examples_to_features(examples,
                 label_list,
                 max_seq_length,
                 tokenizer,
                 cls_token_at_end=False,
                 cls_token="[CLS]",
                 cls_token_segment_id=1,
                 sep_token="[SEP]",
                 sep_token_extra=False,
                 pad_on_left=False,
                 pad_token=0,
                 pad_token_segment_id=0,
                 pad_token_label_id=-1,
                 sequence_a_segment_id=0,
                 mask_padding_with_zero=True,
                 lang='en',
                 sentpiece=False, # 211207
                 ):
  """ Loads a data file into a list of `InputBatch`s
    `cls_token_at_end` define the location of the CLS token:
      - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
      - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
    `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
  """

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d of %d", ex_index, len(examples))

    dist_pos = [] # 211209
    tokens = []
    label_ids = []
    # if ex_index < 5:
    #   logger.info(f"example.words = {example.words}")
    for word, label in zip(example.words, example.labels):
      dist_pos.append(len(tokens)) # 211209
      if isinstance(tokenizer, XLMTokenizer):
        word_tokens = tokenizer.tokenize(word, lang=lang)
      else:
        word_tokens = tokenizer.tokenize(word)
      try:
        if sentpiece and word_tokens[0] == '▁': # 211210: for langauges with few `▁`-head tokens, e.g. ja, zh
          word_tokens = word_tokens[1:]
      except IndexError: # 211212
        logger.warning(f" null word_tokens: word_tokens = [{word_tokens}], len(word_tokens) = {len(word_tokens)}, ex_index = {ex_index}, lang = {lang}, example = {example}")
        word_tokens = [tokenizer.unk_token] # TODO: roll-back `dist_pos` may be better?
      if len(word) != 0 and len(word_tokens) == 0:
        word_tokens = [tokenizer.unk_token]
      tokens.extend(word_tokens)
      # NOTE: Use the real label-id for the FIRST SUB-TOKEN of a word, and padding-ids for the remaining tokens
      label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
      # if ex_index < 5:
      #   logger.info(f"word = {word}")
      #   logger.info(f"word_tokens = {word_tokens}")
    # if ex_index < 5:
    #   logger.info(f"raw_tokens = {tokens}")

    # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
    special_tokens_count = 3 if sep_token_extra else 2 # TODO: for SIMPLE_MODELS
    if len(tokens) > max_seq_length - special_tokens_count:
      print('truncate token', len(tokens), max_seq_length, special_tokens_count)
      tokens = tokens[:(max_seq_length - special_tokens_count)]
      label_ids = label_ids[:(max_seq_length - special_tokens_count)]    

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids:   0   0   0   0  0     0   0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.

    tokens += [sep_token]
    label_ids += [pad_token_label_id]
    if sep_token_extra:
      # roberta uses an extra separator b/w pairs of sentences
      tokens += [sep_token]
      label_ids += [pad_token_label_id]
    segment_ids = [sequence_a_segment_id] * len(tokens)

    if cls_token_at_end:
      tokens += [cls_token]
      label_ids += [pad_token_label_id]
      segment_ids += [cls_token_segment_id]
    else:
      tokens = [cls_token] + tokens
      label_ids = [pad_token_label_id] + label_ids
      segment_ids = [cls_token_segment_id] + segment_ids
      dist_pos = [pos+1 for pos in dist_pos] # 211209: skip [CLS]'s index

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    pos_pad = 0
    dist_pos = dist_pos + [pos_pad] * max_seq_length
    dist_pos = dist_pos[:max_seq_length]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_seq_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids
      input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
      segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
      label_ids = ([pad_token_label_id] * padding_length) + label_ids
    else:
      input_ids += ([pad_token] * padding_length)
      input_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
      segment_ids += ([pad_token_segment_id] * padding_length)
      label_ids += ([pad_token_label_id] * padding_length)

    if example.langs and len(example.langs) > 0:
      langs = [example.langs[0]] * max_seq_length
    else:
      print('example.langs', example.langs, example.words, len(example.langs))
      print('ex_index', ex_index, len(examples))
      langs = None

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(langs) == max_seq_length
    assert len(dist_pos) == max_seq_length

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s", example.guid)
      logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
      logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
      logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
      logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
      logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
      logger.info("langs: {}".format(langs))
      logger.info("dist_pos: {}".format(dist_pos))

    features.append(
        InputFeatures(input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                langs=langs,
                dist_pos=dist_pos,
                ))
  return features


def get_labels(path):
  with open(path, "r") as f:
    labels = f.read().splitlines()
  if "O" not in labels:
    labels = ["O"] + labels
  return labels


def parse_single_batch_tag(batch, model_type): # 211202
  inputs = {"input_ids": batch[0],
      "attention_mask": batch[1],
      "labels": batch[3],
      }

  if model_type != "distilbert":
    # XLM and RoBERTa don"t use segment_ids
    inputs["token_type_ids"] = batch[2] if model_type in ["bert", "xlnet"] else None

  if model_type == "xlm":
    inputs["langs"] = batch[4]

  return inputs


def save_predictions(args, predictions, output_file, text_file, idx_file, output_word_prediction=False):
  '''file format:
    text_file:
      sent-0_token-0\tlabel-0\n
      ...
      sent-0_token-n\tlabel-n\n
      \n
      sent-k_token-0\tlabel-0\n
      ...
      sent-k_token-n\tlabel-n\n
    idx_file:
      0\n
      ...
      0\n
      \n
      k\n
      ...
      k\n
  '''
  # Save predictions
  with open(text_file, "r") as text_reader, open(idx_file, "r") as idx_reader:
    text = text_reader.readlines()
    index = idx_reader.readlines()
    assert len(text) == len(index)

  # Sanity check on the predictions
  with open(output_file, "w") as writer:
    example_id = 0
    prev_id = int(index[0])
    for line, idx in zip(text, index):
      if line == "" or line == "\n":
        example_id += 1
      else:
        cur_id = int(idx)
        output_line = '\n' if cur_id != prev_id else ''
        if output_word_prediction:
          output_line += line.split()[0] + '\t'
        output_line += predictions[example_id].pop(0) + '\n'
        writer.write(output_line)
        prev_id = cur_id

