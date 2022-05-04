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

import copy
import csv
import json
import logging
from transformers import XLMTokenizer, XLMRobertaTokenizer

import torch

logger = logging.getLogger(__name__)


class InputExample(object):
  """
  A single training/test example for simple sequence classification.
  Args:
    guid: Unique id for the example.
    text_a: string. The untokenized text of the first sequence. For single
    sequence tasks, only this sequence must be specified.
    text_b: (Optional) string. The untokenized text of the second sequence.
    Only must be specified for sequence pair tasks.
    label: (Optional) string. The label of the example. This should be
    specified for train and dev examples, but not for test examples.
  """

  def __init__(self, guid, text_a, text_b=None, label=None, language=None):
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.language = language

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
  """
  A single set of features of data.
  Args:
    input_ids: Indices of input sequence tokens in the vocabulary.
    attention_mask: Mask to avoid performing attention on padding token indices.
      Mask values selected in ``[0, 1]``:
      Usually  ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded) tokens.
    token_type_ids: Segment token indices to indicate first and second portions of the inputs.
    label: Label corresponding to the input
  """

  def __init__(self, input_ids, attention_mask=None, token_type_ids=None, langs=None, label=None, input_ids_1=None):
    self.input_ids = input_ids
    # self.input_ids_0 = input_ids_0
    self.input_ids_1 = input_ids_1
    self.attention_mask = attention_mask
    self.token_type_ids = token_type_ids
    self.label = label
    self.langs = langs

  def __repr__(self):
    return str(self.to_json_string())

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
  examples,
  tokenizer,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
  sent_cls=False, # 211101
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    task: GLUE task
    label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
    output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  # is_tf_dataset = False
  # if is_tf_available() and isinstance(examples, tf.data.Dataset):
  #   is_tf_dataset = True

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))
    # if is_tf_dataset:
    #   example = processor.get_example_from_tensor_dict(example)
    #   example = processor.tfds_map(example)

    if sent_cls: # 211101
      inputs = tokenizer.encode_plus(example.text_a, add_special_tokens=True, max_length=max_length)
    else:
      if isinstance(tokenizer, XLMTokenizer):
        # inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, lang=example.language)
        # FIXME: `lang` seems unuseful, but raise warning "Keyword argument {} not recognized" from `transformers/tokenization_utils.py`
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length)
      else:
        inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length)
    
    if isinstance(tokenizer, XLMRobertaTokenizer): # 211018: psedo `token_type_ids`
      input_ids = inputs["input_ids"]
      token_type_ids = [pad_token_segment_id] * len(input_ids) # max_length
    else:
      input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token] * padding_length) + input_ids
      attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
      # if not isinstance(tokenizer, XLMRobertaTokenizer): # 211018
      token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
      input_ids = input_ids + ([pad_token] * padding_length)
      attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
      # if not isinstance(tokenizer, XLMRobertaTokenizer): # 211018
      token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

    if lang2id is not None:
      lid = lang2id.get(example.language, lang2id["en"])
    else:
      lid = 0
    langs = [lid] * max_length

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
      len(attention_mask), max_length
    )
    assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
      len(token_type_ids), max_length
    )

    if output_mode == "classification":
      label = label_map[example.label]
    elif output_mode == "regression":
      label = float(example.label)
    else:
      raise KeyError(output_mode)

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("label: %s (id = %d)" % (example.label, label))
      logger.info("language: %s, (lid = %d)" % (example.language, lid))

    features.append(
      InputFeatures(
        input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, langs=langs, label=label
      )
    )
  return features


def convert_examples_to_features_pair(
  examples,
  tokenizer,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
  sent_cls=False, # 211101
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    task: GLUE task
    label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
    output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  # is_tf_dataset = False
  # if is_tf_available() and isinstance(examples, tf.data.Dataset):
  #   is_tf_dataset = True

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))

    # inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length) # TODO:
    def truncate_sent(ids):
      if len(ids) > max_length:
        return ids[:max_length]
      else:
        return ids

    input_ids = truncate_sent(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.text_a))
        )
    # assert len(input_ids) <= max_length-2, "input_ids #### {} #### {}".format(len(input_ids), input_ids)
    if not sent_cls: # 211101
      input_ids_1 = truncate_sent(
          tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.text_b))
          )
      # assert len(input_ids_1) <= max_length-2, "input_ids_1 #### {} #### {}".format(len(input_ids_1), input_ids_1)
    
    # add special tokens: bos, eos
    # 210913: remove special tokens, FIXME: compatibale to Transformer
    # input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    # input_ids_1 = [tokenizer.cls_token_id] + input_ids_1 + [tokenizer.sep_token_id]
    
    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    input_ids = input_ids + ([pad_token] * padding_length)
    if not sent_cls: # 211101
      padding_length_1 = max_length - len(input_ids_1)
      input_ids_1 = input_ids_1 + ([pad_token] * padding_length_1)

    if lang2id is not None: # XLM
      lid = lang2id.get(example.language, lang2id["en"])
    else:
      lid = 0
    langs = [lid] * max_length

    assert len(input_ids) == max_length, "Error with input length {} vs {}: \n---> {} <---".format(len(input_ids), max_length, input_ids)
    if not sent_cls: # 211101
      assert len(input_ids_1) == max_length, "Error with input length {} vs {}: \n---> {} <---".format(len(input_ids_1), max_length, input_ids_1)

    if output_mode == "classification":
      label = label_map[example.label]
    elif output_mode == "regression":
      label = float(example.label)
    else:
      raise KeyError(output_mode)

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("sentence: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids)))
      if not sent_cls: # 211101
        logger.info("input_ids_1: %s" % " ".join([str(x) for x in input_ids_1]))
        logger.info("sentence_1: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids_1)))
      logger.info("label: %s (id = %d)" % (example.label, label))
      # logger.info("language: %s, (lid = %d)" % (example.language, lid))

    if sent_cls: # 211101
      input_ids_1 = None
    features.append(
      InputFeatures(
        input_ids=input_ids, input_ids_1=input_ids_1, langs=langs, label=label
      )
    )
  return features


def convert_examples_to_features_compat(
  examples,
  tokenizer_plm,
  tokenizer,
  max_length=512,
  label_list=None,
  output_mode=None,
  pad_on_left=False,
  pad_token_plm=0,
  pad_token=0,
  pad_token_segment_id=0,
  mask_padding_with_zero=True,
  lang2id=None,
):
  """
  Loads a data file into a list of ``InputFeatures``
  Args:
    examples: List of ``InputExamples`` or ``tf.data.Dataset`` containing the examples.
    tokenizer: Instance of a tokenizer that will tokenize the examples
    max_length: Maximum example length
    task: GLUE task
    label_list: List of labels. Can be obtained from the processor using the ``processor.get_labels()`` method
    output_mode: String indicating the output mode. Either ``regression`` or ``classification``
    pad_on_left: If set to ``True``, the examples will be padded on the left rather than on the right (default)
    pad_token: Padding token
    pad_token_segment_id: The segment ID for the padding token (It is usually 0, but can vary such as for XLNet where it is 4)
    mask_padding_with_zero: If set to ``True``, the attention mask will be filled by ``1`` for actual values
      and by ``0`` for padded values. If set to ``False``, inverts it (``1`` for padded values, ``0`` for
      actual values)
  Returns:
    If the ``examples`` input is a ``tf.data.Dataset``, will return a ``tf.data.Dataset``
    containing the task-specific features. If the input is a list of ``InputExamples``, will return
    a list of task-specific ``InputFeatures`` which can be fed to the model.
  """
  # is_tf_dataset = False
  # if is_tf_available() and isinstance(examples, tf.data.Dataset):
  #   is_tf_dataset = True

  label_map = {label: i for i, label in enumerate(label_list)}

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      logger.info("Writing example %d" % (ex_index))

    # inputs = tokenizer.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length) # TODO:
    if isinstance(tokenizer_plm, XLMTokenizer):
      inputs = tokenizer_plm.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, lang=example.language)
    else:
      inputs = tokenizer_plm.encode_plus(example.text_a, example.text_b, add_special_tokens=True, max_length=max_length)
    
    input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding_length = max_length - len(input_ids)
    if pad_on_left:
      input_ids = ([pad_token_plm] * padding_length) + input_ids
      attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
      token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
    else:
      input_ids = input_ids + ([pad_token_plm] * padding_length)

    def truncate_sent(ids):
      if len(ids) > max_length-2: # bos, eos
        return ids[:max_length-2]
      else:
        return ids

    input_ids_0 = truncate_sent(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.text_a))
        )
    # assert len(input_ids) <= max_length-2, "input_ids #### {} #### {}".format(len(input_ids), input_ids)
    input_ids_1 = truncate_sent(
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(example.text_b))
        )
    # assert len(input_ids_1) <= max_length-2, "input_ids_1 #### {} #### {}".format(len(input_ids_1), input_ids_1)
    
    # add special tokens: bos, eos
    # 210913: remove special tokens, FIXME: compatibale to Transformer
    # input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
    # input_ids_1 = [tokenizer.cls_token_id] + input_ids_1 + [tokenizer.sep_token_id]
    
    # Zero-pad up to the sequence length.
    padding_length_0 = max_length - len(input_ids_0)
    input_ids_0 = input_ids_0 + ([pad_token] * padding_length_0)
    padding_length_1 = max_length - len(input_ids_1)
    input_ids_1 = input_ids_1 + ([pad_token] * padding_length_1)

    if lang2id is not None: # XLM
      lid = lang2id.get(example.language, lang2id["en"])
    else:
      lid = 0
    langs = [lid] * max_length

    assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
    assert len(input_ids_0) == max_length, "Error with input length {} vs {}".format(len(input_ids_0), max_length)
    assert len(input_ids_1) == max_length, "Error with input length {} vs {}".format(len(input_ids_1), max_length)

    if output_mode == "classification":
      label = label_map[example.label]
    elif output_mode == "regression":
      label = float(example.label)
    else:
      raise KeyError(output_mode)

    if ex_index < 5:
      logger.info("*** Example ***")
      logger.info("guid: %s" % (example.guid))
      logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
      logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
      logger.info("input_ids_0: %s" % " ".join([str(x) for x in input_ids_0]))
      logger.info("input_ids_1: %s" % " ".join([str(x) for x in input_ids_1]))
      logger.info("sentence_pair: %s" % " ".join(tokenizer_plm.convert_ids_to_tokens(input_ids)))
      logger.info("sentence_0: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids_0)))
      logger.info("sentence_1: %s" % " ".join(tokenizer.convert_ids_to_tokens(input_ids_1)))
      logger.info("label: %s (id = %d)" % (example.label, label))
      logger.info("language: %s, (lid = %d)" % (example.language, lid))

    features.append(
      InputFeatures(
        input_ids=input_ids, 
        attention_mask=attention_mask, 
        token_type_ids=token_type_ids,
        input_ids_0=input_ids_0, 
        input_ids_1=input_ids_1, 
        langs=langs,
        label=label,
      )
    )
  return features


class PairDataset(torch.utils.data.Dataset):
  def __init__(self, datasets):
    self.datasets = datasets

  def __getitem__(self, i):
    return tuple(d[i] for d in self.datasets)

  def __len__(self):
    return min(len(d) for d in self.datasets)


def parse_single_batch(batch, model_type, sent_cls): # 211118
  SIMPLE_MODELS = ["cnn", "bilstm", "mlp"] # 210918
  # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]} 
  if model_type in SIMPLE_MODELS: # ["cnn", "bilstm", "mlp"]: # 210914
    if sent_cls:
      inputs = {
          "input_ids": batch[0], "input_ids_1": (None), "labels": batch[1],
          "attention_mask": (None), "token_type_ids": (None),
          }
    else:
      inputs = {
          "input_ids": batch[0], "input_ids_1": batch[1], "labels": batch[2],
          "attention_mask": (None), "token_type_ids": (None),
          }
  else:
    inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
  if model_type != "distilbert":
    inputs["token_type_ids"] = (
      batch[2] if model_type in ["bert"] else None
    )  # XLM and DistilBERT don't use segment_ids
  if model_type == "xlm":
    inputs["langs"] = batch[4] # 210914
  return inputs


