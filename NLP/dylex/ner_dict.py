# coding=utf-8
# 2021/10/11 Changed for Dylex 
#   Huawei Technologies Co., Ltd. 

# Copyright 2018 The Google AI Language Team Authors, The HuggingFace Inc. team.
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


""" Named entity recognition fine-tuning: utilities to work with CLUENER task. """
import torch
import logging
import os
import copy
import json
import numpy as np

from sequence_labeling import get_dict_entities
from utils_ner import DataProcessor

logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, text_a, labels):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.labels = labels

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
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, input_len, segment_ids, label_ids, dict_ids, dict_supervise, dict_mask, loss_mask):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.input_len = input_len
        self.dict_ids = dict_ids
        self.dict_supervise = dict_supervise
        self.dict_mask = dict_mask
        self.loss_mask = loss_mask

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
    examples, ner_label_list, max_seq_length, tokenizer, trietree, dict_label_list, max_dict_num,
    cls_token_at_end=False, cls_token="[CLS]", cls_token_segment_id=0,
    sep_token="[SEP]", pad_on_left=False, pad_token=0, pad_token_segment_id=0,
    sequence_a_segment_id=0, mask_padding_with_zero=True, ):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    ner_label_map = {label: i for i, label in enumerate(ner_label_list)}
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))
        tokens = tokenizer.tokenize(example.text_a)
        label_ids = [ner_label_map[x] for x in example.labels]
        # Account for [CLS] and [SEP] with "- 2".
        special_tokens_count = 2
        if len(tokens) > max_seq_length - special_tokens_count:
            tokens = tokens[: (max_seq_length - special_tokens_count)]
            label_ids = label_ids[: (max_seq_length - special_tokens_count)]

        qrlt, _ = trietree.query(tokens)
        dict_ids, dict_supervise, dict_mask = process_query(
            qrlt, tokens, example.labels, dict_label_list, ner_label_list, max_num=max_dict_num)

        tokens += [sep_token]
        label_ids += [ner_label_map['O']]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens += [cls_token]
            label_ids += [ner_label_map['O']]
            segment_ids += [cls_token_segment_id]
        else:
            tokens = [cls_token] + tokens
            label_ids = [ner_label_map['O']] + label_ids
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [0] + [1 if mask_padding_with_zero else 0] * (len(input_ids)-1) ### 屏蔽slot预测中的cls梯度
        loss_mask = [1 if (mask_padding_with_zero and not token.startswith("##")) else 0 for token in tokens]
        input_len = len(label_ids)
        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
            label_ids = ([pad_token] * padding_length) + label_ids
        else:
            input_ids += [pad_token] * padding_length
            input_mask += [0 if mask_padding_with_zero else 1] * padding_length
            loss_mask += [0 if mask_padding_with_zero else 1] * padding_length
            segment_ids += [pad_token_segment_id] * padding_length
            label_ids += [pad_token] * padding_length
            for x in dict_ids:
                x += [pad_token] * padding_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(dict_ids[0]) == max_seq_length
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s", example.guid)
            logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
            logger.info(f"recover tokens: {' '.join([tokenizer.ids_to_tokens[i] for i in input_ids][:input_len])}")
            logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))
            logger.info(f"label_ids recover: {' '.join([ner_label_list[i] for i in label_ids][:input_len])}")

            for i,dict_id in enumerate(dict_ids):
                if np.array(dict_id).sum() > 1:
                    logger.info(f"dict_id-{i}  {' '.join([str(x) for x in dict_id])}")
                    logger.info(f"dict_rr-{i}  {' '.join([dict_label_list[i] for i in dict_id])}")
            logger.info("dict_supervise: %s", " ".join([str(x) for x in dict_supervise]))
            logger.info("dict_mask: %s", " ".join([str(x) for x in dict_mask]))
            logger.info("loss_mask: %s", " ".join([str(x) for x in loss_mask]))

        features.append(InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            input_len=input_len,
            segment_ids=segment_ids,
            label_ids=label_ids,
            dict_ids=dict_ids,
            dict_supervise=dict_supervise,
            dict_mask=dict_mask,
            loss_mask=loss_mask,
        ))
    return features


def is_match(instance, entities):
    for ent in entities:
        # if (instance.name, instance.start, instance.end) == ent:
        if (instance.start, instance.end) == (ent[1], ent[2]):
            return True
    return False
def process_query(qrlt, tokens, labels, dict_list, label_list, max_num=8, use_cls=False):
    """
    处理trietree树的处理结果
    qrlt: trietree查询的结果
    labels: 文本对应的标签
    tokens: 原始文本列表
    return:
        词典名列表 | 位置向量 | 监督向量
    """
    seq_len = len(tokens)
    names = []
    supervise = []
    dict_mask = []
    if use_cls:
        names.append([dict_list.index("CLS")] * (seq_len + 2))
        supervise.append(1)
        dict_mask.append(1)
    entitys = get_dict_entities(labels)
    qrlt = sorted(qrlt, key=lambda x: x.end - x.start, reverse=True)
    if qrlt:
        for instance in qrlt:
            if instance.end < len(tokens) - 1 and tokens[instance.end + 1].startswith("##"):
                continue
            tmp = ["[CLS]"] + ["O"] * instance.start + [f"B-{instance.name}"] + \
                  [f"I-{instance.name}"] * (instance.end - instance.start) + \
                  ["O"] * (seq_len - instance.end - 1) + ["O"]

            if is_match(instance, entitys):
                supervise.append(1)
            else:
                supervise.append(0)
            dict_mask.append(1)

            tmp = map(lambda x: dict_list.index(x), tmp)
            names.append(list(tmp))
    if len(names) < max_num:
        supervise.extend([0] * (max_num - len(names)))
        dict_mask.extend([0] * (max_num - len(names)))
        names.extend(
            [[dict_list.index("[CLS]")] + [dict_list.index("O")] * (seq_len + 1) for _ in range(max_num - len(names))])

    return names[:max_num], supervise[:max_num], dict_mask[:max_num]


class CnerProcessor(DataProcessor):
    """Processor for the chinese ner data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.char.bmes")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.char.bmes")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.char.bmes")), "test")

    def get_labels(self):
        """See base class."""
        return ["X", 'B-CONT', 'B-EDU', 'B-LOC', 'B-NAME', 'B-ORG', 'B-PRO', 'B-RACE', 'B-TITLE',
                'I-CONT', 'I-EDU', 'I-LOC', 'I-NAME', 'I-ORG', 'I-PRO', 'I-RACE', 'I-TITLE',
                'O', 'S-NAME', 'S-ORG', 'S-RACE', "[START]", "[END]"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line['words']
            # BIOS
            labels = []
            for x in line['labels']:
                if 'M-' in x:
                    labels.append(x.replace('M-', 'I-'))
                elif 'E-' in x:
                    labels.append(x.replace('E-', 'I-'))
                else:
                    labels.append(x)
            examples.append(InputExample(guid=guid, text_a=text_a, labels=labels))
        return examples

class CmoProcessor(CnerProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "train.txt"), delimiter="\t"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "dev.txt"), delimiter="\t"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test.txt"), delimiter="\t"), "test")
    def get_case_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_text(os.path.join(data_dir, "test_baseline_wrong_case_ground_truth.txt"), delimiter="\t"), "case")

    def get_labels(self, data_dir):
        label_path = os.path.join(data_dir, "label.txt")
        labels = file2list(label_path)
        if "O" in labels: labels.remove("O")
        labels = ["O"] + labels
        # 进行排序
        labels = sorted(labels)
        return labels


def file2list(filepath):
    lines = []
    with open(filepath, encoding="utf-8") as fr:
        for line in fr:
            line = line.strip()
            if line: lines.append(line)
    return lines