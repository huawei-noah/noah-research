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
import os
import logging
import pickle
import torch
from torch.utils.data import TensorDataset
from xlm_ra import get_intent_labels, get_slot_labels
from utils import Tasks

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        class_label: (Optional) string. The label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, class_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.class_label = class_label
        self.slot_labels = slot_labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, class_label, slot_labels, guid):
        self.input_ids = input_ids
        self.class_label = class_label
        self.slot_labels = slot_labels
        self.guid = guid


class XNLUProcessor(object):
    """ XNLUProcessor for all data sets """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

    def _create_examples(self, data):
        examples = []
        inputs = data['inputs']
        intents = data['intent_labels']
        all_slots = data['slot_labels']
        unique_ids = data['all_ids']
        for inp, intent, slots, u_id in zip(inputs, intents, all_slots, unique_ids):
            intent_label = self.intent_labels.index(intent)
            s_labels = []
            for s in slots:
                if s == "PAD":
                    s_labels.append(self.args.ignore_index)
                else:
                    s_labels.append(self.slot_labels.index(s))
            assert len(inp) == len(slots)
            examples.append(InputExample(guid=u_id, words=inp, class_label=intent_label, slot_labels=s_labels))
        return examples

    def get_examples(self, mode, languages):
        """
        Args:
            mode: train, dev, test
            languages: which languages to select
        """
        examples = []
        for language in languages.split(","):
            data_path = os.path.join(self.args.data_dir, self.args.task, language, mode)
            logger.info("Reading file: {}".format(data_path))
            examples.extend(self._create_examples(pickle.load(open(os.path.join(data_path, "data.pkl"), "rb"))))
        return examples


def convert_examples_to_features(examples, args, tokenizer):
    # Settings based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id
    max_seq_len = args.max_seq_len
    pad_token_label_id = args.ignore_index

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = copy.deepcopy(example.words)
        slot_labels_ids = copy.deepcopy(example.slot_labels)

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            raise Exception("Increase max_seq_len, please!")

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        if ex_index < 1:
            logger.info("*** Example %d ***" % ex_index)
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("intent_label: %d" % example.class_label)
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(InputFeatures(input_ids=input_ids, class_label=example.class_label,
                                      slot_labels=slot_labels_ids, guid=example.guid))

    return features


def load_and_cache_examples(args, tokenizer, mode):
    processor = XNLUProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.model_dir, 'features_{}_{}.bin'.format(mode, args.task))
    cached_examples_file = os.path.join(args.model_dir, 'examples_{}_{}.bin'.format(mode, args.task))
    assert args.task in [t.value for t in Tasks]

    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        examples = torch.load(cached_examples_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train", args.train_languages)
        elif mode == "dev":
            examples = processor.get_examples("dev", args.dev_languages)
        elif mode == "test":
            examples = processor.get_examples("test", args.test_languages)
        else:
            raise Exception("For mode, only train, dev, test is available...")

        if args.task in [Tasks.MTOD.value, Tasks.MTOP.value, Tasks.M_ATIS.value]:
            features = convert_examples_to_features(examples=examples, args=args, tokenizer=tokenizer)
        else:
            raise Exception("Sorry, the task '%s' is not recognised." % args.task)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
        logger.info("Saving examples into cached file %s", cached_examples_file)
        torch.save(examples, cached_examples_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.class_label for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels for f in features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_label_ids, all_slot_labels_ids, all_guids)
    return dataset, examples
