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

import logging
import os
import numpy as np
import pickle
import torch
from torch.utils.data import TensorDataset
from data_loader import InputFeatures
from utils import load_tokenizer, Tasks
from xlm_ra import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


def generate_alignment_pairs(args):

    # TOP TIP: Experiments could be faster by caching the alignment data
    # cached_alignment_file = os.path.join(args.model_dir, 'alignment_{}.bin'.format(args.task))

    examples = []
    slot_labels = get_slot_labels(args)
    intent_labels = get_intent_labels(args)
    for language in args.align_languages.split(","):
        with open(os.path.join(args.data_dir, args.task, language, "train.tsv"), "r", encoding="utf-8") as tar_f:
            data = pickle.load(open(os.path.join(args.data_dir, args.task, "en", "train", "data.pkl"), "rb"))
            for target_line, label, slots in zip(tar_f, data['intent_labels'], data['slot_labels']):
                examples.append((target_line.strip(), label.strip(), slots))
        logger.info("Read %d lines...." % len(examples))

    tokenizer = load_tokenizer(args.model_name_or_path)
    pad_token_id = tokenizer.pad_token_id

    feats = []
    for ex_id, example in enumerate(examples):
        if ex_id % 5000 == 0:
            logger.info("Processed %d examples..." % ex_id)

        if args.task in [Tasks.MTOD.value, Tasks.MTOP.value, Tasks.M_ATIS.value]:
            tokens = tokenizer.tokenize(example[0])
            ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokens))
            assert tokens is not None and len(ids) <= args.max_seq_len
            ids = ids + ([pad_token_id] * (args.max_seq_len - len(ids)))
        else:
            raise Exception("The task '%s' is not recognised!" % args.task)

        if ids:
            slots_present = np.zeros((len(slot_labels),))
            for s in example[2]:
                if s == "PAD":
                    continue
                slots_present[slot_labels.index(s)] = 1
            feats.append(InputFeatures(ids, intent_labels.index(example[1]), slots_present, None))

    target_input_ids = torch.tensor([f.input_ids for f in feats], dtype=torch.long)
    slots_binary = torch.tensor([f.slot_labels for f in feats], dtype=torch.float32)
    labels = torch.tensor([f.class_label for f in feats], dtype=torch.long)
    train_dataset = TensorDataset(target_input_ids, labels, slots_binary)
    assert len(target_input_ids) == len(slots_binary)

    logger.info("Created %d train/align instances." % len(train_dataset))
    return train_dataset
