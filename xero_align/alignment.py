# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import logging
import os
import pickle
import torch
from torch.utils.data import TensorDataset
from data_loader import InputFeatures
from utils import load_tokenizer, Tasks
# from xlm_ra import get_intent_labels

logger = logging.getLogger(__name__)


# noinspection PyCallingNonCallable
def generate_alignment_pairs(args):

    # Experiments could be faster by caching the alignment data
    # cached_features_file = os.path.join(args.model_dir, 'alignment_{}_{}.bin'.format(mode, args.task))

    examples = []
    # intent_labels = get_intent_labels(args) if args.task != Tasks.PAWS_X.value else None
    for language in args.align_languages.split(","):
        with open(os.path.join(args.data_dir, args.task, language, "train.tsv"), "r", encoding="utf-8") as tar_f, \
             open(os.path.join(args.data_dir, args.task, "en", "train.tsv"), "r", encoding="utf-8") as eng_f:
            data = pickle.load(open(os.path.join(args.data_dir, args.task, "en", "train", "data.pkl"), "rb"))
            for target_line, english_line, label in zip(tar_f, eng_f, data['intent_labels']):
                examples.append(((target_line.strip(), english_line.strip()), label.strip()))
        # ------------------- This is the experimental XLM-RA from the Discussion section ------------------------
        # with open(os.path.join(args.data_dir, args.task, language, "dev.tsv"), "r", encoding="utf-8") as tar_f, \
        #      open(os.path.join(args.data_dir, args.task, "en", "dev.tsv"), "r", encoding="utf-8") as eng_f:
        #     for target_line, english_line in zip(tar_f, eng_f):
        #         examples.append((target_line.strip(), english_line.strip()))
        # with open(os.path.join(args.data_dir, args.task, language, "test.tsv"), "r", encoding="utf-8") as tar_f, \
        #      open(os.path.join(args.data_dir, args.task, "en", "test.tsv"), "r", encoding="utf-8") as eng_f:
        #     for target_line, english_line in zip(tar_f, eng_f):
        #         examples.append((target_line.strip(), english_line.strip()))
        # --------------------------------------------------------------------------------------------------------
        logger.info("Read %d lines...." % len(examples))

    examples = examples[1:] if args.task == Tasks.PAWS_X.value else examples
    tokenizer = load_tokenizer(args.model_name_or_path)
    pad_token_id = tokenizer.pad_token_id

    feats = []
    for ex_id, example in enumerate(examples):
        if ex_id % 5000 == 0:
            logger.info("Processed %d examples..." % ex_id)
        input_ids = []

        if args.task == Tasks.PAWS_X.value:
            for utterance in example[0]:
                utterance = utterance.split("\t")[1:3]
                if len(utterance[0]) == 0 or len(utterance[1]) == 0:
                    continue
                tokens = [tokenizer.tokenize(u) for u in utterance]
                ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokens[0]),
                                                                 tokenizer.convert_tokens_to_ids(tokens[1]))
                assert tokens is not None and len(ids) <= args.max_seq_len
                ids = ids + ([pad_token_id] * (args.max_seq_len - len(ids)))
                input_ids.append(ids)
        elif args.task in [Tasks.MTOD.value, Tasks.MTOP.value, Tasks.M_ATIS.value]:
            for utterance in example[0]:
                tokens = tokenizer.tokenize(utterance)
                ids = tokenizer.build_inputs_with_special_tokens(tokenizer.convert_tokens_to_ids(tokens))
                assert tokens is not None and len(ids) <= args.max_seq_len
                ids = ids + ([pad_token_id] * (args.max_seq_len - len(ids)))
                input_ids.append(ids)
        else:
            raise Exception("The task '%s' is not recognised!" % args.task)

        if len(input_ids) == 2:
            # feats.append(InputFeatures(input_ids, intent_labels.index(example[1]), None, None))
            feats.append(InputFeatures(input_ids, None, None, None))

    # Convert to Tensors and build dataset
    target_input_ids = torch.tensor([f.input_ids[0] for f in feats], dtype=torch.long)
    english_input_ids = torch.tensor([f.input_ids[1] for f in feats], dtype=torch.long)
    # labels = torch.tensor([f.class_label for f in feats], dtype=torch.long)
    # train_dataset = TensorDataset(target_input_ids, english_input_ids, labels)
    # assert len(target_input_ids) == len(english_input_ids) == len(labels)
    train_dataset = TensorDataset(target_input_ids, english_input_ids)
    assert len(target_input_ids) == len(english_input_ids)

    logger.info("Created %d train/align instances." % len(train_dataset))
    return train_dataset
