# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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

import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)


def group_dicts(items):
    """
    Given a list of dictionaries -> list of dicts Group a list of dictionaries based on keys
    """
    new_items = {}
    for ak in items[0].keys():
        new_items[ak] = []

    for i in items:  # for dict in list
        for k, v in i.items():
            # special case for tokenized sentences
            if isinstance(v, list) and all(isinstance(v_, list) for v_ in v):
                new_items[k] += v
            elif isinstance(v, list) and 'tokenized' in k:
                new_items[k] += [v]
            elif isinstance(v, list):
                new_items[k] += v
            elif type(v) == np.ndarray and (k == 'labels_start' or k == 'labels_end'):
                new_items[k] += v.tolist()
            elif type(v) == np.ndarray:
                new_items[k] += np.vsplit(v, v.shape[0])
            else:
                new_items[k] += [v]
    return new_items


class MultipleChoiceProcessor:
    """
    Processor for converting multiple choice examples to
    appropriate format for Transformer Models.
    """
    def __init__(self, config, tokenizer):
        self.tokenizer = tokenizer
        self.config = config
        self.max_len = []
        self.question_dict = {
            "cause": "What was the cause?",
            "effect": "What was the effect?"
        }

    def convert_to_features(self, examples):
        nc = 0
        for key in examples:
            if key.startswith('choice'):
                nc += 1

        choices = [f"choice{i}" for i in range(1, nc+1)]

        first_sentences = [
            [f"{prem} {self.question_dict[quest]}" if quest in ['cause', 'effect'] else f"{prem} {quest}"
             for _ in choices]
            for i, (prem, quest) in enumerate(zip(examples['premise'], examples['question']))
        ]

        second_sentences = [
            [examples[choice][i] for choice in choices] for i, quest in enumerate(examples['question'])
        ]
        labels = examples['label']

        # Flatten out
        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        # Tokenize
        tokenized_examples = self.tokenizer(
            first_sentences,
            second_sentences,
            max_length=self.config['max_seq_length'],
            truncation=True,
            padding=False
        )
        # Un-flatten
        tokenized_inputs = {k: [v[i: i + nc] for i in range(0, len(v), nc)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels
        
        return tokenized_inputs

    def collate_fn(self, features):
        cols = ['input_ids', 'attention_mask']
        label_name = "labels"
        labels = [feature.pop(label_name) for feature in features]
        ids = [feature.pop('id') for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])

        flattened_features = [
            [{k: v[i] for k, v in feature.items() if k in cols} for i in range(num_choices)] for feature in features
        ]
        flattened_features = sum(flattened_features, [])

        batch = self.tokenizer.pad(
            flattened_features,
            padding=True,
            max_length=self.config['max_seq_length'],
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}

        # Add back labels
        batch["labels"] = torch.tensor(labels).long()
        batch["id"] = ids
        return batch


class SequenceClassificationProcessor:
    """
    Processor for converting data into model input
    Single or pair of sentences as input.
    """
    def __init__(self, config, tokenizer, label_map):
        self.label_map = label_map
        self.tokenizer = tokenizer
        self.config = config
        self.max_len = []

    def convert_to_features(self, examples):
        if ('gold_label' in examples) and (examples['gold_label'][0] is not None):
            examples["labels"] = [self.label_map[l] for l in examples["gold_label"]]

        if 'id' in examples:
            examples['id'] = [i for i in examples['id']]
        examples.update(examples)
        return examples

    def collate_fn(self, examples):
        ex_ = group_dicts(examples)

        if 'sentence2' in ex_:
            examples1, examples2 = ex_['sentence1'], ex_['sentence2']

            ex = self.tokenizer(examples1, examples2,
                                padding=True,
                                return_tensors='pt',
                                max_length=self.config['max_seq_length'],
                                truncation='longest_first')

        elif 'sentence1' in ex_:
            examples = ex_['sentence1']

            ex = self.tokenizer(examples,
                                padding=True,
                                return_tensors='pt',
                                max_length=self.config['max_seq_length'],
                                truncation='longest_first')

        else:
            raise Exception('Something is definitely wrong, check your config!')

        if 'labels' in ex_:
            ex['labels'] = torch.tensor(ex_['labels']).long()

        if 'id' in ex_:
            ex['id'] = ex_['id']

        return ex
