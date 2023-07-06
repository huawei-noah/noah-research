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

import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import json
import math
import os
import pandas as pd
import logging
import datasets
from transformers import AutoTokenizer
pd.options.mode.chained_assignment = None

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_seq_class(pred_folder, tokenizer, special_char, name):
    # for each possible model
    logger.info(pred_folder)
    logger.info('Tokenizing dataset ...')
    dataset = datasets.load_from_disk(pred_folder)['train']
    print(dataset)

    vocab = {}

    def tokenize_function(examples):
        if name == 'siqa':
            examples1 = examples['premise']
            examples2 = examples['question']
        else:
            examples1 = examples['sentence1']
            examples2 = examples['sentence2']

        res1 = tokenizer.tokenize(
            examples1,
            padding=False,
            truncation=False
        )
        res2 = tokenizer.tokenize(
            examples2,
            padding=False,
            truncation=False
        )

        new_dict = {"id": examples['id']}
        words = []
        for res_ in [res1, res2]:
            tmp = []
            for i, w in enumerate(res_):
                if i == 0:
                    tmp += [w]
                elif w.startswith(special_char):
                    words.append(tmp)
                    tmp = [w]
                else:
                    tmp += [w]
            words.append(tmp)

        new_dict['tokenized_input'] = []
        # Update word frequencies
        for w in words:
            new_word = tokenizer.convert_tokens_to_string(w).strip(' ').lower()
            new_dict['tokenized_input'] += [new_word]

        new_dict['length'] = len(new_dict['tokenized_input'])
        return new_dict

    dataset = dataset.map(tokenize_function, batched=False)

    # Calculate word_rarity
    for s in dataset:
        for w in s['tokenized_input']:
            if w in vocab:
                vocab[w] += 1
            else:
                vocab[w] = 1

    print('vocab_length', len(vocab))
    n_total = sum(vocab.values())

    def rarity_function(examples):
        sent = {}
        rarity = 0
        for word in examples['tokenized_input']:
            rarity += -math.log(vocab[word] / n_total)
        sent['rarity'] = rarity
        return sent

    dataset = dataset.map(rarity_function, batched=False)

    return dataset


def main(args):
    logger.info('*** Loading {} tokenizer ***'.format(args.model_name.upper()))
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name
    )

    # Load training statistics
    if args.dataset_name in ['pawsx', 'xnli', 'xcopa', 'mldoc', 'siqa', 'qnli', 'rte']:
        if args.model_name.startswith('roberta-'):
            sc = 'Ġ'
        else:
            sc = '▁'
        model_dynamics = load_seq_class(os.path.join(args.data_dir, args.dataset_name), tokenizer, special_char=sc,
                                        name=args.dataset_name)
    else:
        model_dynamics = None

    cols2remove = [cn for cn in model_dynamics.column_names if cn not in ['length', 'id', 'rarity']]
    model_dynamics = model_dynamics.remove_columns(cols2remove)
    print(model_dynamics)
    print(model_dynamics[0])

    final = {}
    for m in model_dynamics:
        final[m['id']] = m

    output_file = os.path.join(args.output_dir, f'{args.dataset_name}_heuristics.json')

    logger.info(f' Writing dynamics statistics to {output_file}')
    with open(output_file, 'w') as outfile:
        json.dump(final, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--output_dir', type=str, default='../../dynamics/')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--analysis', action='store_true')
    args = parser.parse_args()

    main(args)
