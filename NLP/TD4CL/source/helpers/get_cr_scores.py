#!/use/bin/python

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

import os, re, sys
import numpy as np
import argparse
import logging
from tqdm import tqdm
from glob import glob
import json


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def load_seq_class():
    difficulty = {}

    logger.info('Loading models ...')
    for i in range(0, args.shards):
        for j in range(0, args.shards):
            if i != j:
                filename = f'{args.input_dir}/{args.dataset_name}{i}_{args.model_name}_{args.seed}_' \
                           f'LR{args.lr}_LEN{args.len}_BS{args.bs}_E{args.eps}/{args.dataset_name}{j}/val.json'
                logger.info(filename)
                with open(filename, 'r') as infile:
                    for line in tqdm(infile):
                        line = json.loads(line)

                        gold = line['label']
                        pred = line['pred']
                        key = line['id']

                        if key not in difficulty:
                            difficulty[key] = {'correctness': 0, 'variability': 1.0}

                        if pred == gold:
                            difficulty[key]['correctness'] += 1  # correct classification

    return difficulty


def main(args):
    data_difficulty = load_seq_class()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    exp_type = f"{args.dataset_name}_{args.model_name}_{args.seed}_LR{args.lr}_LEN{args.len}_BS{args.bs}_E{args.eps}_cross-review"
    output_file = os.path.join(args.output_dir, f'{exp_type}.json')

    logger.info(f' Writing dynamics statistics to {output_file}')
    with open(output_file, 'w') as outfile:
        json.dump(data_difficulty, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='../../dynamics/')
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--lr', type=str)
    parser.add_argument('--len', type=str)
    parser.add_argument('--bs', type=str)
    parser.add_argument('--eps', type=str)
    parser.add_argument('--shards', type=int)
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--seed', type=str)
    parser.add_argument('--input_dir', type=str, default='../../trained_models_shards/')
    args = parser.parse_args()

    main(args)
