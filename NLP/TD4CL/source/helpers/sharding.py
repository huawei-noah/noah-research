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

import datasets
import sys
import os
from collections import Counter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset_dir', type=str)
parser.add_argument('--output_dataset_dir', type=str)
parser.add_argument('--shards_num', type=int, default=10)
args = parser.parse_args()

dataset = datasets.load_from_disk(args.input_dataset_dir)
dataset = dataset['train']
dataset = dataset.shuffle(seed=42)
print(dataset)


for i in range(0, args.shards_num):
    dataset_shard = dataset.shard(num_shards=args.shards_num, index=i)

    out_dir = args.output_dataset_dir.strip('/')+str(i)
    print(f'Saving data in {out_dir}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(dataset_shard)
    if 'siqa' in args.input_dataset_dir:
        print(Counter(dataset_shard["label"]))
    else:
        print(Counter(dataset_shard["gold_label"]))
    dataset_shard.save_to_disk(out_dir)
