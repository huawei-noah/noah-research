#!/usr/bin/python
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
from datasets import load_dataset
import json
import pandas as pd


# XL-WiC
dataset = {}
for lang in ['bg', 'zh', 'hr', 'da', 'nl', 'et', 'fa', 'ja', 'ko', 'it', 'fr', 'de']:
    dataset[lang] = load_dataset('pasinit/xlwic', f'xlwic_en_{lang}')
    print(dataset[lang])


# load English WiC dataset now
val = []
test = []
with open("../downloaded_datasets/WiC/val.jsonl", 'r') as infile:
    for line in infile:
        new_line = json.loads(line)
        new_new_line = {
            'id': new_line['idx'],
            'context_1': new_line['sentence1'],
            'context_2': new_line['sentence2'],
            'target_word': new_line['word'],
            'language': 'en',
            'label': new_line['label'],
            'pos': 'X',
            'target_word_location_1': {'char_start': new_line['start1'], 'char_end': new_line['end1']},
            'target_word_location_2': {'char_start': new_line['start2'], 'char_end': new_line['end2']}
        }
        val.append(new_new_line)
    print('val', len(val))

with open("../downloaded_datasets/WiC/test.jsonl", 'r') as infile:
    for line in infile:
        new_line = json.loads(line)
        new_new_line = {
            'id': new_line['idx'],
            'context_1': new_line['sentence1'],
            'context_2': new_line['sentence2'],
            'target_word': new_line['word'],
            'language': 'en',
            'label': '',
            'pos': 'X',
            'target_word_location_1': {'char_start': new_line['start1'], 'char_end': new_line['end1']},
            'target_word_location_2': {'char_start': new_line['start2'], 'char_end': new_line['end2']}
        }
        test.append(new_new_line)
    print('test', len(test))


# Make EN data
dataset['en'] = datasets.DatasetDict({'train': dataset['bg']['train'],
                                      'validation': datasets.Dataset.from_pandas(pd.DataFrame(val)),
                                      'test': datasets.Dataset.from_pandas(pd.DataFrame(test))})
print(dataset['en'])


dataset = datasets.DatasetDict({
    f'{lang}': dataset[lang] for lang in ['en', 'bg', 'zh', 'hr', 'da', 'nl', 'et', 'fa', 'ja', 'ko', 'it', 'fr', 'de']
})
print(dataset)
dataset.save_to_disk('../data/xl_wic')
