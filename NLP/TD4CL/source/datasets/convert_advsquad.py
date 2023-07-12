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
import sys
import spacy
import en_core_web_sm
from tqdm import tqdm
import pandas as pd


nlp = en_core_web_sm.load()
data = datasets.load_from_disk('../../original_data/squad_adversarial')
print(data)

instances = []
found = 0

for d in tqdm(data):
    answers = d['answers']['text']
    answers_start = d['answers']['answer_start']
    # print(answers)
    # print(answers_start)
    cntx = nlp(d['context'])

    c = 0
    for i, sentence in enumerate(cntx.sents):
        new_instance = {'sentence1': d['question'], 'sentence2': str(sentence)}
        for answer, answer_start in zip(answers, answers_start):
            if c <= answer_start < (c + len(str(sentence))):
                new_instance['gold_label'] = 'entailment'
                # print(answers)
                # print(c, c + len(str(sentence)), answers_start)
                # print(sentence)
                # print('-----')
                found += 1
                break
        if 'gold_label' not in new_instance:
            new_instance['gold_label'] = 'not_entailment'

        c += len(str(sentence))

        if new_instance not in instances:
            instances.append(new_instance)
        # print(instances[-1])

new_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=instances))
print(new_dataset)
new_dataset.save_to_disk('../../data/squad_adversarial_converted')
