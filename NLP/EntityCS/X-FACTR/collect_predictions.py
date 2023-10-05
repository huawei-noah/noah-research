#!/usr/bin/bash python
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

from subprocess import check_output
import sys
import numpy as np

LANGUAGES=["en", "fr", "nl", "es", "ru", "zh", "he", "tr", "ko", "vi", "el", "mr", "ja", "hu", "bn", "ceb", "war", "tl", "sw", "pa", "mg", "ilo"]


langs = {}
for l in LANGUAGES:
    print(l)

    langs[l] = {}

    if sys.argv[3] == 'ind':
        out = check_output(['python', 'scripts/ana.py', '--model', sys.argv[1], '--lang', l, '--inp', f'{sys.argv[2]}/{l}_ind'])
    else:
        out = check_output(['python', 'scripts/ana.py', '--model', sys.argv[1], '--lang', l, '--inp', f'{sys.argv[2]}/{l}_conf'])

    output = out.decode('utf-8').split('\n')
    overall_acc = output[2].split('acc')[1].lstrip()
    single_acc = output[3].split('single')[1].lstrip()
    multi_acc = output[4].split('multi')[1].lstrip()

    langs[l] = {'all': float(overall_acc), 'single': float(single_acc), 'multi': float(multi_acc)}


# Printing
print('*' * 10)
print(sys.argv[3].upper())
print('*' * 10)

print('\t'.join(['      '] + ['AVG'] + LANGUAGES))

all_ = [langs[l]['all'] for l in LANGUAGES]
single_ = [langs[l]['single'] for l in LANGUAGES]
multi_ = [langs[l]['multi'] for l in LANGUAGES]

print('\t'.join(['all   '] + list(map(str, [np.round(np.mean(all_), 2)] + all_))))
print('\t'.join(['single'] + list(map(str, [np.round(np.mean(single_), 2)] + single_))))
print('\t'.join(['multi '] + list(map(str, [np.round(np.mean(multi_), 2)] + multi_))))


