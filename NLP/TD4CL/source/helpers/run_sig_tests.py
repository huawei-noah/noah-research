#!/usr/bin python

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

import subprocess
import sys

main = "/amc/english_only"

trainset = sys.argv[1]
dataset = sys.argv[2]
model = sys.argv[3]
LR = sys.argv[4]
BS = sys.argv[5]
E = sys.argv[6]
LEN = sys.argv[7]
file_type = sys.argv[8]


seed_steps = {
	   'siqa': {'123': {'xlm-roberta-base': 11676, 'roberta-base': 12093},
				'456': {'xlm-roberta-base': 15429, 'roberta-base': 21267},
				'789': {'xlm-roberta-base': 22518, 'roberta-base': 21684}},

	   'pawsx': {'123': {'xlm-roberta-base': 8778, 'roberta-base': 4774},
				'456': {'xlm-roberta-base': 12166, 'roberta-base': 5852},
				'789': {'xlm-roberta-base': 5698, 'roberta-base': 4004}},

	   'xnli': {'123': {'xlm-roberta-base': 71500, 'roberta-base': 49088},
				'456': {'xlm-roberta-base': 107000, 'roberta-base': 40000},
				'789': {'xlm-roberta-base': 71500, 'roberta-base': 22500}},

	   'qnli': {'123': {'roberta-base': 14306},
				'456': {'roberta-base': 13373},
				'789': {'roberta-base': 14928}},

	   'rte': {'123': {'roberta-base': 560},
			   '456': {'roberta-base': 462},
			   '789': {'roberta-base': 592}},

	   'mldoc': {'123': {'xlm-roberta-base': "1333"},
		  		 '456': {'xlm-roberta-base': "1178"},
				 '789': {'xlm-roberta-base': "1519"}}
}


# vs Random
for curric in ['annealing', 'competence-0.01-0.9-', 'annealing-bias', 'competence-bias-0.01-0.9-']:
	print(f">>>>> {curric.upper()} vs Random <<<<<")

	if curric.startswith('competence-'):
		curric1 = curric + str(seed_steps[trainset]['123'][model])
		curric2 = curric + str(seed_steps[trainset]['456'][model])
		curric3 = curric + str(seed_steps[trainset]['789'][model])
		if trainset == 'rte' or trainset == 'qnli':
			metric = '-confidence'
		else:
			metric = ''
	else:
		curric1 = curric
		curric2 = curric
		curric3 = curric
		if trainset == 'rte' or trainset == 'qnli':
			metric = '-correctness'
		else:
			metric = ''

	llist = [
		'python', 'ar_sigtest.py',
		'--dataA',
		f'{main}/{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}_{curric1}_{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}{metric}/{dataset}/',
		f'{main}/{trainset}_{model}_456_LR{LR}_LEN{LEN}_BS{BS}_E{E}_{curric2}_{trainset}_{model}_456_LR{LR}_LEN{LEN}_BS{BS}_E{E}{metric}/{dataset}/',
		f'{main}/{trainset}_{model}_789_LR{LR}_LEN{LEN}_BS{BS}_E{E}_{curric3}_{trainset}_{model}_789_LR{LR}_LEN{LEN}_BS{BS}_E{E}{metric}/{dataset}/',
		'--dataB',
		f'{main}/{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}/{dataset}/',
		f'{main}/{trainset}_{model}_456_LR{LR}_LEN{LEN}_BS{BS}_E{E}/{dataset}/',
		f'{main}/{trainset}_{model}_789_LR{LR}_LEN{LEN}_BS{BS}_E{E}/{dataset}/',
		'--file_type', f'{file_type}'
	]
	p = subprocess.call(llist)
	print()


# vs CR
for curric in ['annealing', 'competence-0.01-0.9-', 'annealing-bias', 'competence-bias-0.01-0.9-']:
	print(f">>>>> {curric.upper()} vs Cross-Review <<<<<")

	if curric.startswith('competence-'):
		curric1 = curric + str(seed_steps[trainset]['123'][model])
		curric2 = curric + str(seed_steps[trainset]['456'][model])
		curric3 = curric + str(seed_steps[trainset]['789'][model])
		if trainset == 'rte' or trainset == 'qnli':
			metric = '-confidence'
		else:
			metric = ''
	else:
		curric1 = curric
		curric2 = curric
		curric3 = curric
		if trainset == 'rte' or trainset == 'qnli':
			metric = '-correctness'
		else:
			metric = ''

	llist = [
		'python', 'ar_sigtest.py',
		'--dataA',
		f'{main}/{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}_{curric1}_{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}{metric}/{dataset}/',
		f'{main}/{trainset}_{model}_456_LR{LR}_LEN{LEN}_BS{BS}_E{E}_{curric2}_{trainset}_{model}_456_LR{LR}_LEN{LEN}_BS{BS}_E{E}{metric}/{dataset}/',
		f'{main}/{trainset}_{model}_789_LR{LR}_LEN{LEN}_BS{BS}_E{E}_{curric3}_{trainset}_{model}_789_LR{LR}_LEN{LEN}_BS{BS}_E{E}{metric}/{dataset}/',
		'--dataB',
		f'{main}/{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}_annealing_{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}_cross-review{metric}/{dataset}/',
		f'{main}/{trainset}_{model}_456_LR{LR}_LEN{LEN}_BS{BS}_E{E}_annealing_{trainset}_{model}_456_LR{LR}_LEN{LEN}_BS{BS}_E{E}_cross-review{metric}/{dataset}/',
		f'{main}/{trainset}_{model}_789_LR{LR}_LEN{LEN}_BS{BS}_E{E}_annealing_{trainset}_{model}_789_LR{LR}_LEN{LEN}_BS{BS}_E{E}_cross-review{metric}/{dataset}/',
		'--file_type', f'{file_type}'
	]
	p = subprocess.call(llist)
	print()
