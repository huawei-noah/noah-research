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
import argparse

main_dir = "../trained_models"

parser = argparse.ArgumentParser()
parser.add_argument('--train_set', type=str)
parser.add_argument('--eval_set', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--lr', type=str)
parser.add_argument('--bs', type=str)
parser.add_argument('--eps', type=str)
parser.add_argument('--len', type=str)
args = parser.parse_args()

trainset = args.train_set
dataset = args.eval_set
model = args.model
LR = args.lr
BS = args.bs
E = args.eps
LEN = args.len


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

	   'rte': {'123': {'roberta-base': 658},
			   '456': {'roberta-base': 574},
			   '789': {'roberta-base': 728}},

	   'mldoc': {'123': {'xlm-roberta-base': 1333},
		  		 '456': {'xlm-roberta-base': 1178},
				 '789': {'xlm-roberta-base': 1519}}
}


# Random Training
print(f">>>>> RANDOM <<<<<")
llist = [
	'python', 'evaluate.py',
	'--model_name', f'{model}',
	'--model_dir', f'{main_dir}/{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}',
	f"--curriculum_steps", f"{seed_steps[trainset]['123'][model]}", f"{seed_steps[trainset]['456'][model]}", f"{seed_steps[trainset]['789'][model]}",
	'--dataset_name', f'{dataset}'
]
p = subprocess.call(llist)
print()


# Cross-Review
print(f">>>>> CR <<<<<")
metric = '-correctness'
llist = [
	'python', 'evaluate.py',
	'--model_name', f'{model}',
	'--model_dir', f'{main_dir}/{trainset}_{model}_123_LR{LR}_LEN128_BS{BS}_E{E}_annealing_{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}_cross-review{metric}',
	f"--curriculum_steps", f"{seed_steps[trainset]['123'][model]}", f"{seed_steps[trainset]['456'][model]}", f"{seed_steps[trainset]['789'][model]}",
	'--dataset_name', f'{dataset}'
]
p = subprocess.call(llist)
print()


# Task-dependent Metrics
for curric in ['annealing', 'competence-0.01-0.9-', 'annealing-bias', 'competence-bias-0.01-0.9-']:
	print(f">>>>> {curric.upper()} <<<<<")

	if curric.startswith('competence-'):
		curric1 = curric + str(seed_steps[trainset]['123'][model])
		curric2 = curric + str(seed_steps[trainset]['456'][model])
		curric3 = curric + str(seed_steps[trainset]['789'][model])
		metric = '-confidence'
	else:
		curric1 = curric
		curric2 = curric
		curric3 = curric
		metric = '-correctness'

	llist = [
		'python', 'evaluate.py',
		'--model_name', f'{model}',
		'--model_dir', f'{main_dir}/{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}_{curric1}_{trainset}_{model}_123_LR{LR}_LEN{LEN}_BS{BS}_E{E}{metric}',
		f"--curriculum_steps", f"{seed_steps[trainset]['123'][model]}", f"{seed_steps[trainset]['456'][model]}", f"{seed_steps[trainset]['789'][model]}",
		'--dataset_name', f'{dataset}'
	]
	p = subprocess.call(llist)
	print()


# Task-agnostic Metrics
for metric in ['heuristics-length', 'heuristics-rarity', f'{model}_ppl-ppl']:
	curric = 'competence-0.01-0.9-'
	print(f">>>>> {curric.upper()} <<<<<")

	curric1 = curric + str(seed_steps[trainset]['123'][model])
	curric2 = curric + str(seed_steps[trainset]['456'][model])
	curric3 = curric + str(seed_steps[trainset]['789'][model])

	llist = [
		'python', 'evaluate.py',
		'--model_name', f'{model}',
		'--model_dir', f'{main_dir}/{trainset}_{model}_123_LR{LR}_LEN128_BS{BS}_E{E}_{curric1}_{trainset}_{metric}',
		f"--curriculum_steps", f"{seed_steps[trainset]['123'][model]}", f"{seed_steps[trainset]['456'][model]}", f"{seed_steps[trainset]['789'][model]}",
		'--dataset_name', f'{dataset}'
	]
	p = subprocess.call(llist)
	print()
