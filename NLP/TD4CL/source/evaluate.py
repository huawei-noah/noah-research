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

import os
from subprocess import check_output
import ast
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
parser.add_argument('--seed', nargs='*', help='Can be a list of seeds', default=['123', '456', '789'])
parser.add_argument('--model_dir', type=str, help='Model directory')
parser.add_argument('--dataset_name', type=str)
parser.add_argument('--data_dir', type=str, default="../data/")
parser.add_argument('--reference_steps', nargs='*')
parser.add_argument('--curriculum_steps', nargs='*')
args = parser.parse_args()

if args.model_dir.endswith('/'):
    args.model_dir = args.model_dir.rstrip('/')


data_dir = args.data_dir

script = 'main_seq_class.py' if args.dataset_name not in ['xcopa', 'xcopa-en', 'xcopa-tr-test', 'adv_squad',
                                                          'csqa', 'siqa'] else 'main_mchoice.py'

overall_avg_test = {}
overall_avg_val = {}
overall_avg_langs_val = {}
overall_avg_langs_test = {}
for m in ['accuracy']:
    overall_avg_val[m] = []
    overall_avg_test[m] = []
    overall_avg_langs_val[m] = []
    overall_avg_langs_test[m] = []


steps_per_seed = []
for s, seed in enumerate(args.seed):
    exp_name = args.model_dir.split('/')[-1]

    current_seed = exp_name.split('_')[2]

    exp_name = exp_name.replace(f'_{current_seed}_', f'_{seed}_')

    if 'competence' in exp_name:
        current_steps = exp_name.split('_')[7].split('-')[-1]
        exp_name = exp_name.replace(current_steps, str(args.curriculum_steps[s]))

    model_dir = os.path.join('/'.join(args.model_dir.split('/')[:-1]), exp_name)
    print(model_dir)

    """ Taking Steps """
    # take steps
    with open(os.path.join(model_dir, 'train.log'), 'r') as infile:
        for line in infile:
            if '*** Best Step =' in line.rstrip():
                steps = int(line.rstrip().split('*** Best Step = ')[1].split(' ***')[0])
                steps_per_seed.append(steps)

    """ Calling evaluation """
    eval_list = [
        'python',
        script,
        '--dataset_name', f'{args.dataset_name}',
        '--mode', 'eval',
        '--max_seq_length', '256' if args.dataset_name in ['mldoc'] else '128',
        '--log_interval', '100',
        '--device', '0',
        '--show_examples',
        '--gradient_accumulation_steps', '1',
        '--num_training_epochs', '1',
        '--lr', '6e-6',
        '--seed', f'{seed}',
        '--model_dir', model_dir,
        '--data_dir', data_dir,
        '--save_dir', '../trained_models/',
        '--model_name', f'{args.model_name}',
        '--curriculum', 'none'
    ]

    p = check_output(eval_list)
    dict_str = p.decode("UTF-8")
    final = ast.literal_eval(dict_str)

    total_results = {}
    for m in ['accuracy']:
        total_results[m] = {k: v[m] for k, v in final.items()}

    for m in ['accuracy']:  # for each dataset metric
        results_val, results_test = [], []
        val_keys, test_keys = [], []
        for k, v in total_results[m].items():  # languages
            if k.startswith('val') or k.startswith('tr-val'):
                val_keys += [k.split('-')[-1]]
                results_val += [float(v)]
            if k.startswith('test') or k.startswith('tr-test') or k in \
                    ['logic', 'knowledge', 'lexical_semantics', 'predicate_argument_structure', 'test']:
                test_keys += [k.split('-')[-1]]
                results_test += [float(v)]

        if results_val:
            overall_avg_langs_val[m].append(results_val)
            mean_all_langs_val = np.mean(results_val)

            print('\t'.join(['   '] + [' '*len(m)] + val_keys))
            res_per_lang_val = '\t'.join([f"{np.round(r, 2)}" for r in results_val])
            print(f'VAL\t{m}\t{res_per_lang_val}\t{mean_all_langs_val:.2f}\n')

            overall_avg_val[m] += [mean_all_langs_val]

        if results_test:
            overall_avg_langs_test[m].append(results_test)
            mean_all_langs_test = np.mean(results_test)

            print('\t'.join(['    '] + [' '*len(m)] + test_keys))
            res_per_lang_test = '\t'.join([f"{np.round(r, 2)}" for r in results_test])
            print(f'TEST\t{m}\t{res_per_lang_test}\t{mean_all_langs_test:.2f}\n')

            overall_avg_test[m] += [mean_all_langs_test]

print(50 * '-')

for m in ['accuracy']:
    print("Curriculum steps: "+"\t".join(f"{item}" for item in steps_per_seed))
    if args.reference_steps:
        print("Reference steps: "+"\t".join(f"{item}" for item in args.reference_steps))
        print()
        tmp = [int(a) / int(b) for a, b in zip(steps_per_seed, args.reference_steps)]
        print("Speedup: "+"\t".join(f"{t:.4f}" for t in tmp))

        print(f"avg {np.round(np.mean(tmp), 2)}")
    print()

    # Validation set
    if val_keys:
        print("\t".join(f"{item}" for item in overall_avg_val[m])+'\n')

        print('\t'.join(['         '] + [' ' * len(m)] + val_keys))

        res_per_lang = '\t'.join([f"{np.round(r, 2)}" for r in np.mean(np.array(overall_avg_langs_val[m]), axis=0)])
        res_per_lang_std = '\t'.join([f"{np.round(r, 2)}" for r in np.std(np.array(overall_avg_langs_val[m]), axis=0)])
        print(f'VAL mean \t{m}\t{res_per_lang}\t{np.mean(overall_avg_val[m]):.2f}')
        print(f'VAL std \t{m}\t{res_per_lang_std}\t{np.std(overall_avg_val[m]):.2f}\n')

    # Test set
    if test_keys:
        print("\t".join(f"{item}" for item in overall_avg_test[m])+'\n')

        print('\t'.join(['         '] + [' ' * len(m)] + test_keys))
        res_per_lang = '\t'.join([f"{np.round(r, 2)}" for r in np.mean(np.array(overall_avg_langs_test[m]), axis=0)])
        res_per_lang_std = '\t'.join([f"{np.round(r, 2)}" for r in np.std(np.array(overall_avg_langs_test[m]), axis=0)])
        print(f'TEST mean\t{m}\t{res_per_lang}\t{np.mean(overall_avg_test[m]):.2f}')
        print(f'TEST std\t{m}\t{res_per_lang_std}\t{np.std(overall_avg_test[m]):.2f}\n')
