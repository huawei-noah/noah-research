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
import pandas as pd
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging


logger = logging.getLogger(__name__)


sns.set(style='whitegrid', font_scale=1.0, font='Georgia')
sns.set_context('talk')
# sns.set_palette(sns.color_palette("crest"))
sns.set_palette(sns.cubehelix_palette(start=.5, rot=-.5, reverse=False))
# sns.set_palette(sns.color_palette("Set2", 5))
# sns.set_palette(sns.color_palette("husl", 5))
# sns.set_palette(sns.color_palette("cubehelix"))
fig = plt.figure(figsize=(7, 4.6), dpi=300)


mapp = {'random': 'Random', 'competence-bias': 'TD-Var_comp', 'competence': 'TD_comp', 'annealing': 'TD_anneal',
        'annealing-bias': 'TD-Var_anneal', 'cross-review': 'CR_anneal', 'length': "Length", 'rarity': 'Rarity',
        'ppl': 'PPL'}

colors = sns.cubehelix_palette(start=.5, rot=-.5, reverse=False)


def get_scores(args):
    scores = {}
    current_curric = args.model_dir.split('/')[-1].split('_')[7]
    current_seed = args.model_dir.split('/')[-1].split('_')[2]

    for curric in args.curricula:
        filenames = []
        if curric == 'random':
            for seed in args.seeds:
                current = '_'.join(args.model_dir.split('/')[-1].split('_')[7:])
                model_dir = args.model_dir.replace(current, '')
                filenames.append(model_dir.replace(current_seed, str(seed)).strip('_'))

        elif curric == 'cross-review':
            for seed in args.seeds:
                current = '_'.join(args.model_dir.split('/')[-1].split('_')[7:])
                first_three = '_'.join(args.model_dir.split('/')[-1].split('_')[0:7])
                model_dir = args.model_dir.replace(current, f'annealing_{first_three}_cross-review')
                filenames.append(model_dir.replace(current_seed, str(seed)).strip('_'))

        elif curric == 'competence' or curric == 'competence-bias':
            for j, seed in enumerate(args.seeds):
                current = '_'.join(args.model_dir.split('/')[-1].split('_')[7:])
                first = '_'.join(args.model_dir.split('/')[-1].split('_')[0:7])
                model_dir = args.model_dir.replace(current, f'{curric}-0.01-0.9-{args.steps[j]}_{first}')
                filenames.append(model_dir.replace(current_seed, str(seed)).strip('_'))

        elif curric == 'length' or curric == 'rarity':
            for j, seed in enumerate(args.seeds):
                current = '_'.join(args.model_dir.split('/')[-1].split('_')[7:])
                first = '_'.join(args.model_dir.split('/')[-1].split('_')[0:7])
                model_dir = args.model_dir.replace(current, f'competence-0.01-0.9-{args.steps[j]}_{args.dataset_name}_heuristics-{curric}')
                filenames.append(model_dir.replace(current_seed, str(seed)).strip('_'))

        elif curric == 'ppl':
            for j, seed in enumerate(args.seeds):
                current = '_'.join(args.model_dir.split('/')[-1].split('_')[7:])
                first = '_'.join(args.model_dir.split('/')[-1].split('_')[0:7])
                model_dir = args.model_dir.replace(current,
                                                   f'competence-0.01-0.9-{args.steps[j]}_{args.dataset_name}_{args.model_name}_ppl-ppl')
                filenames.append(model_dir.replace(current_seed, str(seed)).strip('_'))

        else:
            for seed in args.seeds:
                model_dir = args.model_dir.replace(current_curric, curric)
                filenames.append(model_dir.replace(current_seed, str(seed)))

        scores[curric] = {}
        scores[f'{curric}_steps'] = {}
        scores[f'{curric}_stdev'] = {}

        for filename in filenames:
            scores[curric].update({filename: []})
            scores[f'{curric}_steps'].update({filename: []})

            with open(os.path.join(filename, 'train.log'), 'r') as infile:
                for line in infile:
                    if re.match(r".* val-langs .*", line):

                        if 'ACCURACY = ' in line:
                            acc = line.rstrip().split('|')[2].strip('ACCURACY =')
                            step = line.rstrip().split('|')[0].split('-')[-1].strip(' step ')

                            scores[curric][filename] += [float(acc)]
                            scores[f'{curric}_steps'][filename] += [int(step)]

        cutoff_steps = np.min([len(scores[f'{curric}_steps'][f]) for f in filenames])
        tmp_scores = np.vstack([np.array(scores[curric][f])[:cutoff_steps] for f in filenames])

        # if curric == 'cross-review':
            # tmp_scores = tmp_scores[:2, :]
            # print(tmp_scores)

        scores[curric] = np.mean(tmp_scores, axis=0)
        scores[f'{curric}_max'] = np.max(tmp_scores, axis=0)
        scores[f'{curric}_min'] = np.min(tmp_scores, axis=0)

        scores[f'{curric}_steps'] = [scores[f'{curric}_steps'][f][:cutoff_steps] for f in filenames][0]

    return scores


def plot_curves(args, acc_scores):
    max_random = 0
    max_random_step = 0
    n = 8
    k = 2
    markers = ['>', '<', 'o', '^', 's']
    for curric in args.curricula[:-1]:
        # plot curric
        plt.plot(acc_scores[f'{curric}_steps'][::n], acc_scores[curric][::n], label=mapp[curric],
                 color=colors[k], marker=markers[k], markersize=5)
        plt.fill_between(acc_scores[f'{curric}_steps'][::n], acc_scores[f'{curric}_min'][::n], acc_scores[f'{curric}_max'][::n],
                         alpha=0.3)
        k += 2

    # plot random
    plt.plot(acc_scores['random_steps'][::n], acc_scores['random'][::n], label='Random',
             color='tab:red')
    plt.fill_between(acc_scores['random_steps'][::n], acc_scores['random_min'][::n], acc_scores['random_max'][::n],
                     alpha=0.3,
                     facecolor='tab:red')

    for i, j in zip(acc_scores['random'], acc_scores['random_steps']):
        if i > max_random:
            max_random = i
            max_random_step = j

    max_curric = 0
    max_curric_step = 0
    max_curric_name = 'random'
    flag = False
    all_best_curric = []
    plotted = []
    plotted_steps = []
    for k, curric in enumerate(args.curricula):
        print(curric)
        for c_perf, c_steps in zip(acc_scores[curric], acc_scores[f'{curric}_steps']):
            # best curriculum
            if c_perf > max_curric:
                max_curric = c_perf
                max_curric_step = c_steps
                max_curric_name = curric

            # closest curriculum with less steps
            if ((c_perf >= max_random) and (c_steps <= max_random_step) and curric not in plotted) or \
                    ((c_perf >= max_random) and (c_steps > max_random_step) and curric not in plotted):
                print(c_perf, c_steps, curric)

                flag = True
                plotted.append(curric)
                plotted_steps.append(c_steps)

            if (c_perf >= max_random) and (c_steps < max_random_step):
                all_best_curric += [{'perf': c_perf,
                                     'steps': c_steps,
                                     'name': curric,
                                     'time': np.round(c_steps/max_random_step, 4),
                                     'color': plt.gca().lines[k].get_color()}]

    logger.info(' Best curricula found')
    for abc in sorted(all_best_curric, key=lambda d: d['perf'], reverse=True):
        logger.info('\t\t'.join([str(b) for a, b in abc.items() if a != 'color']))

    if not flag:  # no better point found -- use max curric value
        print('No better point detected')
        max_curric = max_curric
        max_curric_step = max_curric_step
        max_curric_name = max_curric_name
    else:
        pass
        # sorted_curric = sorted(all_best_curric, key=lambda d: d['time'], reverse=False)
        # close_curric = float(sorted_curric[0]['perf'])
        # close_curric_step = int(sorted_curric[0]['steps'])
        # close_curric_name = sorted_curric[0]['name']

    logger.info(f' --> Maximum Random:  {max_random} @ {max_random_step}')
    # logger.info(f' --> Closest CURRICULA ({close_curric_name}): {close_curric} @ {close_curric_step/max_random_step}')
    logger.info(f' --> Maximum CURRIC ({max_curric_name}): {max_curric} @ {max_curric_step/max_random_step}')

    # =============== PLOT =================== #
    sns.despine()
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Training Steps')
    plt.xlim([0, max(plotted_steps) + 100])

    if args.dataset_name in ['pawsx']:
        # plt.title('PAWS-X')
        plt.ylabel('PAWS-X\nAccuracy (%)')
        plt.ylim([76, 88])
        plt.xlim([0, 15000])
        plt.yticks(np.arange(76, 89, 2))

    elif args.dataset_name in ['xnli']:
        plt.ylabel('XNLI\nAccuracy (%)')
        plt.ylim([64, 76])
        plt.yticks(np.arange(64, 77, 2))
        # plt.yticks(np.arange(66, 77, 2))
        # plt.title(args.dataset_name.upper())
        plt.xlim([0, 100000])
        plt.legend(loc='upper center', fontsize=12, ncol=3, bbox_to_anchor=(0.5, 1.3))

    elif args.dataset_name == 'mldoc':
        plt.ylabel('MLDoc\nAccuracy (%)')
        # plt.title('MLDoc')
        plt.ylim([70, 88])
        plt.yticks(np.arange(70, 89, 3))
        plt.xlim([0, 1400])

    else:
        plt.ylabel('XCOPA\nAccuracy (%)')
        plt.ylim([52, 64])
        plt.xlim([0, 17500])
        plt.yticks(np.arange(52, 65, 2))

    ax = plt.gca()
    ax.xaxis.grid()
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, pos: '{:,.0f}'.format(x / 1000) + 'K'))

    plot_file = f"../plots/{args.dataset_name}_smoothed.png"
    logger.info(f"Saving plot in {plot_file}")
    fig.tight_layout()
    fig.savefig(plot_file, bbox_inches="tight")
    fig.savefig(plot_file.replace('.png', '.pdf'), bbox_inches="tight")


def main(args):
    scores = get_scores(args)
    plot_curves(args, scores)


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
        level=logging.INFO
    )

    parser = argparse.ArgumentParser()
    parser.add_argument('--curricula', nargs='*', help='Curricula to test',
                        default=[#'annealing',
                                 #'annealing-bias',
                                 'cross-review',
                                 #'competence',
                                 'competence-bias',
                                 #'length',
                                 #'rarity',
                                 #'ppl',
                                 'random'])
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_name', type=str, default='xlm-roberta-base')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--steps', type=int, help='baseline best # steps', nargs='*')
    parser.add_argument('--seeds', type=int, nargs='*', default=['123', '456', '789'], help='Seeds to test')
    args = parser.parse_args()

    if args.model_dir.endswith('/'):
        args.model_dir = '/'.join(args.model_dir.split('/')[:-1])

    main(args)
    