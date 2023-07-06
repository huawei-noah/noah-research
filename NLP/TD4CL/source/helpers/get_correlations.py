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
import argparse
import json
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from itertools import combinations, permutations
import numpy as np

pd.options.mode.chained_assignment = None

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def main(args):
    logger.info('Using dynamics ...')
    dynamics = {}
    for dyn in args.dynamics:
        logger.info(dyn)
        metric=dyn

        if dyn in ['correctness', 'confidence', 'variability']:
            dyn_file = f'{args.dataset_name}_{args.model_name}_123_LR{args.lr}_LEN128_BS{args.bs}_E{args.epochs}'

        elif dyn in ['rarity', 'length']:
            dyn_file = f'{args.dataset_name}_heuristics'

        elif dyn == 'cross-review':
            dyn_file = f'{args.dataset_name}_{args.model_name}_123_LR{args.lr}_LEN128_BS{args.bs}_E{args.epochs}_cross-review'
            metric='correctness'
        else:
            print('here')
            dyn_file = f'{args.dataset_name}_{args.model_name}_ppl'
            metric='ppl'

        with open(os.path.join(args.dynamics_dir, dyn_file + '.json'), 'r') as infile:
            tmp = json.load(infile)
            tmp = sorted(tmp.items())

        dynamics[dyn] = [t[1][metric] for t in tmp]

    # Get Spearman correlations
    correlations = {}
    for dyn1, dyn2 in combinations(args.dynamics, 2):
        if dyn1 not in correlations:
            correlations[dyn1] = {}
        rho, pval = spearmanr(dynamics[dyn1], dynamics[dyn2])
        logger.info(f'{dyn1} & {dyn2}: {rho}')
        # exit(0)

        if dyn2 not in correlations[dyn1]:
            correlations[dyn1][dyn2] = np.round(rho, 2)

    ## PLOT
    sns.set(style='ticks', font_scale=1.3, font='Georgia')
    sns.set_context('talk')
    fig = plt.figure(figsize=(8, 6), dpi=300)

    cor = np.ones((len(args.dynamics), len(args.dynamics)))
    matrix = np.triu(cor)
    df = pd.DataFrame(data=correlations)
    ax = sns.heatmap(df, linewidths=1, annot=True, square=False, vmin=-1.0, vmax=1.0,
                     cmap=sns.diverging_palette(20, 220, as_cmap=True),
                     cbar=False if args.dataset_name in ['pawsx', 'xnli'] else True)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

    plot_file = f'../plots/{args.dataset_name}_{args.model_name}_correlations.png'
    print(f' Saving plot in {plot_file}')
    fig.tight_layout()
    fig.savefig(plot_file)
    fig.savefig(plot_file.replace('.png', '.pdf'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--lr', type=str)
    parser.add_argument('--bs', type=str)
    parser.add_argument('--epochs', type=str)
    parser.add_argument('--dynamics', type=str,
                        default=['correctness', 'confidence', 'variability', 'cross-review', 'length', 'rarity', 'PPL'])
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--dynamics_dir', type=str, default='../../dynamics/')
    args = parser.parse_args()

    os.system('mkdir -p ../../plots/')
    main(args)
