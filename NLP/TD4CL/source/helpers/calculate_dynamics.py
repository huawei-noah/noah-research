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
import numpy as np
import argparse
from glob import glob
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging
from scipy.special import softmax
from scipy.spatial import distance
from itertools import combinations
import datasets
pd.options.mode.chained_assignment = None


logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%d/%m/%Y %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def plot_cartography(dynamics, exp_name):
    """
    Plot a cartography table
    Based on: https://www.aclweb.org/anthology/2020.emnlp-main.746/
    """
    logger.info(' Plotting cartography table')
    fig = plt.figure()

    new_dynamics = []
    for id_, item in tqdm(dynamics.items()):
        item['id'] = id_
        new_dynamics += [item]
    dataframe = pd.DataFrame(new_dynamics)

    plt.rcParams['figure.dpi'] = 300
    sns.set(style='whitegrid', font_scale=1.5, font='Georgia', context='notebook')

    # plot a "total" map, then break down
    dataframe = dataframe.sample(n=25000 if dataframe.shape[0] > 25000 else len(dataframe))

    # PLOT
    ax = sns.scatterplot(data=dataframe, x='variability', y='confidence', hue='correctness', s=30,
                    palette=sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True))

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], title='correctness', loc='center left', bbox_to_anchor=(1, 0.5))
    plot_file = f'../../plots/{exp_name}_datamap.png'
    logger.info(f' Saving plot in {plot_file}')
    fig.tight_layout()
    fig.savefig(plot_file)


def calc_dynamics(dynamics):
    """
    Calculate dynamics
    """
    logger.info('Calculating training dynamics ...')

    for item_id, stats in tqdm(dynamics.items()):
        dynamics[item_id]['confidence'] = float(np.mean(stats['confidence_trend']))
        dynamics[item_id]['variability'] = float(np.std(stats['confidence_trend']))
        dynamics[item_id]['correctness'] = int(np.sum(stats['class_trend']))  # number of times classified correctly
        dynamics[item_id].pop('confidence_trend')
        dynamics[item_id].pop('class_trend')
    return dynamics


def load_seq_class(pred_folder):
    dynamics = {}

    logger.info(pred_folder)
    logger.info('Loading epoch dynamics ...')
    for ep in range(int(args.epochs.split('-')[0]), int(args.epochs.split('-')[1])+1):
        for filename in sorted(glob(os.path.join(pred_folder,
                                                 f'{args.dataset_name}_epoch{ep}.json'))):
            logger.info(filename)
            with open(filename, 'r') as infile:
                for line in tqdm(infile):
                    line = json.loads(line)

                    probs = softmax(line['logits']) if 'logits' in line else line['probs']
                    gold = line['label'] if 'label' in line else line['labels']
                    pred = np.argmax(probs)
                    key = line['id']

                    if key not in dynamics:
                        dynamics[key] = {'confidence_trend': [], 'class_trend': []}

                    dynamics[key]['confidence_trend'].append(probs[gold])  # for the gold

                    if pred == gold:
                        dynamics[key]['class_trend'].append(1)  # correct classification
                    else:
                        dynamics[key]['class_trend'].append(0)  # incorrect classification

    return dynamics


def main(args):
    # Load dataset
    data_dict = datasets.load_from_disk(os.path.join(args.data_dir, args.dataset_name))

    if args.model_dir.endswith('/'):
        args.model_dir = '/'.join(args.model_dir.split('/')[:-1])

    exp_type = args.model_dir.split('/')[-1]

    # Load training statistics
    if args.dataset_name in ['pawsx', 'xnli', 'xcopa', 'mldoc', 'siqa', 'qnli', 'rte', 'mnli']:
        model_dynamics = load_seq_class(args.model_dir)
    else:
        model_dynamics = None

    # Calculate dynamics
    dynamics = calc_dynamics(model_dynamics)

    plot_cartography(dynamics, exp_type)

    # Write dynamics to file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file = os.path.join(args.output_dir, f'{exp_type}.json')

    logger.info(f' Writing dynamics statistics to {output_file}')
    with open(output_file, 'w') as outfile:
        json.dump(dynamics, outfile)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='../../dynamics/')
    parser.add_argument('--data_dir', type=str, default='../../data/')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--epochs', type=str, help='FromEpochX-ToEpochY')
    args = parser.parse_args()

    os.system('mkdir -p ../../plots/')
    main(args)
