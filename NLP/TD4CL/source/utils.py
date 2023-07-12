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

from metrics import *
import random
import pandas as pd
import datasets
import torch
import os
import json
from transformers import set_seed


logger = logging.getLogger(__name__)

METRICS = {
    'xnli': 'accuracy',
    'pawsx': 'accuracy',
    'xcopa': 'accuracy',
    'mldoc': 'accuracy'
}


def init_seed(given_seed):
    """Initializes random seeds for sources of randomness. If seed is -1, randomly select seed.
    Sets the random seed for sources of randomness (numpy, torch and python random). If seed is
    specified as -1, the seed will be randomly selected and used to initialize all random seeds.
    The value used to initialize the random seeds is returned.
    Args:
        given_seed (int): random seed.
    Returns:
        int: value used to initialize random seeds.
    """
    used_seed = get_seed(given_seed)
    random.seed(used_seed)
    np.random.seed(used_seed)
    torch.manual_seed(used_seed)
    torch.cuda.manual_seed_all(used_seed)
    torch.backends.cudnn.deterministic = True

    # MAKE SURE THIS IS SET
    logger.info("Using seed: {}".format(used_seed))
    return used_seed


def get_seed(seed):
    """Get random seed if seed is specified as -1, otherwise return seed.
    Args:
        seed (int): random seed.
    Returns:
        int: Random seed if seed is specified as -1, otherwise returns the provided input seed.
    """
    if seed == -1:
        return int(np.random.randint(0, 99999))
    else:
        return seed


def show_random_examples(dataset, num_examples=10, rand=False):
    """
    Show #num_examples examples from the data
    """
    assert num_examples <= len(dataset)
    picks = []

    for i in range(num_examples):
        if rand:
            pick = random.randint(0, len(dataset)-1)
            while pick in picks:
                pick = random.randint(0, len(dataset)-1)
            picks.append(pick)
        else:
            picks.append(i)

    df = pd.DataFrame(dataset[picks])
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        logger.info(df)


def humanized_time(second):
    """
    Print time in hours:minutes:seconds
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def setup_logger(config):
    config['model_dir'] = os.path.join(config['save_dir'], config['model_dir'])

    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%d/%m/%Y %H:%M:%S',
        level=logging.INFO,
        filename=os.path.join(config['model_dir'], config['mode']+'.log'),
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    console.setFormatter(formatter)
    log = logging.getLogger('')
    log.addHandler(console)

    # save training tracking
    with open(os.path.join(config['model_dir'], 'args_config.json'), 'w') as config_outfile:
        json.dump(config, config_outfile)
    return config


def calculate_scores(config, predictions, ground_truth, lang='en'):
    """
    Calculate a score based on the target task.
    Args:
        config: config file
        predictions: predictions (numpy array or list of dicts)
        ground_truth: true labels (numpy array or list of dicts)
        lang: Target language

    Returns: Dictionary with scores
    """
    if config['task'] in ['seq_class', 'multi_choice']:
        return {'accuracy': accuracy(predictions, ground_truth)}
    elif config['task'] == 'qa':
        res = evaluate_qa(predictions, ground_truth, lang, config['dataset_name'])
        return {'f1': res['f1'], 'em': res['em'], 'avg': (res['f1'] + res['em']) / 2}


def write_avg2file(dataset_name, results):
    val_keys, test_keys = [], []
    val_scores, test_scores = [], []
    for k, v in results.items():  # languages
        if k.startswith('val') and (METRICS[dataset_name] in v):
            val_keys += [k.split('-')[-1]]
            val_scores += [float(v[METRICS[dataset_name]])]
        if k.startswith('test') and (METRICS[dataset_name] in v):
            test_keys += [k.split('-')[-1]]
            test_scores += [float(v[METRICS[dataset_name]])]

    if val_keys:
        logger.info('   \t'+'\t'.join(val_keys + ['AVG']))
        crop_scores = list(map(str, np.round(val_scores, 1)))
        logger.info('VAL\t{}\t{}\n'.format('\t'.join(crop_scores), np.round(np.mean(val_scores), 2)))

    if test_keys:
        logger.info('    \t'+'\t'.join(test_keys + ['AVG']))
        crop_scores = list(map(str, np.round(test_scores, 1)))
        logger.info('TEST\t{}\t{}\n'.format('\t'.join(crop_scores), np.round(np.mean(test_scores), 2)))
