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

import logging
import json
import datasets
import os
from datasets import load_dataset, concatenate_datasets
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


LANGS = {
    'pawsx': ['en', 'fr', 'es', 'de', 'zh', 'ja', 'ko'],
    'xnli': ['en', 'fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'hi', 'sw', 'ur'],
    'xcopa': ['en', 'et', 'ht', 'id', 'it', 'qu', 'sw', 'ta', 'th', 'tr', 'vi', 'zh'],
    'mldoc': ['en', 'fr', 'de', 'es', 'it', 'ja', 'ru', 'zh'],
    'mlqa': ['en', 'de', 'es', 'ar', 'zh', 'vi', 'hi']
}


class DataFiles:

    """
    Class to facilitate data loading
    """
    def __init__(self, config):
        self.data_dir = config['data_dir']
        self.all_data_dir = config['all_data_dir']
        self.glue_dir = config['glue_dir']
        self.dynamics_dir = config['dynamics_dir']
        self.config = config

    def filename(self, name):
        logger.info(f'*** Using the {name.upper()} dataset in {self.config["mode"].upper()} mode ***')

        # training mode
        if self.config['mode'] == 'train':

            # GLUE
            if name in ['mnli', 'qnli', 'qqp', 'rte']:
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {'train': data_dict['train'],
                            'val': data_dict['val'],
                            'label_map': data_dict['label_map']}

            elif name in ['xcopa'] and self.config['model_name'].startswith('xlm-'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {'train': data_dict['train'].map(lambda example: {'id': f"en-{example['id']}"}),
                            'val': data_dict['val-en'].map(lambda example: {'id': f"en-{example['id']}"}),
                            'val-langs': concatenate_datasets(
                                [data_dict[f'val-{lang}'] for lang in LANGS[name] ]
                            )}

            elif name == 'siqa' and self.config['model_name'].startswith('xlm-'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                xcopa = datasets.load_from_disk(os.path.join(self.data_dir, 'xcopa'))
                raw_data = {'train': data_dict['train'].map(lambda example: {'id': f"en-{example['id']}"}),
                            'val': xcopa['val-en'].map(lambda example: {'id': f"en-{example['id']}"}),
                            'val-langs': concatenate_datasets(
                                [xcopa[f'val-{lang}'] for lang in LANGS['xcopa'] ]
                            )}

            elif name == 'siqa' and self.config['model_name'].startswith('roberta-'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                xcopa = datasets.load_from_disk(os.path.join(self.data_dir, 'xcopa'))
                raw_data = {'train': data_dict['train'].map(lambda example: {'id': f"en-{example['id']}"}),
                            'val': xcopa['val-en'].map(lambda example: {'id': f"en-{example['id']}"})}

            elif name in ['xcopa'] and self.config['model_name'].startswith('roberta-'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {'train': data_dict['train'].map(lambda example: {'id': f"en-{example['id']}"}),
                            'val': data_dict['val-en'].map(lambda example: {'id': f"en-{example['id']}"})}

            elif name in ['xnli', 'pawsx', 'mldoc'] and self.config['model_name'].startswith('xlm-'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {'train': data_dict['train'],
                            'val': data_dict['val-en'],
                            'val-langs': concatenate_datasets([data_dict[f'val-{lang}'] for lang in LANGS[name]]),
                            'label_map': data_dict['label_map']}

            elif name in ['xnli', 'pawsx', 'mldoc'] and self.config['model_name'].startswith('roberta-'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {'train': data_dict['train'],
                            'val': data_dict[f'val-en'],
                            'test': data_dict[f'test-en'],
                            'label_map': data_dict['label_map']}

            elif '-en' in name:
                if 'xcopa' in name:
                    data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name.split('-en')[0]))
                    raw_data = {'train': data_dict['train'],
                                'val': data_dict['val-en']}
                else:
                    data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name.split('-en')[0]))
                    raw_data = {'train': data_dict['train'],
                                'val': data_dict[f'val-en'],
                                'label_map': data_dict['label_map']}

            elif name.startswith('pawsx'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'pawsx'))
                raw_data = {'train': data_dict,
                            'val': val_n_map['val-en'],
                            'val-langs': concatenate_datasets([val_n_map[f'val-{lang}'] for lang in LANGS['pawsx']]),
                            'label_map': val_n_map['label_map']}

            elif name.startswith('mldoc'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'mldoc'))
                raw_data = {'train': data_dict,
                            'val': val_n_map['val-en'],
                            'val-langs': concatenate_datasets([val_n_map[f'val-{lang}'] for lang in LANGS['mldoc']]),
                            'label_map': val_n_map['label_map']}

            elif name.startswith('xnli'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'xnli'))
                raw_data = {'train': data_dict,
                            'val': val_n_map['val-en'],
                            'val-langs': concatenate_datasets([val_n_map[f'val-{lang}'] for lang in LANGS['xnli']]),
                            'label_map': val_n_map['label_map']}

            elif name.startswith('siqa'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'xcopa'))
                raw_data = {'train': data_dict,
                            'val': val_n_map['val-en'].map(lambda example: {'id': f"en-{example['id']}"}),
                            'val-langs': concatenate_datasets([val_n_map[f'val-{lang}'] for lang in LANGS['xcopa']])
                            }

            elif name.startswith('mnli') or name.startswith('qnli') or name.startswith('rte'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))

                if name.startswith('qnli'):
                    val_ = datasets.load_from_disk(os.path.join(self.glue_dir, 'qnli'))

                elif name.startswith('rte'):
                    val_ = datasets.load_from_disk(os.path.join(self.glue_dir, 'rte'))

                else:
                    val_ = None

                raw_data = {'train': data_dict, 'val': val_['val'], 'label_map': val_['label_map']}

        # evaluation mode
        elif self.config['mode'] == 'eval':
            if name in ['mnli', 'qnli', 'rte']:
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))

                if name == 'mnli':
                    raw_data = {'val': data_dict['val'],
                                'test': data_dict['test'],
                                'label_map': datasets.load_from_disk(os.path.join(self.data_dir, 'xnli'))['label_map']}
                else:
                    raw_data = {'val': data_dict['val'],
                                'test': data_dict['test'],
                                'label_map': data_dict['label_map']}

            elif name in ['xcopa']:
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {f'val-{lang}': data_dict[f'val-{lang}'] for lang in LANGS[name]}
                raw_data.update({f'test-{lang}': data_dict[f'test-{lang}'] for lang in LANGS[name]})

            elif name in ['mldoc', 'xnli', 'pawsx']:
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {f'val-{lang}': data_dict[f'val-{lang}'] for lang in LANGS[name]}
                raw_data.update({f'test-{lang}': data_dict[f'test-{lang}'] for lang in LANGS[name]})
                raw_data['label_map'] = data_dict['label_map']

            elif name == 'paws':
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, 'paws'))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'pawsx'))
                raw_data = {'test': data_dict['test'],
                            'label_map': val_n_map['label_map']}

            elif name == 'siqa':
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, 'siqa'))
                raw_data = {'val': data_dict['val']}
                raw_data.update({'test': data_dict['test']})

            elif name == 'twitter-ppdb':
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'pawsx'))
                raw_data = {'test': data_dict['test'],
                            'label_map': val_n_map['label_map']}

            elif name == 'csqa':
                map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))['validation']
                data_dict = data_dict.map(lambda x, i: {'premise': '',
                                                        'id': i,
                                                        'choice1': x['choices']['text'][0],
                                                        'choice2': x['choices']['text'][1],
                                                        'choice3': x['choices']['text'][2],
                                                        'choice4': x['choices']['text'][3],
                                                        'choice5': x['choices']['text'][4],
                                                        'label': map[x['answerKey']]
                                                        }, with_indices=True, remove_columns=['choices'])

                logger.info(data_dict[0])
                logger.info(data_dict)
                raw_data = {'test': data_dict}

            elif name == 'hans':
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                data_dict = data_dict.rename_column('premise', 'sentence1')
                data_dict = data_dict.rename_column('hypothesis', 'sentence2')
                data_dict = data_dict.map(lambda x, idx: {'gold_label': 'not_entailment'
                if x['label'] == 1 else 'entailment', 'id': idx}, with_indices=True)
                logger.info(data_dict)

                val_n_map = datasets.load_from_disk(os.path.join(self.glue_dir, 'rte'))
                raw_data = {'test': data_dict['validation'],
                            'label_map': val_n_map['label_map']}

            elif name == 'nli-diagnostics':
                data_dict = {'test': load_dataset('csv',
                                                 data_files=os.path.join(self.data_dir, f'{name}.tsv'),
                                                 split='train',
                                                 delimiter='\t').rename_column('pairID', 'id')}

                raw_data = {
                    categ_name: data_dict['test'].filter(lambda example: categ_name in example['category'].split(';'))
                    for categ_name in ['logic', 'lexical_semantics', 'predicate_argument_structure', 'knowledge']
                }
                raw_data.update({'test': data_dict['test']})
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'xnli'))
                raw_data.update({'label_map': val_n_map['label_map']})
                logger.info(raw_data)

            elif name in ['squad_adversarial_AddOneSent_converted', 'squad_adversarial_AddSent_converted',
                          'squad_adversarial_converted']:
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                data_dict = data_dict.map(lambda example, idx: {"id": idx}, with_indices=True)
                val_n_map = datasets.load_from_disk(os.path.join(self.glue_dir, 'qnli'))
                raw_data = {'test': data_dict,
                            'label_map': val_n_map['label_map']}

            elif name.startswith('pawsx'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'pawsx'))
                raw_data = {'val': data_dict,
                            'label_map': val_n_map['label_map']}

            elif name.startswith('xnli'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'xnli'))
                raw_data = {'val': data_dict,
                            'label_map': val_n_map['label_map']}

            elif name.startswith('rte'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'rte'))
                raw_data = {'val': data_dict,
                            'label_map': val_n_map['label_map']}

            elif name.startswith('qnli'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'qnli'))
                raw_data = {'val': data_dict,
                            'label_map': val_n_map['label_map']}

            elif name.startswith('mldoc'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                val_n_map = datasets.load_from_disk(os.path.join(self.all_data_dir, 'mldoc'))
                raw_data = {'val': data_dict,
                            'label_map': val_n_map['label_map']}

            elif name.startswith('siqa'):
                data_dict = datasets.load_from_disk(os.path.join(self.data_dir, name))
                raw_data = {'val': data_dict}

        else:
            raw_data = None
            print('Error! No dataset loaded!')
            exit(0)

        if self.config['use_dynamics']:
            logger.info('Using dynamics ...')
            dat = os.path.join(self.dynamics_dir, self.config['use_dynamics'] + '.json')
            with open(dat, 'r') as infile:
                dynamics = json.load(infile)

            def augment(example):
                if 'cross-review' in self.config['use_dynamics']:
                    # for k, v in dynamics[str(example['id'].strip('en-'))].items():
                    for k, v in dynamics[str(example['id'])].items():
                        example[k] = v
                    return example
                else:
                    if name == 'siqa':
                        for k, v in dynamics[str(example['id'].strip('en-'))].items():
                            example[k] = v
                        return example
                    else:
                        for k, v in dynamics[str(example['id'])].items():
                            example[k] = v
                        return example

            raw_data['train'] = raw_data['train'].map(augment)

        return datasets.DatasetDict(raw_data)
