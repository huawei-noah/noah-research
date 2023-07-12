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

import os, sys
import json
import datasets
import argparse
from datasets import set_caching_enabled, Value
import csv
import pandas as pd
import logging
from subprocess import check_output
from datasets import load_dataset
import ast

datasets.disable_caching()
datasets.utils.logging.set_verbosity_error()
csv.field_size_limit(sys.maxsize)

logger = logging.getLogger(__name__)


LANGS = {
    'pawsx': ['en', 'fr', 'es', 'de', 'zh', 'ja', 'ko'],
    'xnli': ['en', 'fr', 'es', 'de', 'el', 'bg', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'hi', 'sw', 'ur'],
    'xcopa': ['en', 'et', 'ht', 'id', 'it', 'qu', 'sw', 'ta', 'th', 'tr', 'vi', 'zh'],
    'mldoc': ['en', 'de', 'es', 'fr', 'it', 'ja', 'ru', 'zh'],
}


def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])


def process_nli(args, dataset):
    """
    Process SNLI or MNLI datasets.
    """
    raw_data = {
        data_split: load_dataset('json', data_files=dataset[data_split], split='train')
        for data_split in dataset.keys()
    }
    column_names = [c for c in raw_data['train'].column_names if c not in ['sentence1', 'sentence2', 'gold_label']]

    raw_data = datasets.DatasetDict(raw_data)
    # remove examples with -1 or - label as in Bowman et al. 2015
    raw_data = raw_data.filter(lambda example: example['gold_label'] in ['neutral', 'entailment', 'contradiction'])
    raw_data = raw_data.map(lambda example, idx: {'id': idx}, with_indices=True)  # add unique ids
    raw_data = raw_data.remove_columns(column_names)
    logger.info(raw_data)

    out_dir = os.path.join(args.out_data_dir, args.dataset_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    raw_data.save_to_disk(out_dir)


def process_mldoc(dataset_files, dat_name=None):
    """
    MLDOC dataset
    """
    label_map = set()

    def process(file):
        reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE,
                                fieldnames=['gold_label', 'sentence1'])
        df = pd.DataFrame(list(reader))
        new_file = datasets.Dataset.from_pandas(df)

        new_file = new_file.map(
            lambda example: {'sentence1': ast.literal_eval(example['sentence1']).decode("utf-8")})

        for l in new_file.unique('gold_label'):
            label_map.add(l)
        return new_file

    raw_data = {}
    for data_split in dataset_files.keys():
        filename = dataset_files[data_split]

        logger.info(f'Processing: {filename}')
        file = open(filename, encoding="utf-8")
        raw_data[data_split] = process(file)
        logger.info(raw_data[data_split])

    raw_data['label_map'] = datasets.Dataset.from_pandas(
        pd.DataFrame([{'label': label, 'id': i} for i, label in enumerate(sorted(label_map))])
    )

    raw_data = datasets.DatasetDict(raw_data)
    logger.info(raw_data)
    for i in raw_data['label_map']:
        print(i)
    logger.info(raw_data['label_map'])

    raw_data = raw_data.map(lambda example, idx: {'id': idx}, with_indices=True)  # add unique ids
    return raw_data


def process_paws(dataset_files, dat_name=None):
    def process(filepath):
        with open(filepath, encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)

            df = pd.DataFrame(list(reader), columns=['id', 'sentence1', 'sentence2', 'label'])
        dat = datasets.Dataset.from_pandas(df)
        return dat

    raw_data = {}
    for data_split in dataset_files.keys():
        filename = dataset_files[data_split]

        if os.path.isfile(filename):
            logger.info(filename)
            raw_data[data_split] = process(filename)
            print(raw_data[data_split][0])
            raw_data[data_split] = raw_data[data_split].rename_column('label', 'gold_label')
            raw_data[data_split] = raw_data[data_split].map(lambda example: {'gold_label': str(example['gold_label'])})
        else:
            logger.info(f"Could not find file: {filename} ! Skipping ...")
            continue

    raw_data = datasets.DatasetDict(raw_data)
    return raw_data


def process_twitter(dataset_files, dat_name=None):
    raw_data = []
    cnt = 0
    with open(dataset_files['sentence1']) as s1, open(dataset_files['sentence2']) as s2, open(dataset_files['labels']) as l:
        for line_s1, line_s2, line_label in zip(s1, s2, l):
            raw_data.append({'id': cnt, 'sentence1': line_s1.rstrip('\n'), 'sentence2': line_s2.rstrip('\n'),
                             'gold_label': line_label.rstrip('\n')})
            cnt += 1

    raw_data = datasets.DatasetDict({'test': datasets.Dataset.from_pandas(pd.DataFrame(raw_data))})
    print(raw_data['test'][0])
    return raw_data


def process_seqclass(dataset_files, dat_name=None):
    """
    Processes the PAWS dataset
    """
    def process(filepath):
        with open(filepath, encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE,
                                    fieldnames=['sentence1', 'sentence2', 'gold_label'])

            df = pd.DataFrame(list(reader), columns=['sentence1', 'sentence2', 'gold_label'])
        dat = datasets.Dataset.from_pandas(df)
        return dat

    def process_translations(filepath):
        with open(filepath, encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE,
                                    fieldnames=['orig_sentence1', 'orig_sentence2',
                                                'trans_sentence1', 'trans_sentence2', 'gold_label'])

            df = pd.DataFrame(list(reader), columns=['orig_sentence1', 'orig_sentence2',
                                                     'trans_sentence1', 'trans_sentence2', 'gold_label'])
        dat = datasets.Dataset.from_pandas(df)
        dat = dat.remove_columns(['orig_sentence1', 'orig_sentence2'])
        dat = dat.rename_column('trans_sentence1', 'sentence1')
        dat = dat.rename_column('trans_sentence2', 'sentence2')
        return dat

    raw_data = {}
    label_map = []
    for data_split in dataset_files.keys():
        filename = dataset_files[data_split]

        if data_split.startswith('tr-'):
            if os.path.isfile(filename):
                logger.info(filename)
                raw_data[data_split] = process_translations(filename)
            else:
                logger.info(f"Could not find file: {filename} ! Skipping ...")
                continue

        else:
            if os.path.isfile(filename):
                logger.info(filename)
                raw_data[data_split] = process(filename)
            else:
                logger.info(f"Could not find file: {filename} ! Skipping ...")
                continue

        # make sure labels are strings
        new_features = raw_data[data_split].features.copy()
        new_features["gold_label"] = Value('string')
        raw_data[data_split] = raw_data[data_split].cast(new_features)

        # add unique ids
        raw_data[data_split] = raw_data[data_split].map(lambda example, idx: {'id': idx}, with_indices=True)
        raw_data[data_split] = raw_data[data_split].map(
            lambda example: {'gold_label': example['gold_label'].replace('contradictory', 'contradiction')
            if example['gold_label'] is not None else None})
        label_map += raw_data[data_split].unique('gold_label')  # unique labels

    label_map = set([l for l in label_map if ((l != 'None') and (l is not None))])
    raw_data['label_map'] = datasets.Dataset.from_pandas(
        pd.DataFrame([{'label': label, 'id': i} for i, label in enumerate(sorted(label_map))]))

    raw_data = datasets.DatasetDict(raw_data)
    return raw_data


def process_siqa(dataset_files, dat_name=None):
    """
    SIQA dataset
    """
    raw_data = {}
    label_data = {}
    for data_split in dataset_files.keys():
        if '-labels' not in data_split:
            raw_data[data_split] = load_dataset('json', data_files=dataset_files[data_split], split='train')
            logger.info(raw_data[data_split])

        else:
            label_data[data_split] = load_dataset('text', data_files=dataset_files[data_split], split='train')
            logger.info(label_data[data_split])

    raw_data['train'] = raw_data['train'].add_column("text", label_data['train-labels']['text'])
    raw_data['val'] = raw_data['val'].add_column("text", label_data['val-labels']['text'])
    logger.info(raw_data)
    print(raw_data['train'][0])
    print(raw_data['train'].unique('text'))

    mapping = {'1': 0, '2': 1, '3': 2}
    raw_data = datasets.DatasetDict(raw_data)
    raw_data = raw_data.rename_column('context', 'premise')
    raw_data = raw_data.rename_column('answerA', 'choice1')
    raw_data = raw_data.rename_column('answerB', 'choice2')
    raw_data = raw_data.rename_column('answerC', 'choice3')
    raw_data = raw_data.rename_column('text', 'label')
    raw_data = raw_data.map(lambda example: {'label': mapping[example['label']]})
    raw_data = raw_data.map(lambda example, idx: {'id': idx}, with_indices=True)  # add unique ids

    return raw_data


def process_xcopa(dataset_files, dat_name=None):
    """
    XCOPA dataset
    """
    raw_data = {}
    for data_split in dataset_files.keys():
        logger.info(f'Processing: {data_split}')
        raw_data[data_split] = load_dataset('json', data_files=dataset_files[data_split], split='train')
        logger.info(raw_data[data_split])

        if data_split != 'val-en' and data_split != 'test-en' and data_split != 'train' and data_split != 'tr-test-en':
            raw_data[data_split] = raw_data[data_split].remove_columns('changed')

    raw_data = datasets.DatasetDict(raw_data)
    raw_data = raw_data.rename_column('idx', 'id')
    return raw_data


def process_glue(dataset_files, dat_name=None):
    data_dict = datasets.load_from_disk(os.path.join(dataset_files, dat_name))
    print(data_dict)

    class_lab = data_dict['train'].features['label']
    data_dict = data_dict.rename_column('idx', 'id')

    if dat_name == 'qnli':
        data_dict = data_dict.rename_column('question', 'sentence1')
        data_dict = data_dict.rename_column('sentence', 'sentence2')
        data_dict = datasets.DatasetDict({'train': data_dict['train'], 'validation': data_dict['validation']})

    if dat_name == 'mnli':
        data_dict = data_dict.rename_column('premise', 'sentence1')
        data_dict = data_dict.rename_column('hypothesis', 'sentence2')
        data_dict = datasets.DatasetDict({'train': data_dict['train'],
                                          'validation_matched': data_dict['validation_matched'],
                                          'validation_mismatched': data_dict['validation_mismatched']})

    if dat_name == 'rte':
        data_dict = datasets.DatasetDict({'train': data_dict['train'], 'validation': data_dict['validation']})

    print(class_lab)
    print(data_dict['train'][0])
    data_dict = data_dict.map(lambda ex: {'gold_label': class_lab.int2str(ex['label'])}, batched=False)
    data_dict = data_dict.remove_columns(['label'])
    print(data_dict['train'][0])

    label_map = data_dict['train'].unique('gold_label')
    print(label_map)
    data_dict['label_map'] = datasets.Dataset.from_pandas(
        pd.DataFrame([{'label': label, 'id': class_lab.str2int(label)} for label in label_map]))

    if dat_name == 'mnli':
        raw_data = {'train': data_dict['train'],
                    'val': data_dict['validation_matched'],
                    'test': data_dict['validation_mismatched'],
                    'label_map': data_dict['label_map']}

    elif dat_name in ['rte', 'qnli']:
        new_split = data_dict['train'].train_test_split(test_size=0.05, seed=42)
        raw_data = {'train': new_split['train'],
                    'val': new_split['test'],
                    'test': data_dict['validation'],
                    'label_map': data_dict['label_map']}

    print(data_dict['label_map'][0])
    return datasets.DatasetDict(raw_data)


class DataFiles:
    """
    Data class
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def filename(self, name):
        pathfile = os.path.join(self.data_dir, name)

        if name in ['xnli', 'pawsx']:
            files = {'train': os.path.join(pathfile, 'train-en.tsv')}
            files.update({f'val-{lang}': os.path.join(pathfile, f'dev-{lang}.tsv') for lang in LANGS[name]})
            files.update({f'test-{lang}': os.path.join(pathfile, f'test-{lang}.tsv') for lang in LANGS[name]})
            files.update({f'tr-test-{lang}': os.path.join(pathfile, 'tr-test', f'test-{lang}-en-translated.tsv')
                          for lang in LANGS[name]})
            return files

        elif name == 'mldoc':
            files = {'train': os.path.join(self.data_dir, 'mldoc_corpus', 'en.train.10000')}
            files.update({f'val-{lang}': os.path.join(self.data_dir, 'mldoc_corpus', f'{lang}.dev') for lang in LANGS['mldoc']})
            files.update({f'test-{lang}': os.path.join(self.data_dir, 'mldoc_corpus', f'{lang}.test') for lang in LANGS['mldoc']})
            return files

        elif name == 'xcopa':
            files = {}
            files.update({f'val-{lang}': os.path.join(pathfile, 'data', f'{lang}', f'val.{lang}.jsonl') for lang in LANGS['xcopa'] if lang not in ['en']})
            files.update({f'val-en': os.path.join(self.data_dir, 'COPA', 'val.jsonl')})
            files.update({f'test-{lang}': os.path.join(pathfile, 'data', f'{lang}', f'test.{lang}.jsonl') for lang in LANGS['xcopa'] if lang not in ['en']})
            files.update({f'tr-test-{lang}': os.path.join(pathfile, 'data-gmt', f'{lang}',
                                                          f'test.{lang}.jsonl') for lang in LANGS['xcopa'] if lang not in ['en', 'qu']})
            files.update({f'test-en': os.path.join(self.data_dir, 'COPA', 'test.jsonl')})
            return files

        elif name == 'siqa':
            files = {'train': os.path.join(self.data_dir, 'socialiqa-train-dev', 'train.jsonl'),
                     'train-labels': os.path.join(self.data_dir, 'socialiqa-train-dev', 'train-labels.lst'),
                     'val': os.path.join(self.data_dir, 'socialiqa-train-dev', 'dev.jsonl'),
                     'val-labels': os.path.join(self.data_dir, 'socialiqa-train-dev', 'dev-labels.lst')}
            return files

        elif name == 'paws':
            files = {'test': os.path.join(self.data_dir, 'paws', 'test.tsv')}
            return files

        elif name == 'twitter-ppdb':
            files = {'sentence1': os.path.join(self.data_dir, 'TwitterPPDB', 'sentence1.txt'),
                     'sentence2': os.path.join(self.data_dir, 'TwitterPPDB', 'sentence2.txt'),
                     'labels': os.path.join(self.data_dir, 'TwitterPPDB', 'labels.txt')}
            return files

        elif name in ['mnli', 'qnli', 'rte']:
            files = self.data_dir
            return files

        else:
            print('Dataset not supported')
            exit(0)


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str,
                        choices=['xnli', 'pawsx', 'mldoc', 'xcopa', 'siqa', 'paws', 'xcopa', 'twitter-ppdb',
                                 'qnli', 'rte', 'mnli'])
    parser.add_argument('--data_dir', type=str,
                        default='../../original_data/')
    parser.add_argument('--out_data_dir', type=str,
                        default='../../data/')
    args = parser.parse_args()

    # Take appropriate files/folders
    datafiles = DataFiles(args.data_dir).filename(args.dataset_name)

    datasets_dict = {
        'xnli': process_seqclass,
        'pawsx': process_seqclass,
        'siqa': process_siqa,
        'mldoc': process_mldoc,
        'xcopa': process_xcopa,
        'paws': process_paws,
        'twitter-ppdb': process_twitter,
        'qnli': process_glue,
        'rte': process_glue,
        'mnli': process_glue,
    }

    # Process
    final_dat = datasets_dict[args.dataset_name](datafiles, args.dataset_name)
    logger.info(final_dat)

    # save
    out_dir = os.path.join(args.out_data_dir, args.dataset_name)
    logger.info(f'Saving data in {out_dir}')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    final_dat.save_to_disk(out_dir)
