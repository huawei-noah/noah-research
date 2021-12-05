# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

"""
First, we extract SNIPS data
We then get 100 utts for each class in train to form dev data
Then we perform OOD splitting
We fix in-domain class proportion and select in-domain and OOD classes
Finally we save resulting datasets
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import random

from pandas import DataFrame

INTENT_NAMES = ['AddToPlaylist', 'BookRestaurant', 'GetWeather', 'PlayMusic',
                'RateBook', 'SearchCreativeWork', 'SearchScreeningEvent']
VAL_SIZE_PER_CLASS = 100


def extract_utterance(utt_data):
    return "".join([x['text'] for x in utt_data['data']])


def prepare_snips_dataset(load_path: Path):
    training_data = []
    training_labels = []
    validation_data = []
    validation_labels = []
    test_data = []
    test_labels = []
    for intent in INTENT_NAMES:
        intent_path = load_path / intent
        with (intent_path / f"train_{intent}_full.json").open(encoding='latin-1') as f:
            intent_train_data = json.load(f)[intent]
            random.Random(0).shuffle(intent_train_data)
            train_data_sample = intent_train_data[VAL_SIZE_PER_CLASS:]
            val_data_sample = intent_train_data[:VAL_SIZE_PER_CLASS]
            training_data.extend(map(extract_utterance, train_data_sample))
            validation_data.extend(map(extract_utterance, val_data_sample))
            training_labels.extend([intent] * len(train_data_sample))
            validation_labels.extend([intent] * len(val_data_sample))
        with (intent_path / f"validate_{intent}.json").open(encoding='latin-1') as f:
            test_data_sample = json.load(f)[intent]
            test_data.extend(map(extract_utterance, test_data_sample))
            test_labels.extend([intent] * len(test_data_sample))
    assert len(validation_data) == len(test_data)
    assert len(test_data) == len(INTENT_NAMES) * VAL_SIZE_PER_CLASS
    train_df = DataFrame({"text": training_data, "labels": training_labels})
    val_df = DataFrame({"text": validation_data, "labels": validation_labels})
    test_df = DataFrame({"text": test_data, "labels": test_labels})
    print(len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


def form_indomain_classes(train_df, K: float, seed: int):
    index = dict(train_df.labels.value_counts())
    threshold = int(K * len(train_df))
    in_domain_intents = []
    all_intents = list(index)
    random.Random(seed).shuffle(all_intents)
    total_ind_size = 0
    for idx, intent in enumerate(all_intents):
        total_ind_size += index[intent]
        in_domain_intents.append(intent)
        if total_ind_size > threshold:
            break
    return sorted(in_domain_intents)


def create_final_data(train_df, val_df, test_df, indomain_classes):
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df = train_df[train_df.labels.isin(indomain_classes)]
    train_df["is_ood"] = 0
    val_df["is_ood"] = 1
    val_df.loc[val_df.labels.isin(indomain_classes), "is_ood"] = 0
    test_df["is_ood"] = 1
    test_df.loc[test_df.labels.isin(indomain_classes), "is_ood"] = 0

    return train_df, val_df, test_df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--load_path", type=Path)
    parser.add_argument("--save_dir", type=Path)
    args = parser.parse_args()
    train_df, val_df, test_df = prepare_snips_dataset(args.load_path)
    for K in [0.25, 0.75]:
        ind_classes = []
        for seed in range(100):
            ind_classes.append(tuple(sorted(form_indomain_classes(train_df, K, seed))))
        ind_classes = sorted(set(ind_classes))[:5]
        print("\n".join([" ".join(x) for x in ind_classes]))
        for num, in_class in enumerate(ind_classes):
            final_train_df, final_val_df, final_test_df = create_final_data(
                train_df, val_df, test_df, in_class)
            print(len(final_train_df))
            final_train_df.to_csv(f"{args.save_dir.absolute()}/snips_train_{int(K * 100)}_{num}.csv", index=False)
            final_val_df.to_csv(f"{args.save_dir.absolute()}/snips_val_{int(K * 100)}_{num}.csv", index=False)
            final_test_df.to_csv(f"{args.save_dir.absolute()}/snips_test_{int(K * 100)}_{num}.csv", index=False)
