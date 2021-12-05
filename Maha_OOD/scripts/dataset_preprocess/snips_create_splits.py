# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from argparse import ArgumentParser
from pathlib import Path
from itertools import chain, combinations
import random
from pandas import read_csv


def read_snips_csv(path):
    df = read_csv(path)
    df.columns = ["text", "labels"]
    return df


def generate_complement(input_set, universal_set):
    complement_set = []
    for u in universal_set:
        if u not in input_set:
            complement_set.append(u)
    return complement_set


def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


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


def get_splits(label_space, verbose, in_domain_ratio, number_of_splits):
    print("Generating Folders With Unsupervised Splits in the Given Ratio, with ratio being for in-domain domains")
    print("Loading label space")
    if verbose:
        print("Label Space:", label_space)
    label_ids = label_space.keys()
    train_n = sum(label_space.values())
    if verbose:
        print("Generating powersets")
    all_powersets = list(powerset(label_ids))
    if verbose:
        print("Total Number Of Powersets:", len(all_powersets))
    all_powerset_lengths = [sum([label_space[class_name] for class_name in powerset]) for powerset in all_powersets]
    max_domain_ratio = in_domain_ratio * 1.15
    min_domain_ratio = in_domain_ratio * 0.85
    acceptable_powersets = []
    if verbose:
        print("Finding acceptable powersets")
    for i, pset in enumerate(all_powersets):
        if min_domain_ratio * train_n <= all_powerset_lengths[i] <= max_domain_ratio * train_n:
            acceptable_powersets.append(pset)
            if verbose:
                print("Accepted Set:", pset)
                print("Accepted Set Length:", all_powerset_lengths[i], "Total Length:", train_n, "Ratio",
                      all_powerset_lengths[i] / train_n)
                print("Complement Set:", generate_complement(pset, label_ids))
    print("Number Of Accepted Sets:", len(acceptable_powersets))
    random.shuffle(acceptable_powersets)
    acceptable_powersets = acceptable_powersets[:number_of_splits]
    return acceptable_powersets


if __name__ == '__main__':
    parser = ArgumentParser()
    dataset_seed = 4242
    random.seed(dataset_seed)

    parser.add_argument("--load_path", type=Path)
    args = parser.parse_args()

    train_df = read_snips_csv(args.load_path / "train.csv")
    val_df = read_snips_csv(args.load_path / "valid.csv")
    test_df = read_snips_csv(args.load_path / "test.csv")
    print(len(train_df), len(val_df), len(test_df))
    label_space = train_df.labels.value_counts()
    for K in [0.25, 0.75]:
        ind_classes = get_splits(label_space.to_dict(), False, K, 5)
        print("\n".join([" ".join(x) for x in ind_classes]))
        for num, in_class in enumerate(ind_classes):
            final_train_df, final_val_df, final_test_df = create_final_data(
                train_df, val_df, test_df, in_class)
            print(len(final_train_df))
            final_train_df.to_csv(f"{args.load_path.absolute()}/snips_train_{int(K * 100)}_{num}.csv", index=False)
            final_val_df.to_csv(f"{args.load_path.absolute()}/snips_val_{int(K * 100)}_{num}.csv", index=False)
            final_test_df.to_csv(f"{args.load_path.absolute()}/snips_test_{int(K * 100)}_{num}.csv", index=False)
