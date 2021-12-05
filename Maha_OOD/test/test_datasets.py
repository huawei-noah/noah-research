# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import itertools

import pytest
from transformers import BertTokenizer

from lib.data_utils import Vocab
from lib.datasets.datasets import get_dataset_simple, get_dataset_transformers


def label_vocab_test(dataset_1, dataset_2):
    assert dataset_1.label_vocab == dataset_2.label_vocab
    assert "oos" not in dataset_1.label_vocab


def ood_test(train, val, test):
    assert train.n_ood == 0
    assert train.n_indomain
    assert val.n_ood != 0
    assert val.n_indomain < len(val)
    assert test.n_ood != 0
    assert test.n_indomain < len(test)


def _sanity_check(train_dataset, val_dataset, test_dataset):
    def test_single(dataset):
        assert len(dataset.raw_texts) > 1
        assert len(dataset.raw_texts) == len(dataset.raw_labels)
        assert dataset.vectorized_texts
        for idx in range(0, len(dataset.raw_texts), 1000):
            assert " ".join(dataset.tokenized_texts[idx]) == " ".join(dataset.tokenized_texts[idx]).lower()

    test_single(train_dataset)
    test_single(val_dataset)
    test_single(test_dataset)

    label_vocab_test(train_dataset, val_dataset)
    label_vocab_test(train_dataset, test_dataset)

    ood_test(train_dataset, val_dataset, test_dataset)


def _vectorize_simple(train_dataset, val_dataset, test_dataset):
    vocab = Vocab()
    vocab.build(train_dataset.tokenized_texts + val_dataset.tokenized_texts)

    train_dataset.vectorize_texts(vocab)
    val_dataset.vectorize_texts(vocab)
    test_dataset.vectorize_texts(vocab)


@pytest.mark.parametrize("K,version", itertools.product([25, 75], [0, 1, 2, 3, 4]))
def test_snips_dataset_simple(K, version):
    datasets, vocab = get_dataset_simple(f'snips_{K}', version=version)
    train_dataset, val_dataset, test_dataset = datasets
    _vectorize_simple(train_dataset, val_dataset, test_dataset)
    _sanity_check(train_dataset, val_dataset, test_dataset)


@pytest.mark.parametrize("coarse", [True, False])
def test_rostd_dataset_simple(coarse):
    dataset_name = "rostd_coarse" if coarse else "rostd"
    datasets, vocab = get_dataset_simple(dataset_name=dataset_name)
    train_dataset, val_dataset, test_dataset = datasets
    _vectorize_simple(train_dataset, val_dataset, test_dataset)
    _sanity_check(train_dataset, val_dataset, test_dataset)
    if coarse:
        assert len(train_dataset.label_vocab) == 3
    else:
        assert len(train_dataset.label_vocab) == 12


def test_clinc_dataset_simple():
    datasets, vocab = get_dataset_simple('clinc')
    train_dataset, val_dataset, test_dataset = datasets
    _vectorize_simple(train_dataset, val_dataset, test_dataset)
    _sanity_check(train_dataset, val_dataset, test_dataset)


@pytest.fixture(scope="session")
def bert_tok():
    return BertTokenizer.from_pretrained("bert-base-uncased")


@pytest.mark.transformer
def test_snips_dataset_bert(bert_tok):
    train_dataset, val_dataset, test_dataset = get_dataset_transformers(bert_tok, 'snips_75', version=0)
    _sanity_check(train_dataset, val_dataset, test_dataset)


@pytest.mark.transformer
def test_rostd_dataset_bert(bert_tok):
    train_dataset, val_dataset, test_dataset = get_dataset_transformers(bert_tok, 'rostd')
    _sanity_check(train_dataset, val_dataset, test_dataset)


@pytest.mark.transformer
def test_clinc_dataset_bert(bert_tok):
    train_dataset, val_dataset, test_dataset = get_dataset_transformers(bert_tok, 'clinc')
    _sanity_check(train_dataset, val_dataset, test_dataset)
