# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from collections import Counter
from typing import NoReturn

from nltk import word_tokenize
import numpy as np
import torch
from torch.utils.data.dataset import Dataset

from lib.data_utils import Vocab, partial_class
from lib.datasets.loaders import SNIPSLoader, ROSTDLoader, CLINC150Loader, SSTLoader


SUBSETS = ("train", "val", "test")


class OODDataset(Dataset):
    """Defines a dataset for out-of-domain detection."""
    def __init__(self, loader, tok_fn, return_intent_labels=True, to_lower_case=True):
        """
        Create OODDataset given a data loader and a tokenization function.

        Args:
            loader: dataset loader
            tok_fn: tokenization function
            return_intent_labels: whether to return intent labels with instance (default: True)
            to_lower_case: whether to lowercase text data
        """
        super().__init__()
        self.loader = loader
        for attr in ["raw_texts", "raw_labels", "ood_labels"]:
            setattr(self, attr, getattr(self.loader, attr))
        self.n_ood = sum(self.ood_labels)
        self.n_indomain = len(self) - self.n_ood

        if to_lower_case:
            self.raw_texts = [t.lower() for t in self.raw_texts]
        self.tokenized_texts = [tok_fn(t) for t in self.raw_texts]
        self.vectorized_texts = None
        self.return_intent_labels = return_intent_labels
        self.label_vocab, self.vectorized_labels, self.label_cnts = self.vectorize_labels()
        self.encoder = None

    def __len__(self):
        return len(self.raw_texts)

    def __getitem__(self, idx):
        if self.return_intent_labels:
            return self.vectorized_texts[idx], self.vectorized_labels[idx], self.ood_labels[idx]
        return self.vectorized_texts[idx], self.ood_labels[idx]

    def vectorize_labels(self):
        """
        Map raw labels onto their numerical representation.

        Returns:
            - label vocabulary, i.e mapping from labels to indexes
            - vectorized labels
            - label counts, i.e. number of instances in each class
        """
        label_counter = Counter(self.raw_labels)
        if 'oos' in label_counter:
            label_counter.pop('oos')
        unique_labels, label_cnts = zip(*sorted(label_counter.items()))
        unique_labels, label_cnts = list(unique_labels), list(label_cnts)
        label_vocab = {label: index for index, label in enumerate(unique_labels)}
        vectorized_labels = [label_vocab.get(label, -1) for label in self.raw_labels]
        return label_vocab, vectorized_labels, label_cnts

    def vectorize_texts(self, encoder) -> NoReturn:
        """
        Map tokenized texts into respective numerical sequences.

        Args:
            encoder: mapping from tokens to integer values
        """
        self.encoder = encoder
        self.vectorized_texts = [self.encoder.encode(t) for t in self.tokenized_texts]


# Some helper methods for faster build of common dataset setup
def get_simple_splits(loader_cls, add_valid_to_vocab=False, add_test_to_vocab=False,
                      return_intent_labels=True, tok_fn=word_tokenize, to_lower_case=True):
    """
    Get train/dev/test split

    Args:
        loader_cls: class that loads OOD dataset from the raw files.
        add_valid_to_vocab: whether to add words from the validation split to the vocabulary (default: False)
        add_test_to_vocab: whether to add words from the test split to the vocabulary (default: False)
        return_intent_labels: whether to return intent labels with instances (default: True)
        tok_fn: tokenization function
        to_lower_case: whether to lowercase text instances

    Returns:
        (tuple):
            mapping from split names (train/val/test) to Dataset instances
            vocabulary
    """
    datasets = {}
    for subset in SUBSETS:
        dataset = OODDataset(loader=loader_cls(subset=subset), tok_fn=tok_fn,
                             return_intent_labels=return_intent_labels,
                             to_lower_case=to_lower_case)
        datasets[subset] = dataset

    vocab = Vocab()
    texts = datasets['train'].tokenized_texts
    if add_valid_to_vocab:
        texts = texts + datasets['val'].tokenized_texts
    if add_test_to_vocab:
        texts = texts + datasets['test'].tokenized_texts
    vocab.build(texts=texts)
    for subset in SUBSETS:
        datasets[subset].vectorize_texts(vocab)
    return datasets, vocab


def get_transformer_splits(loader_cls, tokenizer, return_intent_labels=True):
    """
    Get train/dev/test split of the OOD dataset in the form suitable for transformer models.
    
    Args:
        loader_cls: class for loading OOD dataset from raw files.
        tokenizer: tokenizer from the `transformers` library
        return_intent_labels: whether to return intent labels with instances (default: True)

    Returns:
        list of dataset splits in order train/dev/test
    """
    datasets = []
    for subset in SUBSETS:
        dataset = OODDataset(loader_cls(subset=subset), tokenizer.tokenize,
                             return_intent_labels)
        dataset.vectorize_texts(tokenizer)
        datasets.append(dataset)
    return datasets


def get_loader(dataset_name, **kwargs):
    """
    Get loader class for a dataset.

    Args:
        dataset_name: name of the dataset
        **kwargs: additional arguments for specific datasets

    Returns:
        dataset loader class
    """
    if dataset_name == "rostd":
        loader = partial_class(ROSTDLoader,
                               data_root_dir="data/rostd",
                               use_coarse_labels=False)
    elif dataset_name == "rostd_coarse":
        loader = partial_class(ROSTDLoader,
                               data_root_dir="data/rostd",
                               use_coarse_labels=True)
    elif dataset_name == "snips_75":
        loader = partial_class(SNIPSLoader,
                               data_root_dir="data/snips",
                               K=75, version=kwargs['version'])
    elif dataset_name == "snips_25":
        loader = partial_class(SNIPSLoader,
                               data_root_dir="data/snips",
                               K=25, version=kwargs['version'])
    elif dataset_name == "clinc":
        loader = partial_class(CLINC150Loader,
                               data_path="data/clinc/data_full.json",
                               unsupervised=True)
    elif dataset_name == "clinc_sup":
        loader = partial_class(CLINC150Loader,
                               data_path="data/clinc/data_full.json",
                               unsupervised=False)
    elif dataset_name == 'sst':
        loader = partial_class(SSTLoader,
                               data_root_dir="data/sst",
                               ood_type=kwargs['ood_type'])
    else:
        raise RuntimeError(f"Bad dataset: {dataset_name}")
    return loader


def get_dataset_transformers(tokenizer, dataset_name, **kwargs):
    """
    Get OOD dataset splits.

    Args:
        tokenizer: tokenizer from `transformers` library related to a specific transformer
        dataset_name: name of the dataset
        **kwargs: additional arguments for a specific dataset

    Returns:
        train/dev/test split of the OOD dataset
    """
    loader = get_loader(dataset_name, **kwargs)
    return get_transformer_splits(loader, tokenizer)


def get_dataset_simple(dataset_name,
                       add_valid_to_vocab=False,
                       add_test_to_vocab=False,
                       to_lower_case=True,
                       tok_fn=word_tokenize,
                       **kwargs):
    """
    Get OOD dataset splits for an OOD dataset.

    Args:
        dataset_name: name of the OOD dataset
        add_valid_to_vocab: whether to add valid words to the vocabulary
        add_test_to_vocab: whether to add test words to the vocabulary
        to_lower_case: whether to transform text to lowercase
        tok_fn: tokenization function
        **kwargs: additional arguments for ta specific dataset

    Returns:
        (tuple):
            mapping from split names (train/val/test) to Dataset instances
            vocabulary
    """
    loader = get_loader(dataset_name, **kwargs)
    return get_simple_splits(loader,
                             add_valid_to_vocab=add_valid_to_vocab,
                             add_test_to_vocab=add_test_to_vocab,
                             tok_fn=tok_fn,
                             to_lower_case=to_lower_case)


def collate_fn_simple(data, pad_idx: int = 0, bos_idx: int = 1, eos_idx: int = 2):
    """
    Collate function for baseline models: LLR, CNN, LSTM.

    Args:
        data: batch of data to be collated
        pad_idx: padding index
        bos_idx: begin of sequence index
        eos_idx: end of sequence index

    Returns:
        (tuple):
            torch.Tensor: batch of numericalized sentences
            torch.Tensor: batch of numericalized labels
            torch.Tensor: batch of bool values: true if sentence is OOD end false otherwise
    """
    max_len = max(len(datum[0]) + 2 for datum in data)
    labels = torch.zeros(len(data), dtype=torch.long)
    ood_labels = torch.zeros(len(data), dtype=torch.long)
    batch = []
    for idx, (numerical_sent, label, is_ood) in enumerate(data):
        labels[idx] = label
        ood_labels[idx] = is_ood
        batch.append([bos_idx] + numerical_sent + [eos_idx] + [pad_idx]*(max_len - len(numerical_sent) - 2))
    return torch.tensor(batch), labels, ood_labels


def collate_fn_bow(data, vocab_size):
    """
    Collate function for BoW method.

    Args:
        data: batch of data to be collated

    Returns:
        tuple of torch.Tensors: numericalized sentence, labels, ood_labels
    """
    labels = torch.zeros(len(data), dtype=torch.long)
    ood_labels = torch.zeros(len(data), dtype=torch.long)
    rows, cols = [], []
    values = []
    for idx, (numerical_sent, label, is_ood) in enumerate(data):
        labels[idx] = label
        ood_labels[idx] = is_ood
        for num, cnt in zip(*np.unique(numerical_sent, return_counts=True)):
            rows.append(idx)
            cols.append(num)
            values.append(cnt)
    indices = np.vstack((rows, cols))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    batch = torch.sparse.FloatTensor(i, v, torch.Size((len(data), vocab_size)))
    return batch, labels, ood_labels
