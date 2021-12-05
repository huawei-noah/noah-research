# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import json
from pathlib import Path

from pandas import read_csv


class Loader:
    """
    Abstract data loader.
    """
    def __init__(self, *, subset):
        assert subset in ["train", "val", "test"]


class CLINC150Loader(Loader):
    """
    Data loader for CLINC150 dataset.
    """
    def __init__(self, data_path, subset, unsupervised=True):
        """
        Load CLINC150 subset.

        Args:
            data_path: path to the data file
            subset: subset to load (train/val/test)
            unsupervised: whether to use unsupervised version of the dataset (default: True)
        """
        super().__init__(subset=subset)
        with open(data_path) as f:
            data = json.load(f)
        # no ood samples used during training
        if subset == "train":
            if unsupervised:
                self.data_pairs = data["train"], list()
            else:
                raise NotImplementedError("Only unsupervised mode supported")
        elif subset == "val":
            self.data_pairs = data["val"], data["oos_val"]
        else:
            self.data_pairs = data["test"], data["oos_test"]
        self.n_indomain, self.n_ood = len(self.data_pairs[0]), len(self.data_pairs[1])
        self.ood_labels = [0] * self.n_indomain + [1] * self.n_ood
        self.data_pairs = self.data_pairs[0] + self.data_pairs[1]
        self.raw_texts, self.raw_labels = zip(*self.data_pairs)


class ROSTDLoader(Loader):
    """
    Loader for the ROSTD dataset

    Note: data path is root directory for FB ROSTD dataset, the class will perform search
    in it
    """
    def __init__(self, data_root_dir, subset, use_coarse_labels=False):
        """
        Load ROSTD subset data.

        Args:
            data_root_dir: path to the directory with the ROSTD dataset
            subset: subset of data (train/val/test)
            use_coarse_labels: whether to use coarse labels (default: False)
        """
        # TODO: maybe support slots
        super().__init__(subset=subset)
        data_root_dir = Path(data_root_dir)
        assert data_root_dir.exists()
        # no ood samples used during training
        ood_data_path = None
        self.ood_data = []
        self.n_ood = 0
        in_domain_dir = data_root_dir / "en"
        if subset == "train":
            data_path = data_root_dir / "OODRemovedtrain.tsv"
        elif subset == "val":
            data_path = data_root_dir / "eval.tsv"
        else:
            data_path = data_root_dir / "test.tsv"
        self.data = read_csv(data_path, sep="\t", header=None)
        self.raw_texts = self.data[2].tolist()
        self.data[0] = self.data[0].map(lambda label: "oos" if label == "outOfDomain" else label)
        self.raw_labels = self.data[0].tolist()
        if use_coarse_labels:
            self.raw_labels = [label.split("/", 1)[0] for label in self.raw_labels]
        self.ood_labels = [1 if label == "FILLER" else 0 for label in self.data[3]]


class SNIPSLoader(Loader):
    def __init__(self, data_root_dir, subset, K: int = 75, version: int = 0):
        """
        Load SNIPS subset

        :param data_root_dir: path to folder with data
        :param subset: one of train, val, test
        :param K: one of 25, 75 - proportion of in-domain training points
        :param version: one of 0,1,2,3,4 - which random split is used
        """
        super().__init__(subset=subset)
        data_path = Path(data_root_dir) / f"snips_{subset}_{K}_{version}.csv"
        assert data_path.exists()
        data = read_csv(data_path)
        self.in_domain_data = data[data.is_ood == 0]
        self.n_indomain = len(self.in_domain_data)
        self.ood_data = data[data.is_ood == 1]
        self.n_ood = len(self.ood_data)
        self.raw_texts = self.in_domain_data.text.tolist() + self.ood_data.text.tolist()
        self.raw_labels = self.in_domain_data.labels.tolist() + ["oos"] * self.n_ood
        self.ood_labels = [0] * self.n_indomain + [1] * self.n_ood


SST_OOD_SETS = (
    'wmt16', 'multi30k', 'rte', 'snli'
)


SST_OOD_MAPPING = {
    'test': {
        'wmt16': 'wmt16/test.en',
        'multi30k': 'multi30k/test.en',
        'rte': 'rte/test.tsv',
        'snli': 'snli/snli_1.0_test.txt',
    },
    'val': {
        'wmt16': 'wmt16/val.en',
        'multi30k': 'multi30k/val.en',
        'rte': 'rte/dev.tsv',
        'snli': 'snli/snli_1.0_dev.txt',
    }
}


class SSTLoader(Loader):
    """
    Loader for SST OOD setup.
    """
    def __init__(self, data_root_dir, subset, ood_type):
        """
        Load subset of SST OOD setup dataset.

        Args:
            data_root_dir: path to the directory with data
            subset: subset to be loaded (train/val/test)
            ood_type: type of the ood data (None/wmt16/multi30k/rte/snli)
        """
        super().__init__(subset=subset)
        assert ood_type in SST_OOD_SETS
        data_path = Path(data_root_dir) / f"{subset}.tsv"
        assert data_path.exists()
        data = read_csv(data_path, sep='\t', header=None)
        data[1] = data[1].map(lambda label: 'positive' if label == 1 else 'negative')
        ood_data = []
        raw_texts = data[0].tolist()
        if subset in ('val', 'test'):
            ood_data_path = Path(data_root_dir) / SST_OOD_MAPPING[subset][ood_type]
            assert ood_data_path.exists()
            if ood_type in ('rte', 'snli'):
                ood_data = read_csv(ood_data_path, sep='\t')
                ood_data['text'] = ood_data.sentence1 + ' ' + ood_data.sentence2
                raw_texts.extend(ood_data.text.tolist())
            else:
                ood_data = read_csv(ood_data_path, sep='\t', header=None)
                raw_texts.extend(ood_data[0].tolist())
        self.n_ood = len(ood_data)
        self.n_indomain = len(data)
        self.ood_labels = [0] * self.n_indomain + [1] * self.n_ood
        self.raw_labels = data[1].tolist() + ['oos'] * self.n_ood
        self.raw_texts = raw_texts
