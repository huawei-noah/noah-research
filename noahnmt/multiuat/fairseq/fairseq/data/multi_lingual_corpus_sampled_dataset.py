# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections import OrderedDict
from typing import Callable, Dict, List
import random
import numpy as np

from . import FairseqDataset

logger = logging.getLogger(__name__)

def uniform_sampler(x):
    # Sample from uniform distribution
    return np.random.choice(x, 1).item()


class MultilingualCorpusSampledDataset(FairseqDataset):
    """
    Stores multiple instances of FairseqDataset together and in every iteration
    creates a batch by first sampling a dataset according to a specified
    probability distribution and then getting instances from that dataset.

    Args:
        datasets: an OrderedDict of FairseqDataset instances.
        sampling_func: A function for sampling over list of dataset keys.
            The default strategy is to sample uniformly.
    """

    def __init__(
        self,
        cfg,
        datasets: Dict[str, FairseqDataset],
        eval_key=None
    ):
        super().__init__()
        assert isinstance(datasets, OrderedDict)
        self.cfg = cfg
        self.datasets = datasets
        self.eval_key = eval_key

        self.total_num_instances = 0
        self.dataset_sizes = []
        for _, dataset in datasets.items():
            assert isinstance(dataset, FairseqDataset)
            self.total_num_instances += len(dataset)
            self.dataset_sizes.append(len(dataset))

        self._ordered_indices = None
        self.dataset_sizes = np.array(self.dataset_sizes)
        self.p = self.calc_p(t=float(self.cfg.temperature))
        

    def get_sample_with_key(self, key):
        """
        Get some samples with a given key
        """
        dataset = self.datasets[key]
        sample_indices = random.sample(range(len(dataset)), k=len(dataset))
        samples, count = [], 0
        for i in sample_indices:
            samples.append(dataset[i])
            count += dataset.num_tokens(i)
            if count >= self.cfg.max_sample_tokens: break
        
        return OrderedDict([
            (key, self.datasets[key].collater(samples, tag=key))
            ])

    def get_sample_prob(self):
        dic = OrderedDict()
        for x, y, z in list(zip(self.datasets.keys(), self.dataset_sizes, self.p.tolist())):
            dic[x] = (y, z)
            logger.info("Domain: %7s | size = %7d | Sampling probability = %6.3f%% " % (x, y, z*100))
        return dic

    def calc_p(self, t):
        q = self.dataset_sizes / np.sum(self.dataset_sizes)
        p = (q**(1/t)) / (np.sum(q**(1/t)))
        return p

    def sampling_func(self, x):
        # Sample from uniform distribution
        return np.random.choice(x, 1, p=self.p).item()
    
    def update_sampling_p(self, p):
        logger.info("******* Previous Sampling Probability *******")
        self.get_sample_prob()
        self.p = p
        logger.info("******* Current Sampling Probability *******")
        self.get_sample_prob()

    def collater(self, samples: List[Dict]):
        """
        Generate a mini-batch for this dataset.
        To convert this into a regular mini-batch we use the following
        logic:
            1. Select a dataset using the specified probability distribution.
            2. Call the collater function of the selected dataset.
        """
        if self.eval_key is None:
            if len(samples) == 0:
                return None

            selected_key = self.sampling_func(list(self.datasets.keys()))
            selected_samples = [sample[selected_key] for sample in samples]
            return self.datasets[selected_key].collater(selected_samples, tag=selected_key)
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key].collater(samples)

    def __len__(self):
        """
        Length of this dataset is the sum of individual datasets
        """
        return self.total_num_instances

    def ordered_indices(self):
        """
        Ordered indices for batching. Here we call the underlying
        dataset's ordered_indices() so that we get the same random ordering
        as we would have from using the underlying dataset directly.
        """
        if self._ordered_indices is None:
            self._ordered_indices = OrderedDict(
                [
                    (key, dataset.ordered_indices())
                    for key, dataset in self.datasets.items()
                ]
            )


        return np.arange(len(self))

    def _map_index_to_dataset(self, key: int, index: int):
        """
        Different underlying datasets have different lengths. In order to ensure
        we are not accessing an index outside the range of the current dataset
        size, we wrap around. This function should be called after we have
        created an ordering for this and all underlying datasets.
        """
        assert (
            self._ordered_indices is not None
        ), "Must call MultiCorpusSampledDataset.ordered_indices() first"
        mapped_index = index % len(self.datasets[key])
        return self._ordered_indices[key][mapped_index]

    def __getitem__(self, index: int):
        """
        Get the item associated with index from each underlying dataset.
        Since index is in the range of [0, TotalNumInstances], we need to
        map the index to the dataset before retrieving the item.
        """
        if self.eval_key is None:
            return OrderedDict(
                [
                    (key, dataset[self._map_index_to_dataset(key, index)])
                    for key, dataset in self.datasets.items()
                ]
            )
        else:
            # at evaluation time it's useful to pass-through batches from a single key
            return self.datasets[self.eval_key][self._map_index(self.eval_key, index)]


    def num_tokens(self, index: int):
        """
        Return an example's length (number of tokens), used for batching. Here
        we return the max across all examples at index across all underlying
        datasets.
        """
        return max(
            dataset.num_tokens(self._map_index_to_dataset(key, index))
            for key, dataset in self.datasets.items()
        )

    def size(self, index: int):
        """
        Return an example's size as a float or tuple. Here we return the max
        across all underlying datasets. This value is used when filtering a
        dataset with max-positions.
        """
        # return max(
        #     dataset.size(self._map_index_to_dataset(key, index))
        #     for key, dataset in self.datasets.items()
        # )
        return OrderedDict(
                [
                    (key, dataset.size(self._map_index_to_dataset(key, index)))
                    for key, dataset in self.datasets.items()
                ]
            )

    @property
    def supports_prefetch(self):
        return all(
            getattr(dataset, "supports_prefetch", False)
            for dataset in self.datasets.values()
        )

    def prefetch(self, indices):
        for key, dataset in self.datasets.items():
            dataset.prefetch(
                [self._map_index_to_dataset(key, index) for index in indices]
            )

    @property
    def supports_fetch_outside_dataloader(self):
        return all(
            self.datasets[key].supports_fetch_outside_dataloader
            for key in self.datasets
        )
