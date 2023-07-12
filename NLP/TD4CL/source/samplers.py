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

from torch.utils.data import Sampler
from utils import *
import math
from scipy.stats import norm


logger = logging.getLogger(__name__)


class BatchPacingSampler(Sampler):
    """
    Custom Sampler to get samples based on an external pacing function
    """
    def __init__(self, data, config, metric1=None, metric2=None, batch_size=None, c0=0.01,
                 total_steps=None, sample=False, selection=None):
        # collect metric1 and metric2
        self.data = data
        self.batch_size = batch_size

        scores = np.asarray([d[metric1] for d in data])
        if (metric2 is not None) and (metric2 != ''):
            self.second_scores = np.asarray([d[metric2] for d in data])
        self.percentage = 1.0 if selection is None else selection

        logger.info('Sorting data ...')
        if metric1 in ['correctness', 'confidence']:
            logger.info('Sorting from high to low ...')
            indices = np.argsort(np.negative(scores))  # from high to low
            print(scores[indices[0]], scores[indices[-1]])
        else:
            logger.info('Sorting from low to high ...')
            indices = np.argsort(scores)
            sorted_scores = np.sort(scores)
            print(scores[indices[0]], scores[indices[-1]])

        # Form batches
        logger.info('Forming batches ...')

        total_steps = total_steps * config['gradient_accumulation_steps']
        self.batches = [[] for _ in range(total_steps)]

        for train_step in range(0, total_steps):
            current_competence = self.pacing_root(train_step, c0, total_steps)

            fraction = int(current_competence * len(data))
            selected = indices[:fraction+1]

            if sample == 'bias':
                weights = self.second_scores[selected]

                if len(weights) < batch_size:
                    take = torch.multinomial(torch.from_numpy(weights), len(weights), False).numpy()
                else:
                    take = torch.multinomial(torch.from_numpy(weights), batch_size, False).numpy()
                self.batches[train_step] = selected[take].tolist()

            elif sample == 'most':
                weights = self.second_scores[selected]
                take = np.argsort(weights)[-batch_size:]
                self.batches[train_step] = selected[take].tolist()

            else:
                if len(selected.tolist()) < batch_size:
                    self.batches[train_step] = random.sample(selected.tolist(), k=len(selected.tolist()))
                else:
                    self.batches[train_step] = random.sample(selected.tolist(), k=batch_size)

        if selection:
            self.data_selection()

    @staticmethod
    def pacing_root(step, c0, total_steps, root=2):
        """
        Root function
        Args:
            step: Current step
            c0: Initial portion of data, by default equal to 1%
            total_steps: Total number of steps
            root: Square root, Cubic root, etc

        Returns (float): Portion of data to use
        """
        return min(1, ((step * ((1 - (c0 ** root)) / total_steps)) + (c0 ** root)) ** (1/root))

    def data_selection(self):
        """
        Select a portion of data based on the variability metric
        """
        select = math.ceil(self.percentage * len(self.data))
        keep_examples = np.argsort(self.second_scores)[-select:]  # take the most ambiguous
        sections = math.ceil(len(keep_examples) / self.batch_size)

        # Do 10 passes over this smaller dataset
        for _ in range(0, 10):
            ids_select = [keep_examples[i] for i in torch.randperm(len(keep_examples))]  # permute/shuffle
            lists = [a.tolist() for a in np.array_split(ids_select, sections)]
            logger.info(
                f'Remaining examples to train: {len(keep_examples)} samples --> {len(lists)} batches formed')
            self.batches.extend(lists)

    def __iter__(self):
        return iter(self.batches[i] for i in range(len(self.batches)))

    def __len__(self):
        return len(self.batches)


class BatchShardingSampler(Sampler):
    """
    Custom Sampler to get samples based on sharding
    """
    def __init__(self, data, metric1=None, metric2=None, curric=None, batch_size=None, sample=False, selection=None):
        self.data = data
        self.batch_size = batch_size
        self.percentage = 1.0 if selection is None else selection
        self.batches = []

        # sort based on metric
        logger.info('Sorting data ...')
        self.scores = np.asarray([d[metric1] for d in data])

        if metric2 is not None:
            self.second_scores = np.asarray([d[metric2] for d in data])

        if metric1 == 'correctness':  # Sort from high to low
            self.rev = True
            logger.info('Sorting from high to low ...')
        else:
            self.rev = False
            logger.info('Sorting from low to high ...')

        # Form batches (train one shard for 1 epoch)
        logger.info('Forming batches ...')
        if 'one-pass' in curric:
            self.one_step()
        elif 'baby-step' in curric:
            self.baby_step_cumsum()
        elif 'annealing' in curric:
            self.baby_step_annealing(sample=sample)
        else:
            print('Wrong curriculum')
            exit(0)

        if selection:
            self.data_selection()

    def extend_batches(self, shard_id, available_ins):
        """
        Using the available examples: shuffle them and split them into batches

        Args:
            shard_id (int): Shard ID
            available_ins (list): Indices we can use (indexes)

        Returns: Extended batch list
        """
        sections = math.ceil(len(available_ins) / self.batch_size)
        ids_select = [available_ins[i] for i in torch.randperm(len(available_ins))]  # permute/shuffle
        lists = [a.tolist() for a in np.array_split(ids_select, sections)]

        logger.info(f'Shard {shard_id} contains {len(available_ins)} samples --> {len(lists)} batches formed')
        self.batches.extend(lists)

    def one_step(self):
        """
        One-pass: Use only current shard in the current step
        """
        unique_shards = list(set(self.scores))
        for num, i in enumerate(sorted(unique_shards, reverse=self.rev)):
            idx = np.where(self.scores == i)[0].tolist()  # elements with correctness == i

            self.extend_batches(num, idx)

    def baby_step_cumsum(self):
        """
        Cumulative Step: Accumulate all previous shard + current in the current phase
        """
        unique_shards = list(set(self.scores))
        valid_samples = []

        for num, i in enumerate(sorted(unique_shards, reverse=self.rev)):
            idx = np.where(self.scores == i)[0].tolist()  # current shard

            valid_samples.extend(idx)
            self.extend_batches(num, valid_samples)

    def baby_step_annealing(self, sample=False):
        """
        Annealing: Select 1/N from each previous shard in the current phase
        """
        unique_shards = list(set(self.scores))
        seen_shards = []  # list of lists
        for num, i in enumerate(sorted(unique_shards, reverse=self.rev)):
            valid_samples = []
            idx = np.where(self.scores == i)[0].tolist()
            seen_shards.append(idx)

            if num == 0:  # 1st shard
                valid_samples.extend(idx)
            else:
                valid_samples.extend(idx)
                if sample == 'bias':
                    for k, shard in enumerate(seen_shards[:-1]):
                        select = (math.ceil(len(shard) / len(unique_shards)))
                        vals = torch.multinomial(torch.from_numpy(self.second_scores[shard]), select, False).numpy()
                        valid_samples.extend((np.array(shard)[vals]).tolist())

                elif sample == 'most':
                    for k, shard in enumerate(seen_shards[:-1]):
                        select = (math.ceil(len(shard) / len(unique_shards)))
                        vals = np.argsort(self.second_scores[shard])[-select:]
                        valid_samples.extend((np.array(shard)[vals]).tolist())

                else:
                    for k, shard in enumerate(seen_shards[:-1]):
                        select = (math.ceil(len(shard) / len(unique_shards)))
                        rand_select = random.sample(shard, k=select)
                        valid_samples.extend(rand_select)

            self.extend_batches(num, valid_samples)

    def baby_step_cumsum_random(self):
        """
        Same as Baby Step but put random examples inside each shard (sanity check)
        """
        unique_shards = list(set(self.scores))
        shard_size = []
        for num, i in enumerate(sorted(unique_shards, reverse=self.rev)):
            idx = np.where(self.scores == i)[0].tolist()  # current shard
            shard_size.append(len(idx))

        indices = np.arange(len(self.scores))
        indices = np.random.permutation(indices)
        indices = np.split(indices, np.cumsum(shard_size)[:-1])

        valid_samples = []
        for num, ind in enumerate(indices):
            valid_samples.extend(ind.tolist())
            self.extend_batches(num, valid_samples)

    def data_selection(self):
        """
        Select a portion of data based on the variability metric
        """
        select = math.ceil(self.percentage * len(self.data))
        keep_examples = np.argsort(self.second_scores)[-select:]  # take the most ambiguous
        sections = math.ceil(len(keep_examples) / self.batch_size)

        # Do 10 passes over this smaller dataset
        for _ in range(0, 10):
            ids_select = [keep_examples[i] for i in torch.randperm(len(keep_examples))]  # permute/shuffle
            lists = [a.tolist() for a in np.array_split(ids_select, sections)]
            logger.info(
                f'Remaining examples to train: {len(keep_examples)} samples --> {len(lists)} batches formed')
            self.batches.extend(lists)

    def __iter__(self):
        return iter(self.batches[i] for i in range(len(self.batches)))

    def __len__(self):
        return len(self.batches)
