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
import random
from torch.utils.data import Sampler
from utils import get_sampling_probability_from_counts
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BatchLanguageSampler(Sampler):
    """
    Custom Sampler to get samples based on sharding
    """

    def __init__(self, batch_size, train_lang_nums, langs, lang_idx):
        # lang_idx is a dictionary of (start, end) for each lang {'fr': (0, 300), 'de': (600, 700)}
        self.batch_size = batch_size
        self.langs = langs
        self.batches = []
        self.total_num_batches = sum(train_lang_nums) // batch_size + 1
        self.lang_idx = lang_idx
        _, self.sampling_probs_mlm = get_sampling_probability_from_counts(
            train_lang_nums
        )
        self.drop_last = False
        self.create_batches()

    def create_batches(self):
        # keep track of which idx a language should start sampling from
        cur_idx = {k: v[0] for k, v in self.lang_idx.items()}

        for _ in tqdm(range(self.total_num_batches), disable=True):
            # random choose a lang that we sample a batch from
            lang = random.choices(self.langs, weights=self.sampling_probs_mlm)[0]

            start_idx, end_idx = self.lang_idx[lang]
            if cur_idx[lang] + self.batch_size - 1 <= end_idx:
                # enough data for lang to sample
                self.batches.append(
                    list(range(cur_idx[lang], cur_idx[lang] + self.batch_size))
                )
                if cur_idx[lang] + self.batch_size - 1 == end_idx:
                    # next time needs to start from the start_idx of this lang
                    cur_idx[lang] = start_idx
                else:
                    cur_idx[lang] += self.batch_size
            else:
                # not enough data to sample, we need to take the remaining and then take the rest from the beginning
                num_samples_front = self.batch_size - (end_idx - cur_idx[lang] + 1)
                self.batches.append(
                    list(range(cur_idx[lang], end_idx + 1))
                    + list(range(start_idx, start_idx + num_samples_front))
                )
                cur_idx[lang] = num_samples_front + start_idx

    def __iter__(self):
        return iter(self.batches[i] for i in range(len(self.batches)))

    def __len__(self):
        return len(self.batches)
