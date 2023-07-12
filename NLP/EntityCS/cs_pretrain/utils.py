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
import argparse
import numpy as np

logger = logging.getLogger(__name__)


def str2bool(i):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(i, bool):
        return i
    if i.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif i.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def humanized_time(second):
    """
    Print time in hours:minutes:seconds
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def get_sampling_probability_from_counts(datasets_counts_list):
    # following: https://papers.nips.cc/paper/8928-cross-lingual-language-model-pretraining.pdf (Section 3.1)
    alpha = 0.5

    count = np.array(datasets_counts_list)
    initial_probs = count / np.sum(count)

    # create an output vector where the probability of sampling an example from a dataset is up-sampled
    # for the low-resource language but not too much.
    new_probs = initial_probs**alpha

    final_weights_per_example = new_probs / np.sum(new_probs)
    final_weights_per_dataset = new_probs / np.sum(new_probs)

    return final_weights_per_example.tolist(), final_weights_per_dataset.tolist()


def mean_pool_embedding(all_layer_outputs, masks):
    """
    Args:
      all_layer_outputs: list of torch.FloatTensor, (B, L, D)
      masks: torch.FloatTensor, (B, L)
    Return:
      sent_emb: list of torch.FloatTensor, (B, D)
    """
    sent_embeds = []
    for embeds in all_layer_outputs:
        embeds = (embeds * masks.unsqueeze(2).float()).sum(dim=1) / masks.sum(
            dim=1
        ).view(-1, 1).float()
        sent_embeds.append(embeds)
    return sent_embeds[-1]  # return last one for now
