# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from typing import Any, Optional
import torch
import itertools
import numpy as np


def create_attn_masks_and_pos(lengths, prefix_token_idxs):
    # 1. locate [EOT] token for the current instance
    # 2. attention_mask = causal_attention_mask, so replicate this as a BxNxN matrix to be given to the model
    b = len(lengths)
    n = max([sum(l) for l in lengths])
    attention_masks = torch.zeros((b, n, n))
    position_ids = -100 * torch.ones((b, n)).long()
    prefix_mask = torch.zeros((b, n, n)).long()

    for i, l in enumerate(lengths):  # for each new instance
        csum_lengths = np.cumsum(l)
        prefixes = prefix_token_idxs[i]

        for j in range(len(csum_lengths)):
            cur, prev = csum_lengths[j], csum_lengths[j - 1]
            if j == 0:
                attention_masks[i, :cur, :cur] = torch.tril(torch.ones(cur, cur))
                prefix_mask[i, :prefixes[j]+1, :prefixes[j]+1] = 1

                if len(csum_lengths) == 1:
                    position_ids[i, :n] = torch.arange(0, n)  # single element
                else:
                    position_ids[i, :cur] = torch.arange(0, cur)

            else:
                attention_masks[i, prev:cur, prev:cur] = torch.tril(torch.ones(cur - prev, cur - prev))
                prefix_mask[i, prev:prev + prefixes[j]+1, prev:prev + prefixes[j]+1] = 1

                # if last element in the batch, fill position ids until the end
                if j == len(csum_lengths) - 1:
                    position_ids[i, prev:n] = torch.arange(0, n - prev)
                else:
                    position_ids[i, prev:cur] = torch.arange(0, cur - prev)

    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # import pandas as pd
    # fig = plt.figure(figsize=(20,15))
    # sns.heatmap(prefix_mask[0].numpy(), vmin=0.0, vmax=1.0, linewidth=0.5, cmap=sns.color_palette("rocket_r", as_cmap=True))
    # fig.tight_layout()
    # fig.savefig('tmp.png', bbox_inches='tight', dpi=600)
    #
    # fig = plt.figure(figsize=(20, 15))
    # sns.heatmap(attention_masks[0].numpy(), vmin=0.0, vmax=1.0, linewidth=0.5,
    #             cmap=sns.color_palette("rocket_r", as_cmap=True))
    # fig.tight_layout()
    # fig.savefig('tmp2.png', bbox_inches='tight', dpi=600)
    #
    # exit(0)
    assert torch.all(position_ids != -100), f"Careful! Position IDs contain -100!\n{position_ids}"
    return attention_masks, position_ids, prefix_mask


class DataCollatorWithPaddingForCorruptCLM:
    """
    Data collator used for causal language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    """
    def __init__(self, tokenizer=None, predict_code=False, prefix_lm=False, code_mask=False):
        self.mlm_probability = 0.15
        self.tokenizer = tokenizer
        self.predict_code = predict_code
        self.prefix_lm = prefix_lm
        self.code_mask = code_mask

        self.unk_token = '<|unkoftext|>' if '<|unkoftoken|>' in self.tokenizer.all_special_tokens else '<unk>'
        self.pad_token = '<|padoftext|>' if '<|padoftext|>' in self.tokenizer.all_special_tokens else '<pad>'

    def mask_tokens(self, inputs: Any, special_tokens_mask: Optional[Any] = None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids('<mask>')

        # 20% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 1.0)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(high=len(self.tokenizer),
                                     size=labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels

    def __call__(self, examples):

        docstr_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['docstr_mask']) for e in examples],
            batch_first=True,
            padding_value=0
        ).bool()
        code_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['code_mask']) for e in examples],
            batch_first=True,
            padding_value=0
        ).bool()

        # Inputs for MLM
        mlm_inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['input_ids']) for e in examples],
            batch_first=True,
            padding_value=self.tokenizer.convert_tokens_to_ids(self.pad_token)
        )
        special_tokens_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['special_tokens_mask']).bool() for e in examples],
            batch_first=True,
            padding_value=1
        )
        special_tokens_mask = torch.where(code_mask, 1, special_tokens_mask.long())

        mlm_inputs, _ = self.mask_tokens(
            mlm_inputs, special_tokens_mask=special_tokens_mask
        )

        clm_inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['input_ids']) for e in examples],
            batch_first=True,
            padding_value=self.tokenizer.convert_tokens_to_ids(self.unk_token)
        ).long()

        # Give corrupted docstring to the input
        inputs = torch.where(docstr_mask, mlm_inputs, clm_inputs)

        attention_masks, position_ids, prefix_mask = create_attn_masks_and_pos(
            [e['length'] for e in examples],
            [e['prefix_lm_token_idx'] for e in examples]
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['input_ids']) for e in examples],
            batch_first=True,
            padding_value=-100
        )
        if self.predict_code:
            # -100 to all places with zeros (i.e. non-code tokens)
            labels[docstr_mask] = -100

        output_dict = {
            'input_ids': inputs,
            'labels': labels,
            "attention_mask": attention_masks,
            "position_ids": position_ids,
            'docstr_mask': docstr_mask
        }

        if self.code_mask:
            output_dict.update({"code_mask": code_mask})

        if self.prefix_lm:
            output_dict.update({"prefix_lm_mask": prefix_mask})
        return dict(output_dict)


class DataCollatorWithPaddingForCLM:
    """
    Data collator used for causal language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    """
    def __init__(self, predict_code=False, tokenizer=None, prefix_lm=False, code_mask=False):
        self.predict_code = predict_code
        self.tokenizer = tokenizer
        self.prefix_lm = prefix_lm
        self.code_mask = code_mask

        self.eot_token = 'eot'
        self.unk_token = '<|unkoftext|>' if '<|unkoftoken|>' in self.tokenizer.all_special_tokens else '<unk>'
        self.pad_token = '<|padoftext|>' if '<|padoftext|>' in self.tokenizer.all_special_tokens else '<pad>'
        self.eot_token_id = self.tokenizer.convert_tokens_to_ids(self.eot_token)
        self.unk_token_id = self.tokenizer.convert_tokens_to_ids(self.unk_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.pad_token)

    def __call__(self, examples):

        inputs = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['input_ids']).long() for e in examples],
            batch_first=True,
            padding_value=self.unk_token_id
        )

        attention_masks, position_ids, prefix_mask = create_attn_masks_and_pos(
            [e['length'] for e in examples],
            [e['prefix_lm_token_idx'] for e in examples]
        )

        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['input_ids']).long() for e in examples],
             batch_first=True,
             padding_value=-100
        )

        code_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['code_mask']) for e in examples],
            batch_first=True,
            padding_value=0
        ).bool()

        docstr_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(e['docstr_mask']) for e in examples],
            batch_first=True,
            padding_value=0
        ).bool()

        if self.predict_code:
            # -100 to all places with zeros (i.e. non-code tokens)
            labels[docstr_mask] = -100

        output_dict = {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_masks,
            "position_ids": position_ids,
            "docstr_mask": docstr_mask
        }

        if self.code_mask:
            output_dict.update({"code_mask": code_mask})

        if self.prefix_lm:
            output_dict.update({"prefix_lm_mask": prefix_mask.bool()})

        return dict(output_dict)
