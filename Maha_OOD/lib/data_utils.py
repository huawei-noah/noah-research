# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from collections import Counter
from functools import partialmethod
from typing import List, Optional


class Vocab:
    def __init__(self, max_tokens: Optional[int] = None):
        self.word2id = None
        self.id2word = None
        self.word_counts = None
        # special token idxs
        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3
        self.sep_idx = 4
        self.special_token_idxs = [0, 1, 2, 3, 4]
        self.max_tokens = max_tokens

    def build(self, texts: List[List[str]]):
        vocab = Counter()
        for text in texts:
            assert isinstance(texts, list)
            vocab.update(text)
        count_pairs = sorted(vocab.items(), key=lambda x: (-x[1], x[0]))
        if self.max_tokens:
            count_pairs = count_pairs[:self.max_tokens]
        words, counts = zip(*count_pairs)
        words, counts = list(words), list(counts)
        words = ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[SEP]"] + words
        self.word_counts = [0, 0, 0, 0, 0] + counts
        self.word2id = {word: i for i, word in enumerate(words)}
        self.id2word = words

    def decode(self, indices):
        def decode_sentence(sent_indices):
            try:
                idx = sent_indices.index(self.eos_idx)
                tokens = [self.id2word[idx] for idx in sent_indices[:idx]]
            except KeyError:
                tokens = [self.id2word[idx] for idx in sent_indices]
            return " ".join(tokens)

        if not indices:
            return ""
        if isinstance(indices[0], List):
            return "\n".join(decode_sentence(indices[i]) for i in range(len(indices)))
        return decode_sentence(indices)

    def encode(self, tokens):
        return [self.word2id.get(t, self.unk_idx) for t in tokens]

    def __len__(self):
        return len(self.word2id)

    def __iter__(self):
        return iter(self.id2word)

    def __getitem__(self, item):
        return self.word2id[item]


def partial_class(cls, *args, **kwargs):
    class PartClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return PartClass
