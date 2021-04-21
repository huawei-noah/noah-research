"""Module for all the layers."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor, ops


class EmbTransform(nn.Cell):
    """Embedding Transform."""

    def __init__(self, num_uniqs, **conf):
        super().__init__()
        self.emb_sizes = conf.get('emb_sizes')
        if self.emb_sizes is None:
            self.emb_sizes = [min(50, 1 + v // 2) for v in num_uniqs]

        self.emb = nn.CellList()
        for num_uniq, emb_size in zip(num_uniqs, self.emb_sizes):
            self.emb.append(nn.Embedding(num_uniq, emb_size))

        self.cat = ops.Concat(axis=1)

    @property
    def num_out(self) -> int:
        """Return number of embeddings."""
        return sum(self.emb_sizes)

    def construct(self, xe):
        """Construct embedding matrix from variables."""
        return self.cat([self.emb[i](xe.astype(ms.int32)[:, i]).view(
            xe.shape[0], -1) for i in range(len(self.emb))])


class OneHotTransform(nn.Cell):
    """One hot encoding transformation."""

    def __init__(self, num_uniqs):
        super().__init__()
        self.num_uniqs = num_uniqs
        self.one_hot_op = ops.OneHot()
        self.cat = ops.Concat(axis=1)

    def one_hot(self, x, num_uniq: int):
        """One hot encoding transform of variables x."""
        return self.one_hot_op(
            x, num_uniq, Tensor(
                1.0, ms.float32), Tensor(
                0., ms.float32))

    @property
    def num_out(self) -> int:
        """Return number of unique values."""
        return sum(self.num_uniqs)

    def construct(self, xe):
        """Encode variables xe."""
        return self.cat([self.one_hot(xe.astype(ms.int32)[:, i],
                        self.num_uniqs[i]) for i in range(xe.shape[1])])
