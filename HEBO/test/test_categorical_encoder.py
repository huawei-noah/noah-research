# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from hebo.models.layers import OneHotTransform, EmbTransform
import hebo.mindspore as hebo_ms
import pytest
from mindspore import Tensor
import mindspore.nn as nn
import mindspore as ms
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')


def test_encoders():
    layer1 = EmbTransform([5, 5], emb_sizes=[1, 1])
    layer2 = EmbTransform([5, 5], emb_sizes=[2, 2])
    layer3 = OneHotTransform([5, 5])
    layer4 = EmbTransform([5, 5])
    
    xe = Tensor(np.random.randint(0, 5, (10, 2)))
    assert (layer1(xe).shape[1] == 2)
    assert (layer1.num_out == 2)
    
    assert (layer2(xe).shape[1] == 4)
    assert (layer2.num_out == 4)
    
    assert (layer4(xe).shape[1] == layer4.num_out)
    
    assert (layer3(xe).shape[1] == 10)
    assert (layer3.num_out == 10)
    
    or_op = ms.ops.LogicalOr()
    assert or_op(layer3(xe) == 0, layer3(xe) == 1).all()
    assert (hebo_ms.sum(layer3(xe), axis=1) == xe.shape[1]).all()
    
    model1 = nn.SequentialCell([
        OneHotTransform([5, 5]),
        nn.Dense(10, 1)
    ])
    model2 = nn.SequentialCell([
        EmbTransform([5, 5], emb_sizes=[2, 2]),
        nn.Dense(4, 1)
    ])
    assert (model1(xe).shape == model2(xe).shape)
