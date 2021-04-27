# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')
import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import hebo.mindspore as hebo_ms
from hebo.models.scalers import MindSporeStandardScaler, MindSporeMinMaxScaler, MindSporeIdentityScaler

@pytest.mark.parametrize('scaler',
        [MindSporeIdentityScaler(), MindSporeStandardScaler(), MindSporeMinMaxScaler((-3, 7))],
        ids = ['Idendity', 'Standard', 'MinMax']
        )
def test_identity(scaler):
    x      = hebo_ms.randn(10, 3) * 5 + 4
    scaler = scaler.fit(x)
    xx     = scaler.inverse_transform(scaler.transform(x))
    res    = (x - xx).abs()
    assert hebo_ms.sum(res) < 1e-4

@pytest.mark.parametrize('scaler',
        [MindSporeIdentityScaler(), MindSporeStandardScaler(), MindSporeMinMaxScaler((-3, 7))],
        ids = ['Idendity', 'Standard', 'MinMax']
        )
def test_identity_same(scaler):
    x      = hebo_ms.ones((10, 1)) * 5 + 4
    scaler = scaler.fit(x)
    xx     = scaler.inverse_transform(scaler.transform(x))
    res    = (x - xx).abs()
    assert(hebo_ms.sum(res) < 1e-4)

    xrand  = hebo_ms.randn(10, 3)
    res    = (xrand - scaler.inverse_transform(scaler.transform(xrand))).abs()
    assert(hebo_ms.sum(res) < 1e-4)

def test_standard_scaler():
    x      = hebo_ms.randn(10, 3) * 5 + 4
    scaler = MindSporeStandardScaler().fit(x)
    x_tr   = scaler.transform(x)
    assert((x_tr.asnumpy().mean(axis = 0) - 0).sum() < 1e-6)
    assert((x_tr.asnumpy().std(axis = 0)  - 1).sum() < 1e-6)

@pytest.mark.parametrize('scaler',
        [MindSporeStandardScaler(), MindSporeMinMaxScaler((-3, 7))],
        ids = ['Standard', 'MinMax']
        )
def test_invalid(scaler):
    x      = np.random.randn(10, 3) * 5 + 4
    x[0,0] = np.nan
    x      = ms.Tensor(x, ms.float32)
    scaler.fit(x)
    x_tr   = scaler.transform(x)
    if isinstance(scaler, MindSporeStandardScaler):
        assert((x_tr[:, 1:].asnumpy().mean(axis = 0) - 0).sum() < 1e-6)
        assert((x_tr[:, 1:].asnumpy().std(axis = 0)  - 1).sum() < 1e-6)

def test_min_max_scaler():
    x      = hebo_ms.randn(10, 3) * 5 + 4
    scaler = MindSporeMinMaxScaler().fit(x)
    x_tr   = scaler.transform(x)
    assert((x_tr.asnumpy().min(axis = 0) - scaler.range_lb).sum() < 1e-6)
    assert((x_tr.asnumpy().max(axis = 0) - scaler.range_ub).sum() < 1e-6)

@pytest.mark.parametrize('scaler',
        [MindSporeIdentityScaler(), MindSporeStandardScaler(), MindSporeMinMaxScaler((-3, 7))],
        ids = ['Idendity', 'Standard', 'MinMax']
        )
def test_one_sample(scaler):
    x      = hebo_ms.zeros((1, 1))
    scaler = scaler.fit(x)
    xx     = scaler.inverse_transform(scaler.transform(x))
    res    = (x - xx).abs()
    assert(hebo_ms.sum(res) < 1e-4)
