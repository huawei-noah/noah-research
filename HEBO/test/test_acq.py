# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from hebo.acquisitions.acq import Tensor, Mean, Sigma, LCB, SingleObjectiveAcq, MOMeanSigmaLCB, MACE, GeneralAcq
from hebo.models.rf.rf import RF
import numpy as np
import pytest
import mindspore as ms
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')

X = Tensor(np.random.randn(10, 1), ms.float32)
y = X
model = RF(1, 0, 1)
model.fit(X, None, y)


@pytest.mark.parametrize('acq_cls',
                         [Mean, Sigma, LCB])
def test_acq(acq_cls):
    acq = acq_cls(model, best_y=0.)
    acq_v = acq(X, None)
    if isinstance(acq, SingleObjectiveAcq):
        assert acq.num_obj == 1
        assert acq.num_constr == 0
    assert isinstance(acq_v, Tensor)
    assert acq_v.shape[1] == acq.num_obj + acq.num_constr


def test_mo_acq():
    acq = MOMeanSigmaLCB(model, best_y=0.)
    acq_v = acq(X, None)
    assert isinstance(acq_v, Tensor)
    assert acq.num_obj == 2
    assert acq.num_constr == 1


def test_mace():
    acq = MACE(model, best_y=0.)
    acq_v = acq(X, None)
    assert isinstance(acq_v, Tensor)
    assert acq.num_obj == 3
    assert acq.num_constr == 0


def test_general():
    acq = GeneralAcq(model, 1, 0)
    acq_v = acq(X, None)
    assert isinstance(acq_v, Tensor)
    assert acq.num_obj == 1
    assert acq.num_constr == 0
