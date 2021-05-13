# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import hebo.mindspore as hebo_ms
import mindspore as ms
import numpy as np
from hebo.acquisitions.acq import GeneralAcq
from mindspore import Tensor


class DummyModel:
    def __init__(self, num_obj, num_constr):
        self.num_obj = num_obj
        self.num_constr = num_constr
        self.num_out = num_obj + num_constr
    
    @property
    def noise(self) -> Tensor:
        return hebo_ms.zeros((self.num_out))
    
    def predict(self, x, xe):
        num_data = x.shape[0]
        py = hebo_ms.zeros((num_data, self.num_out))
        ps2 = hebo_ms.ones((num_data, self.num_out))
        return py, ps2


def test_dummy():
    model = DummyModel(3, 4)
    acq = GeneralAcq(
        model,
        model.num_obj,
        model.num_constr,
        kappa=3,
        c_kappa=4,
        use_noise=True)
    x = hebo_ms.randn(10, 3)
    xe = hebo_ms.ones((10, 3)).astype(ms.int32)
    acq_v = acq(x, xe).asnumpy()
    assert np.abs(acq_v[:, :acq.num_obj] + acq.kappa * np.ones((10, 3))).max() < 1e-6
    assert np.abs(acq_v[:, acq.num_obj:] + acq.c_kappa * np.ones((10, 4))).max() < 1e-6
