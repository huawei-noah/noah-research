"""Base surrogate model."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import mindspore as ms
from mindspore import Tensor
from abc import ABC, abstractmethod

import hebo.mindspore as hebo_ms


class BaseModel(ABC):
    """Base class for probabilistic regression models."""
    
    support_ts = False
    support_grad = False
    support_multi_output = False
    support_warm_start = False
    
    def __init__(self,
                 num_cont: int,
                 num_enum: int,
                 num_out: int,
                 **conf):
        """conf: configuration dict."""
        self.num_cont = num_cont
        self.num_enum = num_enum
        self.num_out = num_out
        self.conf = conf
        assert (self.num_cont >= 0)
        assert (self.num_enum >= 0)
        assert (self.num_out > 0)
        assert (self.num_cont + self.num_enum > 0)
        if self.num_enum > 0:
            assert 'num_uniqs' in self.conf
            assert isinstance(self.conf['num_uniqs'], type([]))
            assert len(self.conf['num_uniqs']) == self.num_enum
        if not self.support_multi_output:
            assert self.num_out == 1, "Model only support single-output"
    
    @abstractmethod
    def fit(self,
            Xc: Tensor,
            Xe: Tensor,
            y: Tensor):
        """Fit surrogate model."""
        pass
    
    @abstractmethod
    def predict(self,
                Xc: Tensor,
                Xe: Tensor) -> (Tensor, Tensor):
        """Return (possibly approximated) Gaussian predictive distribution.

        Return py and ps2 where py is the mean and ps2 predictive variance.
        """
        pass
    
    @property
    def noise(self) -> Tensor:
        """Return estimated noise variance.

        For example, GP can view noise level as a hyperparameter and optimize it via MLE, another strategy could be
        using the MSE of training data as noise estimation.

        Should return a (self.n_out, ) float tensor.
        """
        return hebo_ms.zeros(self.num_out)
    
    def sample_f(self):
        """Thompson Sampling."""
        # Thompson sampling
        raise NotImplementedError("Thompson sampling is not supported")
    
    def sample_y(self, Xc: Tensor, Xe: Tensor, n_samples: int) -> Tensor:
        """Sample y's."""
        py, ps2 = self.predict(Xc, Xe)
        ps = hebo_ms.sqrt(ps2)
        norm_data = hebo_ms.randn(n_samples, py.shape[0], self.num_out)
        return py + ps * norm_data
