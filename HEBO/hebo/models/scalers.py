"""mindspore scaling methods."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class MindSporeIdentityScaler(nn.Cell):
    """Identity Scalar."""
    
    def __init__(self):
        super().__init__()
    
    def fit(self, x: Tensor):
        """Fit scalar."""
        return self
    
    def construct(self, x: Tensor) -> Tensor:
        """Construct scaler on x."""
        return x
    
    def transform(self, x: Tensor) -> Tensor:
        """Transform input x."""
        return self.construct(x)
    
    def inverse_transform(self, x: Tensor) -> Tensor:
        """Inverse transform input x."""
        return x


class MindSporeStandardScaler(nn.Cell):
    """Standard scalar."""
    
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None
        self.fitted = False
    
    def fit(self, x: Tensor):
        """Fit scalar."""
        assert (x.dim() == 2)
        
        scaler = StandardScaler().fit(x.asnumpy())
        mean = scaler.mean_.copy().reshape(-1)
        std = scaler.scale_.copy().reshape(-1)
        self.mean = Tensor(mean, ms.float32)
        self.std = Tensor(std, ms.float32)
        return self
    
    def construct(self, x: Tensor) -> Tensor:
        """Construct scaler on x."""
        return (x - self.mean) / self.std
    
    def transform(self, x: Tensor) -> Tensor:
        """Transform input x."""
        return self.construct(x)
    
    def inverse_transform(self, x: Tensor) -> Tensor:
        """Inverse transform input x."""
        return x * self.std + self.mean


class MindSporeMinMaxScaler(nn.Cell):
    """Min Max scalar."""
    
    def __init__(self, range: tuple = (0, 1)):
        super().__init__()
        self.range_lb = float(range[0])
        self.range_ub = float(range[1])
        assert (self.range_ub > self.range_lb)
        
        self.scale_ = None
        self.min_ = None
        self.fitted = False
    
    def fit(self, x: Tensor):
        """Fit scalar."""
        assert (x.dim() == 2)
        scaler = MinMaxScaler((self.range_lb, self.range_ub)).fit(x.asnumpy())
        self.scale_ = Tensor(scaler.scale_, ms.float32)
        self.min_ = Tensor(scaler.min_, ms.float32)
        self.fitted = True
        return self
    
    def construct(self, x: Tensor) -> Tensor:
        """Construct scaler on x."""
        return self.scale_ * x + self.min_
    
    def transform(self, x: Tensor) -> Tensor:
        """Transform input x."""
        return self.construct(x)
    
    def inverse_transform(self, x: Tensor) -> Tensor:
        """Inverse transform input x."""
        return (x - self.min_) / self.scale_
