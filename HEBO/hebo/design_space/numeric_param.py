"""Numerical Parameter."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np

from .param import Parameter


class NumericPara(Parameter):
    """Parameter class: Numerical."""
    
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.lb = param_dict['lb']
        self.ub = param_dict['ub']
    
    def sample(self, num=1):
        """Sample."""
        assert (num > 0)
        return np.random.uniform(self.lb, self.ub, num)
    
    def transform(self, x):
        """Transform."""
        return x
    
    def inverse_transform(self, x):
        """Inverse Transform."""
        return x
    
    @property
    def is_numeric(self):
        """Is numerical?."""
        return True
    
    @property
    def opt_lb(self):
        """Lower bound."""
        return self.lb
    
    @property
    def opt_ub(self):
        """Upper bound."""
        return self.ub
    
    @property
    def is_discrete(self):
        """Is discrete?."""
        return False
    
    @property
    def is_discrete_after_transform(self):
        """Is discrete after transform?."""
        return False
