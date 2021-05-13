"""Integer Parameter."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np

from .param import Parameter


class IntegerPara(Parameter):
    """Integer Parameter Class."""
    
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.lb = round(param_dict['lb'])
        self.ub = round(param_dict['ub'])
    
    def sample(self, num=1):
        """Sample."""
        assert (num > 0)
        return np.random.randint(self.lb, self.ub + 1, num)
    
    def transform(self, x):
        """Transform."""
        return x.astype(float)
    
    def inverse_transform(self, x):
        """Inverse Transform."""
        return x.round().astype(int)
    
    @property
    def is_numeric(self):
        """Is numeric."""
        return True
    
    @property
    def opt_lb(self):
        """Lower bound."""
        return float(self.lb)
    
    @property
    def opt_ub(self):
        """Upper bound."""
        return float(self.ub)
    
    @property
    def is_discrete(self):
        """Is discrete."""
        return True
    
    @property
    def is_discrete_after_transform(self):
        """Is discrete after transform."""
        return True
