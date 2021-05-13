"""Parameter configuration for power integers."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import sys
import numpy as np
from .param import Parameter


class PowIntegerPara(Parameter):
    """Power Integer Class.

    e.g. a^0, a^1, a^2 etc.
    """
    
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.base = param_dict.get('base', 10.)
        self.lb = np.log(param_dict['lb']) / np.log(self.base)
        self.ub = np.log(param_dict['ub']) / np.log(self.base)
        assert param_dict['lb'] >= 1
    
    def sample(self, num=1):
        """Sample power integer."""
        assert (num > 0)
        return (self.base ** np.random.uniform(self.lb,
                                               self.ub, num)).round().astype(int)
    
    def transform(self, x):
        """Transform to numerically stable rep."""
        return np.log(x) / np.log(self.base)
    
    def inverse_transform(self, x):
        """Inv transform to original rep."""
        return (self.base ** x).round().astype(int)
    
    @property
    def is_numeric(self):
        """Return true."""
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
        """Return True."""
        return True
    
    @property
    def is_discrete_after_transform(self):
        """Return False."""
        return False
