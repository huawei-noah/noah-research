"""Power Parameter."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np

from .param import Parameter


class PowPara(Parameter):
    """Power Parameter."""
    
    def __init__(self, param_dict):
        super().__init__(param_dict)
        self.base = param_dict.get('base', 10.)
        self.lb = np.log(param_dict['lb']) / np.log(self.base)
        self.ub = np.log(param_dict['ub']) / np.log(self.base)
    
    def sample(self, num=1):
        """Sample."""
        assert (num > 0)
        return self.base ** np.random.uniform(self.lb, self.ub, num)
    
    def transform(self, x):
        """Transform."""
        return np.log(x) / np.log(self.base)
    
    def inverse_transform(self, x):
        """Inverse Transform."""
        return self.base ** x
    
    @property
    def is_numeric(self):
        """Is Numeric."""
        return True
    
    @property
    def opt_lb(self):
        """Opt lower bound."""
        return self.lb
    
    @property
    def opt_ub(self):
        """Opt upper bound."""
        return self.ub
    
    @property
    def is_discrete(self):
        """Check is discrete."""
        return False
    
    @property
    def is_discrete_after_transform(self):
        """Check if discrete after transform."""
        return False
