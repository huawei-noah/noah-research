"""BoolParam Configuration."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
from sklearn.preprocessing import LabelEncoder
from .param import Parameter


class BoolPara(Parameter):
    """Bool Parameter class."""

    def __init__(self, param):
        super().__init__(param)
        self.lb = 0
        self.ub = 1

    def sample(self, num=1):
        """Sample True or False."""
        assert(num > 0)
        return np.random.choice([True, False], num, replace=True)

    def transform(self, x):
        """Transform to float 0 or 1."""
        return x.astype(float)

    def inverse_transform(self, x):
        """Inverse transform float to true or false."""
        return x > 0.5

    @property
    def is_numeric(self):
        """Return if numeric."""
        # XXX: It's OK to view boolean as numeric value, this may reduce
        # dimensions if catecorical variables are procecessed via one-hot or
        # embedding
        return True

    @property
    def is_discrete(self):
        """Return discrete."""
        return True

    @property
    def is_discrete_after_transform(self):
        """Return true after inv transform."""
        return True

    @property
    def opt_lb(self):
        """Return lower bound."""
        return self.lb

    @property
    def opt_ub(self):
        """Return upper bound."""
        return self.ub
