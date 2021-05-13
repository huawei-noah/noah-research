# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

from hebo.mindspore import randn, isfinite
from hebo.models.util import filter_nan
import numpy as np
import pytest
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)) + '/../')


def test_filter():
    x = randn(10, 1)
    y = randn(10, 3)
    y.asnumpy()[0] *= np.nan
    y.asnumpy()[1, 1] *= np.nan
    
    xf, xef, yf = filter_nan(x, None, y)
    assert (yf.shape[0] == 9)
    
    xf, xef, yf = filter_nan(x, None, y, keep_rule='all')
    assert (yf.shape[0] == 8)
    assert (isfinite(yf).all())
    
    xf, xef, yf = filter_nan(x, None, y, keep_rule='any')
