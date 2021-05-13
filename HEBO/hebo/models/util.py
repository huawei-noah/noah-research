"""Utils for surrogate model."""

# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import hebo.mindspore as hebo_ms
import numpy as np
from mindspore import Tensor


def filter_nan(x_: Tensor, xe_: Tensor, y_: Tensor,
               keep_rule='any') -> (Tensor, Tensor, Tensor):
    """Filter out nan's."""
    x = x_.asnumpy() if x_ is not None else np.zeros((y_.shape[0], 0))
    xe = xe_.asnumpy() if xe_ is not None else np.zeros((y_.shape[0], 0))
    y = y_.asnumpy()
    
    assert np.isfinite(x).all()
    assert np.isfinite(xe).all()
    assert np.isfinite(y).any(), "No valid data in the dataset"
    
    if keep_rule == 'any':
        valid_id = np.isfinite(y).any(axis=1)
    else:
        valid_id = np.isfinite(y).all(axis=1)
    x_filtered = hebo_ms.from_numpy(x[valid_id]) if x is not None else None
    xe_filtered = hebo_ms.from_numpy(xe[valid_id]) if xe is not None else None
    y_filtered = hebo_ms.from_numpy(y[valid_id])
    return x_filtered, xe_filtered, y_filtered
