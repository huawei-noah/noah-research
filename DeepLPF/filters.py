# -*- coding: utf-8 -*-
#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the MIT License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the MIT License for more details.
"""The three local parametric filters of Sec. 3.2, gathered for import.

* :class:`CubicFilter`      polynomial ("cubic-20") filter, Sec. 3.2.4, Eq. 6
* :class:`GraduatedFilter`  graduated filter, Sec. 3.2.2, Eqs. 1-3
* :class:`EllipticalFilter` elliptical filter, Sec. 3.2.3, Eq. 4

Each lives in its own module; this one re-exports them, together with the
shared pieces in :mod:`filtercommon`, so that ``filters.CubicFilter`` and the
rest keep resolving.
"""
from cubic import CubicFilter  # noqa: F401
from elliptical import EllipticalFilter  # noqa: F401
from filtercommon import (_GRID_CACHE, BinaryLayer, SignSTE,  # noqa: F401
                          _apply_gates, _coord_grid_powers, _coord_grids)
from graduated import GraduatedFilter  # noqa: F401
