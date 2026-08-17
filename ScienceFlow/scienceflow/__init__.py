# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

"""ScienceFlow package — prefer repo-bundled deepcraft over other PYTHONPATH/site-packages copies."""

from __future__ import annotations

import sys
from pathlib import Path

# When multiple deepcraft_core installs exist (e.g. another repo on PYTHONPATH), ensure
# this repository's editable deepcraft wins so symbols like StreamHandle match ScienceFlow code.
_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent
for _rel in ("deepcraft/tool_ext", "deepcraft/agent", "deepcraft/core"):
    _root = _repo_root / _rel
    if _root.is_dir():
        _p = str(_root.resolve())
        if _p not in sys.path:
            sys.path.insert(0, _p)
