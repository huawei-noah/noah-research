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

"""Unified writable root under ``/tmp`` for tests that need execution space.

Default: ``/tmp/scienceflow_tests``. Override with env ``SCIENCEFLOW_TEST_TMP_ROOT`` (absolute path).

All ``tmp_path`` / ``tempfile`` helpers in tests should use :data:`TEST_WORKSPACE_ROOT`
(see ``tests/conftest.py``) so nothing is written next to the repo.
"""

import os
from pathlib import Path

TEST_WORKSPACE_ROOT = Path(
    os.environ.get("SCIENCEFLOW_TEST_TMP_ROOT", "/tmp/scienceflow_tests"),
)
