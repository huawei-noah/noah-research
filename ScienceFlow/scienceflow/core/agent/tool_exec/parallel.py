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

"""Parallel tool bundles: compose readonly + bash mixins."""

from __future__ import annotations

from scienceflow.core.agent.tool_exec.parallel_bash import ParallelBashBundleMixin
from scienceflow.core.agent.tool_exec.parallel_readonly import ParallelReadonlyMixin


class ParallelToolExecMixin(ParallelReadonlyMixin, ParallelBashBundleMixin):
    """Classify and run multi-tool bundles in parallel where safe."""
