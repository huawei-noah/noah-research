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

"""Global merge finalization for LNR runs."""

from scienceflow.solver.lnr.global_merge.fallback import metric_float
from scienceflow.solver.lnr.global_merge.final_artifacts import (
    materialize_best_stage_final,
)
from scienceflow.solver.lnr.global_merge.runner import run_global_merge

__all__ = ["materialize_best_stage_final", "metric_float", "run_global_merge"]
