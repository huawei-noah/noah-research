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

"""Safety policies and guards around agent-produced code (for example, LNR)."""

from .execution_policy import (
    EnsureFullRunResult,
    FULLRUN_TAIL_SNAPSHOT_NAME,
    clear_embedded_full_run_result,
    compute_fullrun_text_tail,
    drop_warning_noise_lines,
    ensure_full_execution,
    try_consume_embedded_full_run_result,
    write_embedded_full_run_result,
    write_fullrun_tail_snapshot,
)
from .leakage_detector import validate_no_leakage
from .stdout_checker import check_stdout_hostile, strip_tqdm

__all__ = [
    "EnsureFullRunResult",
    "FULLRUN_TAIL_SNAPSHOT_NAME",
    "clear_embedded_full_run_result",
    "compute_fullrun_text_tail",
    "drop_warning_noise_lines",
    "ensure_full_execution",
    "try_consume_embedded_full_run_result",
    "write_embedded_full_run_result",
    "write_fullrun_tail_snapshot",
    "validate_no_leakage",
    "check_stdout_hostile",
    "strip_tqdm",
]
